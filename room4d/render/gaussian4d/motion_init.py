"""Build/load a motion-init bundle into a DynamicScene.

This is the swappable ``MotionInitializer`` seam, with two backends that return
the same ``(DynamicScene, {"train"/"val": [BorrowedCamera]})`` shape:

- :func:`load_borrowed_init` loads a bundle dumped by
  ``scripts/borrow_som_motion_init.py`` (SoM's track-based init run in the
  reference env — the original stopgap).
- :class:`NativeMotionInitializer` builds the same scene natively from a
  DyCheck-format sequence: room4d lifts the 2D tracks to 3D (PR-A,
  ``room4d.lifting.native_track_lift``) and runs its own clustering + Procrustes
  + Gaussian placement (``room4d.render.gaussian4d.procrustes_init``). The only
  borrowed input left is the 2D point tracker output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from room4d.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
from room4d.render.gaussian4d.gaussian_field import GaussianField
from room4d.render.gaussian4d.motion import MotionBases
from room4d.render.gaussian4d.rasterizer import RenderCamera

if TYPE_CHECKING:
    from room4d.lifting.geometry_provider import GeometryProvider


@dataclass
class BorrowedCamera:
    """A normalized camera from the bundle (w2c is gsplat-ready world->cam)."""

    frame_name: str
    time_id: int
    w2c: np.ndarray  # (4, 4)
    K: np.ndarray  # (3, 3)

    def render_camera(self, width: int, height: int, k_scale: float = 1.0) -> RenderCamera:
        """RenderCamera at (width, height); k_scale rescales K for a coarser factor."""
        K = self.K.astype(np.float32).copy()
        if k_scale != 1.0:
            K[:2] *= float(k_scale)
        return RenderCamera(viewmat=self.w2c.astype(np.float32), K=K, width=int(width), height=int(height))


def _cameras(data, split: str) -> list[BorrowedCamera]:
    w2cs, ks = data[f"{split}_w2cs"], data[f"{split}_Ks"]
    tids, names = data[f"{split}_time_ids"], data[f"{split}_frame_names"]
    return [
        BorrowedCamera(str(names[i]), int(tids[i]), w2cs[i], ks[i])
        for i in range(len(names))
    ]


def load_borrowed_init(npz_path: str, device: str = "cuda") -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
    """Load a borrowed init bundle -> (DynamicScene, {"train"/"val": cameras})."""
    data = np.load(npz_path, allow_pickle=True)

    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(data[key], dtype=torch.float32, device=device)

    fg = DynamicGaussianField(
        means=t("fg_means"),
        quats=t("fg_quats"),
        log_scales=t("fg_scales"),
        logit_opacities=t("fg_opacities").reshape(-1),
        colors_logit=t("fg_colors"),
        motion_coefs=t("fg_motion_coefs"),
    )
    bases = MotionBases(rots=t("motion_rots"), transls=t("motion_transls"))
    bg = None
    if "bg_means" in data.files:
        bg = GaussianField(
            means=t("bg_means"),
            quats=t("bg_quats"),
            log_scales=t("bg_scales"),
            logit_opacities=t("bg_opacities").reshape(-1),
            colors_logit=t("bg_colors"),
        )
    scene = DynamicScene(fg, bases, bg).to(device)
    cameras = {split: _cameras(data, split) for split in ("train", "val")}
    return scene, cameras


def _to_borrowed_cameras(records) -> list[BorrowedCamera]:
    """Adapt PR-A ``CameraRecords`` -> the ``BorrowedCamera`` list the render path expects."""
    w2cs = records.w2cs.detach().cpu().numpy()
    ks = records.Ks.detach().cpu().numpy()
    time_ids = records.time_ids.detach().cpu().numpy()
    return [
        BorrowedCamera(records.frame_names[i], int(time_ids[i]), w2cs[i], ks[i])
        for i in range(len(records.frame_names))
    ]


@dataclass
class NativeMotionInitializer:
    """Build a :class:`DynamicScene` natively from a DyCheck-format sequence.

    room4d lifts the borrowed 2D tracks to 3D (PR-A) and runs its own
    clustering + Procrustes + Gaussian placement (PR-B), producing the same
    bundle shape as :func:`load_borrowed_init`. Cameras are full-res (factor 1,
    matching the borrow), so the render-proof's ``k_scale`` handling is unchanged.

    Lifting + init run on CPU (the lift holds all full-res frames in memory and
    KMeans/kNN are CPU anyway); only the assembled scene is moved to ``device``.
    """

    dataset_root: str
    factor: int = 1
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_bases: int = 10
    depth_type: str = "depth_anything_colmap"
    camera_type: str = "refined"
    seed: int | None = None
    # Optional override: read 2D tracks from room4d's own tracker output instead
    # of the borrowed 2d_tracks dir (see lifting.native_track_lift.generate_native_tracks).
    tracks_dir: str | None = None
    # Optional geometry source (cameras + per-frame depth/rgb/fg). Defaults to the
    # borrowed DyCheck ground truth (built from dataset_root/factor/depth_type/
    # camera_type); pass a native backend (e.g. VGGT) to reconstruct geometry.
    provider: "GeometryProvider | None" = None
    # Optional in-memory all-pairs loader: pair_loader(i, j) -> (N_i, 4). When set,
    # the lift reads 2D tracks from this (the pipeline's AllPairsTracks.pair) instead
    # of disk -- no per-pair files, no dataset_root needed.
    pair_loader: "Callable[[int, int], np.ndarray] | None" = None

    def build(self, device: str = "cuda") -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
        # Lazy imports: only the native path needs the lift / sklearn machinery.
        from room4d.lifting.native_track_lift import build_native_init_inputs
        from room4d.render.gaussian4d.procrustes_init import (
            init_bg,
            init_fg_from_tracks_3d,
            init_motion_params_with_procrustes,
        )

        tracks, bg_points, cameras, _norm = build_native_init_inputs(
            self.dataset_root, factor=self.factor, num_fg=self.num_fg, num_bg=self.num_bg,
            depth_type=self.depth_type, camera_type=self.camera_type, device="cpu", seed=self.seed,
            tracks_dir=self.tracks_dir, provider=self.provider, pair_loader=self.pair_loader,
        )
        # Canonical frame = the one where the most tracks are visible.
        cano_t = int(tracks.visibles.sum(dim=0).argmax().item())
        bases, motion_coefs, tracks = init_motion_params_with_procrustes(
            tracks, self.num_bases, cano_t, seed=self.seed
        )
        fg = init_fg_from_tracks_3d(cano_t, tracks, motion_coefs)
        bg = init_bg(bg_points)
        scene = DynamicScene(fg, bases, bg).to(device)
        cams = {split: _to_borrowed_cameras(records) for split, records in cameras.items()}
        return scene, cams


def build_native_init(
    dataset_root: str,
    *,
    factor: int = 1,
    num_fg: int = 40_000,
    num_bg: int = 100_000,
    num_bases: int = 10,
    depth_type: str = "depth_anything_colmap",
    camera_type: str = "refined",
    device: str = "cuda",
    seed: int | None = None,
    tracks_dir: str | None = None,
    provider: "GeometryProvider | None" = None,
    pair_loader: "Callable[[int, int], np.ndarray] | None" = None,
) -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
    """Convenience wrapper around :class:`NativeMotionInitializer`.build."""
    return NativeMotionInitializer(
        dataset_root=dataset_root, factor=factor, num_fg=num_fg, num_bg=num_bg,
        num_bases=num_bases, depth_type=depth_type, camera_type=camera_type, seed=seed,
        tracks_dir=tracks_dir, provider=provider, pair_loader=pair_loader,
    ).build(device=device)

"""Build/load a motion-init bundle into a DynamicScene.

This is the swappable ``MotionInitializer`` seam, with two backends that return
the same ``(DynamicScene, {"train"/"val": [BorrowedCamera]})`` shape:

- :func:`load_borrowed_init` loads a bundle dumped by
  ``scripts/borrow_som_motion_init.py`` (SoM's track-based init run in the
  reference env — the original stopgap).
- :class:`NativeMotionInitializer` builds the same scene natively from a
  DyCheck-format sequence: reifyscene lifts the 2D tracks to 3D (PR-A,
  ``reifyscene.lifting.native_track_lift``) and runs its own clustering + Procrustes
  + Gaussian placement (``reifyscene.render.gaussian4d.procrustes_init``). The only
  borrowed input left is the 2D point tracker output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from reifyscene.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
from reifyscene.render.gaussian4d.gaussian_field import GaussianField
from reifyscene.render.gaussian4d.motion import MotionBases
from reifyscene.render.gaussian4d.rasterizer import RenderCamera

if TYPE_CHECKING:
    from reifyscene.lifting.geometry_provider import GeometryProvider
    from reifyscene.lifting.native_track_lift import SceneNorm


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


def _bundle_has(data, key: str) -> bool:
    """Membership check that works for both an npz ``NpzFile`` and a plain dict."""
    return (key in data.files) if hasattr(data, "files") else (key in data)


def _cameras(data, split: str) -> list[BorrowedCamera]:
    if not _bundle_has(data, f"{split}_w2cs"):
        return []  # native in-pipeline bundles may carry train only (no held-out val).
    w2cs, ks = data[f"{split}_w2cs"], data[f"{split}_Ks"]
    tids, names = data[f"{split}_time_ids"], data[f"{split}_frame_names"]
    return [
        BorrowedCamera(str(names[i]), int(tids[i]), w2cs[i], ks[i])
        for i in range(len(names))
    ]


def bundle_to_scene(data, device: str = "cuda") -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
    """Reconstruct a ``DynamicScene`` + cameras from a bundle mapping.

    ``data`` is any ``str -> array`` mapping with the borrowed-init schema -- an
    npz ``NpzFile`` (on-disk bundle) or an in-memory dict (the pipeline's
    serialized ``SceneArtifact.dynamic_bundle``). This is the inverse of
    :func:`dump_bundle`.
    """

    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(data[key]), dtype=torch.float32, device=device)

    owned = _bundle_has(data, "fg_owner")

    def owner(prefix: str) -> torch.Tensor | None:
        if not owned:
            return None
        return torch.as_tensor(np.asarray(data[f"{prefix}_owner"]).astype(np.int64), device=device)

    fg = DynamicGaussianField(
        means=t("fg_means"),
        quats=t("fg_quats"),
        log_scales=t("fg_scales"),
        logit_opacities=t("fg_opacities").reshape(-1),
        colors_logit=t("fg_colors"),
        motion_coefs=t("fg_motion_coefs"),
        owner=owner("fg"),
    )
    bases = MotionBases(rots=t("motion_rots"), transls=t("motion_transls"))
    bg = None
    if _bundle_has(data, "bg_means"):
        bg = GaussianField(
            means=t("bg_means"),
            quats=t("bg_quats"),
            log_scales=t("bg_scales"),
            logit_opacities=t("bg_opacities").reshape(-1),
            colors_logit=t("bg_colors"),
            owner=owner("bg"),
        )
    scene = DynamicScene(fg, bases, bg).to(device)
    if owned:
        # Background is stored as "" (npz string arrays cannot hold None).
        scene.owner_node_ids = [str(x) or None for x in np.asarray(data["owner_node_ids"])]
    if _bundle_has(data, "identity_classifier_w"):
        _load_identity(scene, data, device)
    cameras = {split: _cameras(data, split) for split in ("train", "val")}
    return scene, cameras


def _load_identity(scene: DynamicScene, data, device: str) -> None:
    """Attach a trained identity (both fields' vectors + classifier + class table)."""
    import torch.nn as nn

    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(data[key]), dtype=torch.float32, device=device)

    scene.fg.identity = nn.Parameter(t("fg_identity"))
    if scene.bg is not None:
        scene.bg.identity = nn.Parameter(t("bg_identity"))
    w, b = t("identity_classifier_w"), t("identity_classifier_b")
    head = nn.Linear(int(w.shape[1]), int(w.shape[0])).to(device)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    scene.identity_classifier = head
    # Background is stored as "" (npz string arrays cannot hold None).
    scene.identity_node_ids = [str(x) or None for x in np.asarray(data["identity_node_ids"])]


def dump_bundle(
    scene: DynamicScene,
    cams: dict[str, list[BorrowedCamera]],
    *,
    factor: int = 1,
    scene_norm: "SceneNorm | None" = None,
) -> dict[str, np.ndarray]:
    """Serialize a ``DynamicScene`` + cameras to a numpy bundle (torch -> numpy).

    Inverse of :func:`bundle_to_scene`: emits the same npz schema the borrowed
    init uses, so the pipeline can persist a natively-built scene into a
    ``SceneArtifact`` and reload it unchanged in the render stage. Splits with no
    cameras (e.g. a train-only in-pipeline build) are omitted. ``scene_norm``, if
    given, is recorded so held-out dataset cameras can be normalized into the
    scene frame at score time.
    """

    def n(p: torch.Tensor) -> np.ndarray:
        return p.detach().cpu().numpy()

    fg = scene.fg
    dump: dict[str, np.ndarray] = {
        "fg_means": n(fg.means),
        "fg_quats": n(fg.quats),
        "fg_scales": n(fg.log_scales),      # log-space (matches load)
        "fg_colors": n(fg.colors_logit),    # logit
        "fg_opacities": n(fg.logit_opacities),  # logit
        "fg_motion_coefs": n(fg.motion_coefs),  # raw (softmax at use)
        "motion_rots": n(scene.motion_bases.rots),      # (K, T, 6)
        "motion_transls": n(scene.motion_bases.transls),  # (K, T, 3)
        "num_bases": np.asarray(int(scene.motion_bases.num_bases)),
        "factor": np.asarray(int(factor)),
    }
    if scene_norm is not None:
        dump["scene_norm_transfm"] = np.asarray(
            scene_norm.transfm.detach().cpu().numpy() if hasattr(scene_norm.transfm, "detach")
            else scene_norm.transfm, dtype=np.float32,
        )
        dump["scene_norm_scale"] = np.asarray(float(scene_norm.scale), dtype=np.float32)
    if scene.bg is not None:
        dump.update({
            "bg_means": n(scene.bg.means),
            "bg_quats": n(scene.bg.quats),
            "bg_scales": n(scene.bg.log_scales),
            "bg_colors": n(scene.bg.colors_logit),
            "bg_opacities": n(scene.bg.logit_opacities),
        })
    if scene.has_owners:
        dump["fg_owner"] = n(fg.owner).astype(np.int32)
        if scene.bg is not None:
            dump["bg_owner"] = n(scene.bg.owner).astype(np.int32)
        dump["owner_node_ids"] = np.asarray([nid or "" for nid in scene.owner_node_ids])
    if scene.has_identity:
        dump["fg_identity"] = n(fg.identity)
        if scene.bg is not None:
            dump["bg_identity"] = n(scene.bg.identity)
        dump["identity_classifier_w"] = n(scene.identity_classifier.weight)
        dump["identity_classifier_b"] = n(scene.identity_classifier.bias)
        dump["identity_node_ids"] = np.asarray([nid or "" for nid in scene.identity_node_ids])
    for split, cam_list in cams.items():
        if not cam_list:
            continue
        dump[f"{split}_w2cs"] = np.stack([np.asarray(c.w2c, dtype=np.float32) for c in cam_list], axis=0)
        dump[f"{split}_Ks"] = np.stack([np.asarray(c.K, dtype=np.float32) for c in cam_list], axis=0)
        dump[f"{split}_time_ids"] = np.asarray([int(c.time_id) for c in cam_list], dtype=np.int64)
        dump[f"{split}_frame_names"] = np.asarray([str(c.frame_name) for c in cam_list])
    return dump


def load_borrowed_init(npz_path: str, device: str = "cuda") -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
    """Load a borrowed init bundle -> (DynamicScene, {"train"/"val": cameras})."""
    return bundle_to_scene(np.load(npz_path, allow_pickle=True), device=device)


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

    reifyscene lifts the borrowed 2D tracks to 3D (PR-A) and runs its own
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
    # Optional override: read 2D tracks from reifyscene's own tracker output instead
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
    # Optional FP2c motion gate: {"enabled", "percentile", "threshold"}. When
    # enabled, only ever-moved tracks stay foreground; static tracks -> background.
    motion_gate: dict | None = None

    def build(self, device: str = "cuda", *, return_norm: bool = False):
        """Build the scene + cameras; with ``return_norm`` also return the scene-norm.

        Returns ``(scene, cams)`` by default. ``return_norm=True`` appends the
        :class:`SceneNorm` (the world align/center + scale that places the scene in
        its normalized frame) so a caller can normalize held-out dataset cameras
        into the same frame at score time -- needed when the geometry provider has
        no ``val`` split (the in-pipeline ``ArtifactGeometryProvider``).
        """
        # Lazy imports: only the native path needs the lift / sklearn machinery.
        from reifyscene.lifting.native_track_lift import build_native_init_inputs
        from reifyscene.render.gaussian4d.procrustes_init import (
            init_bg,
            init_fg_from_tracks_3d,
            init_motion_params_with_procrustes,
        )

        tracks, bg_points, cameras, norm = build_native_init_inputs(
            self.dataset_root, factor=self.factor, num_fg=self.num_fg, num_bg=self.num_bg,
            depth_type=self.depth_type, camera_type=self.camera_type, device="cpu", seed=self.seed,
            tracks_dir=self.tracks_dir, provider=self.provider, pair_loader=self.pair_loader,
            motion_gate=self.motion_gate,
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
        return (scene, cams, norm) if return_norm else (scene, cams)


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

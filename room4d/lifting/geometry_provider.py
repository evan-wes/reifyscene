"""Render-path geometry providers — the swappable source of cameras + per-frame
depth/rgb/fg for the native motion-init path.

Today the render proof (``run_render_proof.py``) reads geometry through hardcoded
dataset loaders in :mod:`room4d.lifting.native_track_lift`. This seam puts that
behind an interface so the geometry *source* becomes swappable — the borrowed
DyCheck ground truth (:class:`DyCheckGeometryProvider`, the permanent oracle /
baseline) vs. a native reconstruction backend (VGGT, future) — scored in the same
harness against the same eval.

A provider yields exactly what the lift consumes: per-split
:class:`~room4d.lifting.native_track_lift.CameraRecords` (poses + intrinsics +
frame names) and, for the train split, a
:class:`~room4d.lifting.native_track_lift.FrameStack` (cameras + per-frame
depth/rgb/fg/valid) in the **raw** (pre-scene-norm) frame — scene normalization
stays in :func:`~room4d.lifting.native_track_lift.build_native_init_inputs`, so
the provider is purely the geometry source.

:class:`DyCheckGeometryProvider` is a thin wrapper over the existing loaders
(no logic moves), so the seam is behavior-preserving: the native path produces
byte-identical inputs whether it goes through the provider or the old direct reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from room4d.lifting.native_track_lift import CameraRecords, FrameStack


class GeometryProvider(ABC):
    """Swappable source of render-path geometry (cameras + per-frame stacks)."""

    @abstractmethod
    def cameras(self, split: str, *, device: str = "cpu") -> "CameraRecords":
        """Return a split's cameras (frame names + world→cam poses + intrinsics)."""
        raise NotImplementedError

    @abstractmethod
    def frame_stack(self, split: str, *, device: str = "cpu") -> "FrameStack":
        """Return a split's cameras + per-frame depth/rgb/fg/valid stacks (raw frame)."""
        raise NotImplementedError


class DyCheckGeometryProvider(GeometryProvider):
    """The borrowed DyCheck ground truth — dataset-shipped depth/poses/masks.

    The permanent oracle/baseline every native backend is scored against. Delegates
    to :mod:`room4d.lifting.native_track_lift`'s loaders unchanged, so swapping the
    render path onto this provider is behavior-preserving.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        factor: int = 1,
        depth_type: str = "depth_anything_colmap",
        camera_type: str = "refined",
    ) -> None:
        self.dataset_root = str(dataset_root)
        self.factor = int(factor)
        self.depth_type = depth_type
        self.camera_type = camera_type

    def _backend(self, split: str):
        # Lazy import keeps this module free of the heavy geometry deps until used.
        from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer

        return DatasetGeometryInitializer(
            {"dataset_root": self.dataset_root, "factor": self.factor, "split": split}
        )

    def cameras(self, split: str, *, device: str = "cpu") -> "CameraRecords":
        from room4d.lifting.native_track_lift import _load_cameras

        return _load_cameras(
            self._backend(split), split, device, self.dataset_root, self.factor, self.camera_type
        )

    def frame_stack(self, split: str, *, device: str = "cpu") -> "FrameStack":
        from room4d.lifting.native_track_lift import _load_frame_stack

        return _load_frame_stack(
            self._backend(split), split, device,
            self.dataset_root, self.factor, self.depth_type, self.camera_type,
        )


class ArtifactGeometryProvider(GeometryProvider):
    """Render-path geometry sourced from the pipeline's in-memory stage artifacts.

    Wraps a (pose-refined) ``GeometryArtifact`` — cameras + per-frame depth — plus
    the GT frames (via ``metadata['debug_image_paths']``) and optional foreground
    masks from perception, so the native motion-init path lifts from the
    pipeline's own reconstruction instead of a DyCheck dataset directory (the
    documented swap point). Tensors are built in the **raw** (pre-scene-norm)
    frame; normalization stays in ``build_native_init_inputs``.

    Only the ``train`` split is meaningful in-pipeline (a single processed video).
    Held-out ``val`` views are a dataset concern handled by the masked-NVS eval,
    so ``val`` is empty here.
    """

    def __init__(self, geometry_artifact, *, stream_id: str | None = None, fg_mask_by_frame: dict | None = None) -> None:
        self.geometry_artifact = geometry_artifact
        self.fg_mask_by_frame = dict(fg_mask_by_frame or {})
        frames = list(geometry_artifact.selected_frames)
        if stream_id is not None:
            frames = [fr for fr in frames if str(fr.stream_id) == str(stream_id)]
        self.train_frames = sorted(frames, key=lambda fr: (str(fr.stream_id), int(fr.frame_idx)))

    @staticmethod
    def _frame_key(fr) -> str:
        return f"{fr.stream_id}:{int(fr.frame_idx)}"

    def _state_for(self, fr):
        states = list((self.geometry_artifact.camera_states_by_stream or {}).get(fr.stream_id, []))
        if not states:
            return None
        return min(states, key=lambda s: abs(float(s.time_sec) - float(fr.timestamp_sec)))

    def _empty_cameras(self, device: str):
        import torch
        from room4d.lifting.native_track_lift import CameraRecords

        return CameraRecords(
            frame_names=[],
            time_ids=torch.zeros((0,), dtype=torch.long, device=device),
            w2cs=torch.zeros((0, 4, 4), dtype=torch.float32, device=device),
            Ks=torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
        )

    def cameras(self, split: str, *, device: str = "cpu") -> "CameraRecords":
        # Only train cameras come from the pipeline; held-out val is dataset-coupled
        # (handled by the masked-NVS eval), so it is empty here.
        if split != "train":
            return self._empty_cameras(device)
        return self.frame_stack("train", device=device).cameras

    def frame_stack(self, split: str, *, device: str = "cpu") -> "FrameStack":
        import numpy as np
        import torch
        from PIL import Image

        from room4d.common.depth import valid_depth_mask
        from room4d.lifting.native_track_lift import CameraRecords, FrameStack, _intrinsics_matrix

        if split != "train":
            raise NotImplementedError("ArtifactGeometryProvider only provides the 'train' frame stack.")

        debug = (self.geometry_artifact.metadata or {}).get("debug_image_paths", {}) or {}
        depthmaps = self.geometry_artifact.depthmaps_by_frame or {}
        names: list[str] = []
        w2cs: list[torch.Tensor] = []
        ks: list[torch.Tensor] = []
        depths: list[torch.Tensor] = []
        rgbs: list[torch.Tensor] = []
        fgs: list[torch.Tensor] = []
        valids: list[torch.Tensor] = []

        for fr in self.train_frames:
            key = self._frame_key(fr)
            state = self._state_for(fr)
            payload = depthmaps.get(key)
            path = debug.get(key)
            if state is None or payload is None or path is None:
                continue
            depth = np.squeeze(np.asarray(payload["depth"], dtype=np.float32))
            if depth.ndim != 2:
                continue
            h, w = depth.shape
            try:
                rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            if rgb.shape[:2] != (h, w):
                rgb = np.array(Image.fromarray(rgb).resize((w, h)), dtype=np.uint8)
            fg = self.fg_mask_by_frame.get(key)
            if fg is None:
                fg_arr = np.zeros((h, w), dtype=np.float32)
            else:
                fg_arr = np.asarray(fg, dtype=np.float32)
                if fg_arr.shape[:2] != (h, w):
                    fg_arr = np.array(
                        Image.fromarray((np.clip(fg_arr, 0.0, 1.0) * 255).astype(np.uint8)).resize((w, h)),
                        dtype=np.float32,
                    ) / 255.0
            twc = torch.as_tensor(np.asarray(state.pose.T_world_cam, dtype=np.float32), device=device)
            names.append(key)
            w2cs.append(torch.linalg.inv(twc))
            ks.append(_intrinsics_matrix(state.intrinsics, device))
            depths.append(torch.as_tensor(depth, device=device))
            rgbs.append(torch.as_tensor(rgb, device=device))
            fgs.append(torch.as_tensor(fg_arr, device=device))
            # Valid only where depth is usable (finite & > near) — a zero/NaN hole must not
            # be fed to the render trainer as valid supervision (was unconditional ones).
            valids.append(torch.as_tensor(valid_depth_mask(depth).astype(np.float32), device=device))

        n = len(names)
        cameras = CameraRecords(
            frame_names=names,
            time_ids=torch.arange(n, dtype=torch.long, device=device),
            w2cs=torch.stack(w2cs) if w2cs else torch.zeros((0, 4, 4), dtype=torch.float32, device=device),
            Ks=torch.stack(ks) if ks else torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
        )
        return FrameStack(
            cameras=cameras,
            depths=torch.stack(depths) if depths else torch.zeros((0, 0, 0), dtype=torch.float32, device=device),
            rgbs=torch.stack(rgbs) if rgbs else torch.zeros((0, 0, 0, 3), dtype=torch.uint8, device=device),
            fg=torch.stack(fgs) if fgs else torch.zeros((0, 0, 0), dtype=torch.float32, device=device),
            valid=torch.stack(valids) if valids else torch.zeros((0, 0, 0), dtype=torch.float32, device=device),
        )

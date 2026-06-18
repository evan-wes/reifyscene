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

"""Dataset-backed geometry initializer.

Reads pre-computed DyCheck/Nerfies-format camera poses, intrinsics, and depth
from disk and emits a standard :class:`GeometryArtifact`, so borrowed dataset
geometry flows through reifyscene's real contracts (bypassing on-the-fly SfM/VGGT).
One backend handles both NVIDIA Dynamic Scenes (e.g. Balloon1) and DyCheck
iPhone (e.g. paper-windmill): both ship the same on-disk layout::

    <root>/camera/{cam}_{time:05d}.json   # focal_length, principal_point,
                                          # image_size, orientation (world->cam R),
                                          # position (camera center in world)
    <root>/depth/{factor}x/{name}.npy     # metric depth, HxW(x1), meters
    <root>/rgb/{factor}x/{name}.png
    <root>/splits/{train,val}.json        # camera_ids / frame_names / time_ids

Camera convention (matches Shape-of-Motion's iPhone loader): the JSON stores a
world-to-camera rotation ``R`` (``orientation``) and the camera center ``p``
(``position``), so ``w2c = [R | -R p]`` and the reifyscene pose is
``T_world_cam = inv(w2c)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.depth import unproject_depth_to_world
from reifyscene.common.frame_keys import frame_key
from reifyscene.common.types import (
    CameraIntrinsics,
    CameraPose,
    FrameRef,
    TimedCameraState,
)
from reifyscene.geometry.base import GeometryInitializer
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import ImageTransform, Resize, transform_record
from reifyscene.ingest.datasets import DyCheckSequence


class DatasetGeometryInitializer(GeometryInitializer):
    """Emit a GeometryArtifact from on-disk DyCheck-format dataset poses/depth."""

    backend_name = "dataset"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg
        if not cfg.get("dataset_root"):
            raise ValueError(
                "DatasetGeometryInitializer requires cfg['dataset_root'] "
                "(path to a DyCheck-format sequence directory)."
            )
        self.dataset_root = Path(cfg["dataset_root"])
        self.sequence = DyCheckSequence(self.dataset_root)
        self.factor = int(cfg.get("factor", 1))
        self.split = str(cfg.get("split", "train"))
        self.run_id = str(cfg.get("run_id", f"dataset:{self.dataset_root.name}"))
        self.save_dir = Path(cfg.get("geometry_save_dir", "artifacts/geometry_dataset"))

    # -- the sequence, at this initializer's factor ---------------------------

    def frame_name(self, stream_id: Any, frame_idx: int) -> str:
        return self.sequence.frame_name(stream_id, frame_idx)

    def load_camera(self, frame_name: str) -> tuple[CameraIntrinsics, np.ndarray, tuple[int, int]]:
        """Return (intrinsics, T_world_cam 4x4, (width, height)) at the ``factor`` resolution."""
        return self.sequence.load_camera(frame_name, self.factor)

    def load_depth(self, frame_name: str) -> np.ndarray:
        return self.sequence.load_depth(frame_name, self.factor)

    def load_rgb(self, frame_name: str) -> np.ndarray:
        return self.sequence.load_rgb(frame_name, self.factor)

    def load_fg_mask(self, frame_name: str) -> np.ndarray:
        return self.sequence.load_fg_mask(frame_name, self.factor)

    def load_covisible(self, frame_name: str, split: str = "val") -> np.ndarray:
        return self.sequence.load_covisible(frame_name, self.factor, split)

    def split_frames(self, split: str | None = None) -> list[tuple[str, int]]:
        return self.sequence.split_frames(split or self.split)

    # -- artifact construction --------------------------------------------

    def _resolve_specs(self, frames, ingest_artifact) -> list[tuple[str, int, int]]:
        """Resolve the (stream_id, frame_idx, time_id) list to load."""
        if frames:
            source = frames
        elif ingest_artifact is not None and getattr(ingest_artifact, "keyframes", None):
            source = ingest_artifact.keyframes
        else:
            # Standalone (render-proof) mode: enumerate the configured split.
            specs = [(cam, time_id, time_id) for cam, time_id in self.split_frames()]
            cameras = self.cfg.get("cameras")
            if cameras is not None:
                allow = {str(c) for c in cameras}
                specs = [s for s in specs if s[0] in allow]
            stride = max(1, int(self.cfg.get("frame_stride", 1)))
            specs = specs[::stride]
            max_frames = self.cfg.get("max_frames")
            if max_frames is not None:
                specs = specs[: int(max_frames)]
            return specs
        return [(str(fr.stream_id), int(fr.frame_idx), int(fr.frame_idx)) for fr in source]

    def _image_transforms(
        self, depth_sizes: dict[str, tuple[int, int]], ingest_artifact
    ) -> dict[str, Any]:
        """Ingest frame -> this dataset's ``{factor}x`` pixels, per stream.

        The dataset ships its frames already scaled, so the op is a plain resize
        whose filter is unknown; the ``debug_image_paths`` images are the dataset's
        own, already in geometry pixels. Without an ingest artifact (standalone
        use) the geometry's own size is the frame size.
        """
        streams = {s.stream_id: s for s in getattr(ingest_artifact, "streams", []) or []}
        records: dict[str, Any] = {}
        for stream_id, out_size in depth_sizes.items():
            stream = streams.get(stream_id)
            source = out_size
            if stream is not None and stream.width > 0:
                source = (int(stream.width), int(stream.height))
            ops = () if source == out_size else (Resize(out_size, "unknown"),)
            records[stream_id] = transform_record(
                ImageTransform(source, ops), recorded_by=self.backend_name
            )
        return records

    def initialize(
        self,
        frames=None,
        *,
        ingest_artifact=None,
        sync_artifact=None,
    ) -> GeometryArtifact:
        specs = self._resolve_specs(frames, ingest_artifact)
        if not specs:
            raise ValueError(
                f"DatasetGeometryInitializer found no frames for split '{self.split}' "
                f"under {self.dataset_root}."
            )

        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}
        debug_image_paths: dict[str, str] = {}
        depth_sizes: dict[str, tuple[int, int]] = {}
        selected_frames: list[FrameRef] = []

        stream_fps = {
            str(s.stream_id): float(s.fps)
            for s in getattr(ingest_artifact, "streams", []) or []
            if float(getattr(s, "fps", 0) or 0) > 0
        }
        for stream_id, frame_idx, time_id in specs:
            stream_id = str(stream_id)
            frame_idx = int(frame_idx)
            name = self.frame_name(stream_id, frame_idx)
            intrinsics, t_world_cam, _wh = self.load_camera(name)
            depth = np.squeeze(np.asarray(self.load_depth(name), dtype=np.float32))
            if stream_id in stream_fps:
                time_sec = float(frame_idx) / stream_fps[stream_id]  # the ingest clock
            else:
                if not self.sequence.fps:
                    raise ValueError(f"{self.dataset_root}: no ingest clock, and the dataset "
                                     "records no frame rate")
                time_sec = float(time_id) / self.sequence.fps

            frame_ref = FrameRef(stream_id=stream_id, frame_idx=frame_idx, timestamp_sec=time_sec)
            selected_frames.append(frame_ref)
            intrinsics_by_stream[stream_id] = intrinsics
            camera_states_by_stream.setdefault(stream_id, []).append(
                TimedCameraState(
                    stream_id=stream_id,
                    time_sec=time_sec,
                    intrinsics=intrinsics,
                    pose=CameraPose(T_world_cam=t_world_cam),
                )
            )
            # Complete oracle geometry: back-project GT depth to WORLD points (VGGT
            # convention), NaN at depth holes with a confidence = validity mask, so
            # ground-plane + every point-map consumer treat it like any other backend.
            world_pts, valid = unproject_depth_to_world(depth, intrinsics, t_world_cam)
            conf = valid.astype(np.float32)
            key = frame_key(stream_id, frame_idx)
            depthmaps_by_frame[key] = {"depth": depth, "confidence": conf}
            pointmaps_by_frame[key] = {"points": world_pts, "confidence": conf}
            # Wire the GT RGB frame so scene/render seeding + debug export can find it
            # (same metadata['debug_image_paths'] contract VGGT populates).
            rgb_path = self.sequence.rgb_path(name, self.factor)
            if rgb_path.exists():
                debug_image_paths[key] = str(rgb_path.resolve())
            depth_sizes.setdefault(stream_id, (int(depth.shape[1]), int(depth.shape[0])))

        return GeometryArtifact(
            run_id=self.run_id,
            stage_name="geometry",
            save_dir=self.save_dir,
            backend_name=self.backend_name,
            selected_frames=selected_frames,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            depthmaps_by_frame=depthmaps_by_frame,
            pointmaps_by_frame=pointmaps_by_frame,
            metadata={
                "debug_image_paths": debug_image_paths,
                "image_transforms": self._image_transforms(depth_sizes, ingest_artifact),
            },
            notes=[
                f"dataset_root={self.dataset_root}",
                f"factor={self.factor}",
                f"split={self.split}",
                f"frames={len(selected_frames)}",
                "GT-oracle geometry: poses + depth + world point maps (depth holes → NaN, "
                "confidence = validity); matches the VGGT/fused point-map contract.",
            ],
        )

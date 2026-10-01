"""Artifact for calibration-mode outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, CameraIntrinsics, FrameRef, TimedCameraState, VideoStreamRef
from reifyscene.geometry.geometry_artifact import GeometryArtifact


@dataclass
class CalibrationArtifact(BaseArtifact):
    """Persist sampled-frame calibration outputs and imported COLMAP poses."""

    streams: list[VideoStreamRef] = field(default_factory=list)
    sampled_frames: list[FrameRef] = field(default_factory=list)
    sampled_image_index: dict[str, str] = field(default_factory=dict)
    intrinsics_by_stream: dict[str, CameraIntrinsics] = field(default_factory=dict)
    camera_states_by_stream: dict[str, list[TimedCameraState]] = field(default_factory=dict)
    sparse_points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    sparse_points_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    sparse_point_errors: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    connected_components: list[list[str]] = field(default_factory=list)
    reprojection_stats: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def save(self) -> None:
        ensure_dir(self.save_dir)
        sparse_path = self.save_dir / "sparse_points.npz"
        np.savez_compressed(
            sparse_path,
            xyz=np.asarray(self.sparse_points_xyz, dtype=np.float32),
            rgb=np.asarray(self.sparse_points_rgb, dtype=np.uint8),
            errors=np.asarray(self.sparse_point_errors, dtype=np.float32),
        )
        payload = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "streams": [self._serialize_value(stream) for stream in self.streams],
            "sampled_frames": [GeometryArtifact._frame_ref_to_dict(fr) for fr in self.sampled_frames],
            "sampled_image_index": dict(self.sampled_image_index),
            "intrinsics_by_stream": {stream_id: GeometryArtifact._intrinsics_to_dict(intr) for stream_id, intr in self.intrinsics_by_stream.items()},
            "camera_states_by_stream": {
                stream_id: [GeometryArtifact._camera_state_to_dict(state) for state in states]
                for stream_id, states in self.camera_states_by_stream.items()
            },
            "connected_components": [list(comp) for comp in self.connected_components],
            "reprojection_stats": dict(self.reprojection_stats),
            "notes": list(self.notes),
            "metadata": self._serialize_value(self.metadata),
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> "CalibrationArtifact":
        raw = read_json(Path(save_dir) / cls.FILE)
        npz = np.load(Path(save_dir) / "sparse_points.npz")
        return cls(
            run_id=str(raw["run_id"]),
            stage_name=str(raw["stage_name"]),
            save_dir=Path(save_dir),
            streams=[cls._deserialize_value(VideoStreamRef, item) for item in raw["streams"]],
            sampled_frames=[GeometryArtifact._frame_ref_from_dict(fr) for fr in raw["sampled_frames"]],
            sampled_image_index={str(k): str(v) for k, v in raw["sampled_image_index"].items()},
            intrinsics_by_stream={stream_id: GeometryArtifact._intrinsics_from_dict(intr) for stream_id, intr in raw["intrinsics_by_stream"].items()},
            camera_states_by_stream={
                stream_id: [GeometryArtifact._camera_state_from_dict(state) for state in states]
                for stream_id, states in raw["camera_states_by_stream"].items()
            },
            sparse_points_xyz=np.asarray(npz["xyz"], dtype=np.float32),
            sparse_points_rgb=np.asarray(npz["rgb"], dtype=np.uint8),
            sparse_point_errors=np.asarray(npz["errors"], dtype=np.float32),
            connected_components=[list(comp) for comp in raw["connected_components"]],
            reprojection_stats={str(k): float(v) for k, v in raw["reprojection_stats"].items()},
            notes=[str(v) for v in raw["notes"]],
            metadata=cls._deserialize_untyped(raw["metadata"]),
        )

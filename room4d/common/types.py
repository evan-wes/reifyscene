"""Shared dataclasses and artifact base types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir


@dataclass
class VideoStreamRef:
    """Reference to a single decoded video stream."""

    stream_id: str
    video_path: Path
    fps: float
    num_frames: int
    width: int
    height: int
    has_audio: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRef:
    """Reference to a specific frame in a video stream."""

    stream_id: str
    frame_idx: int
    timestamp_sec: float


@dataclass
class CameraIntrinsics:
    """Minimal intrinsic calibration parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "brown"
    distortion_params: Optional[np.ndarray] = None


@dataclass
class CameraPose:
    """SE(3) camera pose represented as a 4x4 matrix."""

    T_world_cam: np.ndarray


@dataclass
class TimedCameraState:
    """Camera state at a specific time."""

    stream_id: str
    time_sec: float
    intrinsics: CameraIntrinsics
    pose: CameraPose


@dataclass
class TrackObservation:
    """Single 2D point-track observation."""

    stream_id: str
    frame_idx: int
    track_id: int
    xy: np.ndarray
    visible: bool
    confidence: float


@dataclass
class Track3D:
    """Triangulated or optimized 3D track state."""

    track_id: int
    xyz: np.ndarray
    is_static: bool
    confidence: float
    obs_refs: list[tuple[str, int, int]]


@dataclass
class HumanState:
    """Human detection and fitted state for a frame."""

    human_id: str
    frame_ref: FrameRef
    bbox_xyxy: np.ndarray
    mask_rle: Optional[dict[str, Any]]
    keypoints_2d: Optional[np.ndarray]
    densepose: Optional[dict[str, Any]]
    smpl_params: Optional[dict[str, Any]]
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectState:
    """Generic object detection and lifted state for a frame."""

    object_id: str
    frame_ref: FrameRef
    category: str
    bbox_xyxy: np.ndarray
    mask_rle: Optional[dict[str, Any]]
    embedding: Optional[np.ndarray]
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """Scene-graph node record."""

    node_id: str
    node_type: str
    label: str
    confidence: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Scene-graph edge record."""

    src: str
    dst: str
    edge_type: str
    confidence: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseArtifact:
    """Base class for stage artifacts saved to disk."""

    run_id: str
    stage_name: str
    save_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifact_path(self) -> Path:
        """Return the canonical metadata file for the artifact."""
        return self.save_dir / "artifact.json"

    def save(self) -> None:
        """Persist artifact metadata to disk."""
        ensure_dir(self.save_dir)
        serializable = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": self.metadata,
        }
        write_json(self.artifact_path(), serializable)

    @classmethod
    def load(cls, save_dir: Path) -> "BaseArtifact":
        """Load an artifact metadata file from disk."""
        raw = read_json(save_dir / "artifact.json")
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
        )

    def summary(self) -> dict[str, Any]:
        """Return a concise in-memory summary."""
        return asdict(self)

"""Optimization variable containers."""

from dataclasses import dataclass


@dataclass
class CameraVariable:
    """Optimization flags for a camera stream."""

    stream_id: str
    optimize_pose: bool = True
    optimize_intrinsics: bool = False
    optimize_distortion: bool = False


@dataclass
class TimeVariable:
    """Optimization flags for per-stream timing."""

    stream_id: str
    offset_sec: float
    optimize: bool = True


@dataclass
class HumanVariable:
    """Optimization flags for a human track."""

    human_id: str
    optimize_shape: bool = False
    optimize_pose_seq: bool = True


@dataclass
class ObjectVariable:
    """Optimization flags for an object track."""

    object_id: str
    rigid: bool = True
    optimize_pose_seq: bool = True

"""Shared dataclasses and artifact base types."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import numpy as np

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir


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
    dense: Optional[dict[str, Any]] = None  # Sapiens dense outputs: {seg, depth, normals}


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

    @staticmethod
    def _is_optional(annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin is not None and type(None) in get_args(annotation)

    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "value": value.tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return {
                "__type__": "dataclass",
                "class": value.__class__.__name__,
                "value": {
                    f.name: cls._serialize_value(getattr(value, f.name))
                    for f in fields(value)
                },
            }
        if isinstance(value, tuple):
            return {"__type__": "tuple", "value": [cls._serialize_value(v) for v in value]}
        if isinstance(value, list):
            return [cls._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): cls._serialize_value(v) for k, v in value.items()}
        return value

    @classmethod
    def _deserialize_value(cls, annotation: Any, raw: Any) -> Any:
        if raw is None:
            return None
        origin = get_origin(annotation)
        args = get_args(annotation)

        if annotation is Any or annotation is None:
            return cls._deserialize_untyped(raw)

        if cls._is_optional(annotation):
            non_none = [arg for arg in args if arg is not type(None)]
            inner = non_none[0] if non_none else Any
            return cls._deserialize_value(inner, raw)

        if origin is list:
            inner = args[0] if args else Any
            return [cls._deserialize_value(inner, v) for v in raw]

        if origin is dict:
            key_type = args[0] if len(args) > 0 else Any
            val_type = args[1] if len(args) > 1 else Any
            return {
                cls._deserialize_value(key_type, k): cls._deserialize_value(val_type, v)
                for k, v in raw.items()
            }
        if origin is tuple:
            raw_values = raw.get('value', raw) if isinstance(raw, dict) and raw.get('__type__') == 'tuple' else raw
            inner_types = args if args else (Any,)
            if len(inner_types) == 2 and inner_types[1] is Ellipsis:
                return tuple(cls._deserialize_value(inner_types[0], v) for v in raw_values)
            return tuple(
                cls._deserialize_value(inner_types[min(i, len(inner_types) - 1)], v)
                for i, v in enumerate(raw_values)
            )

        if annotation is Path:
            if isinstance(raw, dict) and raw.get("__type__") == "path":
                return Path(raw["value"])
            return Path(raw)
        if annotation is np.ndarray:
            if isinstance(raw, dict) and raw.get("__type__") == "ndarray":
                return np.asarray(raw["value"], dtype=np.dtype(raw["dtype"]))
            return np.asarray(raw)
        if isinstance(annotation, type) and is_dataclass(annotation):
            payload = raw.get("value", raw) if isinstance(raw, dict) else raw
            kwargs = {}
            nested_hints = get_type_hints(annotation)
            for f in fields(annotation):
                if f.name in payload:
                    kwargs[f.name] = cls._deserialize_value(nested_hints.get(f.name, f.type), payload[f.name])
            return annotation(**kwargs)
        return raw

    @classmethod
    def _deserialize_untyped(cls, raw: Any) -> Any:
        if isinstance(raw, list):
            return [cls._deserialize_untyped(v) for v in raw]
        if isinstance(raw, dict):
            tag = raw.get("__type__")
            if tag == "path":
                return Path(raw["value"])
            if tag == "ndarray":
                return np.asarray(raw["value"], dtype=np.dtype(raw["dtype"]))
            if tag == "tuple":
                return tuple(cls._deserialize_untyped(v) for v in raw["value"])
            if tag == "dataclass":
                return {k: cls._deserialize_untyped(v) for k, v in raw.get("value", {}).items()}
            return {str(k): cls._deserialize_untyped(v) for k, v in raw.items()}
        return raw

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for the artifact."""
        payload = {}
        for f in fields(self):
            payload[f.name] = self._serialize_value(getattr(self, f.name))
        payload['save_dir'] = self._serialize_value(self.save_dir)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BaseArtifact":
        """Reconstruct an artifact from a serialized payload."""
        kwargs = {}
        type_hints = get_type_hints(cls)
        for f in fields(cls):
            annotation = type_hints.get(f.name, f.type)
            if f.name in payload:
                kwargs[f.name] = cls._deserialize_value(annotation, payload[f.name])
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        artifact = cls(**kwargs)
        artifact.save_dir = Path(artifact.save_dir)
        return artifact

    def save(self) -> None:
        """Persist artifact metadata to disk."""
        ensure_dir(self.save_dir)
        write_json(self.artifact_path(), self.to_payload())

    @classmethod
    def load(cls, save_dir: Path) -> "BaseArtifact":
        """Load an artifact metadata file from disk."""
        raw = read_json(save_dir / "artifact.json")
        artifact = cls.from_payload(raw)
        artifact.save_dir = Path(save_dir)
        return artifact

    def summary(self) -> dict[str, Any]:
        """Return a concise in-memory summary."""
        return asdict(self)

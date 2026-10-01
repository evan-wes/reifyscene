"""Shared dataclasses and artifact base types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import numpy as np

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir


class UnknownFrameRate(ValueError):
    """A time in seconds was asked of a stream whose source records no frame rate."""


@dataclass
class VideoStreamRef:
    """Reference to a single decoded video stream.

    ``fps`` is what the source records (a video container, a dataset's own record) or None;
    it is never a default. Frames are identified by their number, not by a time: a frame
    folder's numbers are the ones its file names carry (``metadata["frame_index"]``), a
    video's are ``0..num_frames-1``.
    """

    stream_id: str
    video_path: Path
    fps: float | None
    num_frames: int
    width: int
    height: int
    has_audio: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_numbers(self) -> list[int]:
        """Every frame this stream has, in order."""
        index = self.metadata.get("frame_index")
        if index is not None:
            return sorted(int(k) for k in index)
        return list(range(int(self.num_frames)))

    def seconds(self, frame_idx: int) -> float:
        """Time of a frame from the stream's start; raises when the frame rate is unknown."""
        if not self.fps:
            raise UnknownFrameRate(
                f"stream {self.stream_id!r}: its source records no frame rate, so frame "
                f"{int(frame_idx)} has no time in seconds")
        return float(frame_idx) / float(self.fps)

    def frame_ref(self, frame_idx: int) -> FrameRef:
        """A reference to one of this stream's frames (timed when the frame rate is known)."""
        return FrameRef(stream_id=str(self.stream_id), frame_idx=int(frame_idx),
                        timestamp_sec=self.seconds(frame_idx) if self.fps else None)

    def frame_refs(self) -> list[FrameRef]:
        return [self.frame_ref(i) for i in self.frame_numbers]

    def stride_frames(self, *, frames: int | None = None, seconds: float | None = None,
                      setting: str = "stride") -> int:
        """A spacing given in frames, or in seconds converted with the recorded frame rate.

        Seconds on a stream with no frame rate raise rather than assume one; giving both is
        refused as ambiguous.
        """
        if frames is not None and seconds is not None:
            raise ValueError(f"{setting}: give it in frames or in seconds, not both")
        if seconds is not None:
            if not self.fps:
                raise UnknownFrameRate(
                    f"{setting} is given in seconds, but stream {self.stream_id!r} records no "
                    "frame rate: give it in frames")
            return max(1, int(round(float(seconds) * float(self.fps))))
        return max(1, int(frames or 1))


@dataclass
class FrameRef:
    """Reference to a specific frame in a video stream.

    ``frame_idx`` is the frame's identity; ``timestamp_sec`` is its time from the stream's
    start, or None when the source records no frame rate.
    """

    stream_id: str
    frame_idx: int
    timestamp_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"stream_id": str(self.stream_id), "frame_idx": int(self.frame_idx),
                "timestamp_sec": None if self.timestamp_sec is None else float(self.timestamp_sec)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FrameRef:
        t = raw["timestamp_sec"]
        return cls(stream_id=str(raw["stream_id"]), frame_idx=int(raw["frame_idx"]),
                   timestamp_sec=None if t is None else float(t))

    @property
    def key(self) -> tuple[str, int]:
        return str(self.stream_id), int(self.frame_idx)

    def describe(self) -> str:
        """``<stream>:<frame>``, with its time when the frame rate is known (for titles)."""
        t = "" if self.timestamp_sec is None else f" | t={float(self.timestamp_sec):.3f}s"
        return f"{self.stream_id}:{int(self.frame_idx)}{t}"


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
    """Camera state at one frame of a stream.

    Identified by ``(stream_id, frame_idx)``; ``time_sec`` is that frame's time when the
    source records a frame rate, else None. Matching a state to a frame is by frame number,
    never by nearest time.
    """

    stream_id: str
    frame_idx: int
    intrinsics: CameraIntrinsics
    pose: CameraPose
    time_sec: float | None = None

    @property
    def key(self) -> tuple[str, int]:
        return str(self.stream_id), int(self.frame_idx)


def camera_state_at(states: list[TimedCameraState] | dict[str, list[TimedCameraState]],
                    frame: FrameRef) -> TimedCameraState | None:
    """The camera state of exactly ``frame`` (its stream, its frame number), or None.

    ``states`` is one stream's list or a ``stream_id -> list`` map. A state of another frame is
    never substituted: a camera that moved between the two would put the frame in the wrong
    place.
    """
    if isinstance(states, dict):
        states = states.get(str(frame.stream_id), [])
    for state in states:
        if state.key == frame.key:
            return state
    return None


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
        """Reconstruct an artifact from a serialized payload.

        ``to_payload`` writes every dataclass field, so a missing field means an old
        or truncated artifact: it raises instead of being filled with a default.
        """
        kwargs = {}
        type_hints = get_type_hints(cls)
        for f in fields(cls):
            if f.name not in payload:
                raise KeyError(
                    f"{cls.__name__} artifact payload is missing field {f.name!r} "
                    "(old or truncated artifact; re-run the stage that writes it)"
                )
            annotation = type_hints.get(f.name, f.type)
            kwargs[f.name] = cls._deserialize_value(annotation, payload[f.name])
        artifact = cls(**kwargs)
        artifact.save_dir = Path(artifact.save_dir)
        return artifact

    def attach_manifest(self, manifest: Any) -> None:
        """Record how this stage resolved, so its fidelity is readable downstream.

        Every stage owes this, not only the ones that choose between backends: the
        guarantee that matters to a consumer is about the artifact it is holding, and a
        stage that stamps nothing is indistinguishable from one that ran perfectly.
        """
        self.metadata['capability'] = manifest.to_dict()

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

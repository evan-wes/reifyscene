"""What each stage's output depends on: the fingerprint a resume checks.

A resume loads earlier stages from a prior run instead of re-running them, which is
only sound when a loaded artifact is exactly what this invocation would produce. So
every stage declares a :class:`StageContract`:

- ``settings``: the config paths its output depends on, from ANY section (segment
  reads ``objects.*``, tracking reads ``lifting.*``). A path to a mapping covers the
  whole subtree; a leaf carries the default the code uses when the key is absent, so
  an absent key and an explicit default fingerprint the same. ``env:NAME`` is an
  environment variable. Debug, device and machine-only keys are excluded
  (:data:`ALWAYS_EXCLUDED`, plus per-stage ``exclude`` / ``exclude_when``).
- ``version``: bump it whenever the stage's output changes for the same settings and
  input -- a new algorithm, OR A CHANGED CODE DEFAULT for a setting it reads (the
  fingerprint records the value the config gives, not what the code falls back to).
- ``pinned``: settings the run script picks from the machine (free GPU memory) when
  they are not given explicitly. A resume takes them from the prior run instead of
  re-picking them, so a busy GPU does not change what "the same run" means.
- ``upstream``: the stages whose artifacts it actually consumes (lineage).

Every artifact records ``metadata["contract"]``: its stage, version, effective
settings, the identity of the input frames, the digests of the upstream artifacts it
was built from, and its own digest (over all of that). A loaded stage whose recorded
upstream digests differ from those of the artifacts it is loaded with is stale.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MISSING = object()

# Never part of any fingerprint: where or how verbosely a stage runs, not what it makes.
ALWAYS_EXCLUDED: tuple[str, ...] = (
    "*.device",
    "*.cpu_threads",
    "*.debug",
    "*.debug_*",
    "*.save_debug_*",
    "*.oom_retry_on_cpu",
    "*.system_report",
    "*.allow_github",
    "*.frame_cache_dir",
    "*.output_cache_dir",
)


@dataclass(frozen=True)
class StageContract:
    """What one stage's output depends on (see the module docstring)."""

    stage: str
    version: int = 1
    settings: Mapping[str, Any] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()
    exclude_when: Callable[[Mapping[str, Any]], Iterable[str]] | None = None
    pinned: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()

    def excluded(self, cfg: Mapping[str, Any]) -> tuple[str, ...]:
        extra = tuple(self.exclude_when(cfg)) if self.exclude_when is not None else ()
        return ALWAYS_EXCLUDED + self.exclude + extra


# -------------------------------------------------------------- config paths

def get_path(cfg: Mapping[str, Any], path: str, default: Any = MISSING) -> Any:
    """``cfg["a"]["b"]`` for ``"a.b"``; ``default`` when any part is absent."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    """Set ``cfg["a"]["b"] = value`` for ``"a.b"``, creating mappings on the way."""
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def stable(value: Any) -> Any:
    """A JSON-stable copy: sorted mappings, lists for tuples, strings for paths."""
    if isinstance(value, Mapping):
        return {str(k): stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        return value.item()  # numpy scalar
    return value


def _prune(value: Any, prefix: str, patterns: tuple[str, ...]) -> Any:
    if not isinstance(value, Mapping):
        return value
    out = {}
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if any(fnmatch.fnmatchcase(path, p) for p in patterns):
            continue
        out[key] = _prune(child, path, patterns)
    return out


def effective_settings(stage: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The settings ``stage``'s output depends on, as this config gives them."""
    contract = contract_for(stage)
    patterns = contract.excluded(cfg)
    out: dict[str, Any] = {}
    for path, default in contract.settings.items():
        if any(fnmatch.fnmatchcase(path, p) for p in patterns):
            continue
        if path.startswith("env:"):
            value = os.environ.get(path[4:], default)
        else:
            value = get_path(cfg, path, MISSING)
            if value is MISSING:
                value = copy.deepcopy(default)
        out[path] = stable(_prune(value, path, patterns))
    return out


def recorded_setting(record: Mapping[str, Any], path: str) -> Any:
    """The value a fingerprint record holds for ``path`` (inside a subtree entry too)."""
    settings = record.get("settings") or {}
    if path in settings:
        return settings[path]
    for entry, value in settings.items():
        if path.startswith(entry + ".") and isinstance(value, Mapping):
            return get_path(value, path[len(entry) + 1:], MISSING)
    return MISSING


# -------------------------------------------------------------- input identity

def _file_signature(path: Path) -> list[Any]:
    st = path.stat()
    return [path.name, int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))]


def input_identity(video_paths: Iterable[str | Path]) -> dict[str, Any]:
    """What the run reads, independent of where it lives.

    A frame directory is identified by the frames the decoder actually reads from
    it (name, size and modification time of each), a video file by its own: moving
    or renaming the folder keeps the identity, editing a used frame changes it, and
    unrelated files in the folder (another camera's frames) do not matter.
    """
    from reifyscene.ingest.decoder import VideoDecoder

    items = []
    for idx, raw in enumerate(video_paths):
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            stream = VideoDecoder().decode(path, stream_id=f"cam_{idx:02d}")
            index = stream.metadata.get("frame_index", {})
            files = [Path(index[k]) for k in sorted(index, key=lambda k: int(k))]
            kind = "frames"
        else:
            files = [path]
            kind = "video"
        digest = hashlib.sha256(
            json.dumps([_file_signature(f) for f in files], separators=(",", ":")).encode()
        ).hexdigest()[:16]
        items.append({"kind": kind, "num_files": len(files), "digest": digest})
    return {"inputs": items}


# -------------------------------------------------------------- fingerprints

def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(stable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fingerprint(
    stage: str,
    settings: Mapping[str, Any],
    input_id: Mapping[str, Any],
    upstream: Mapping[str, str | None],
    version: int | None = None,
) -> dict[str, Any]:
    """The record an artifact carries in ``metadata["contract"]``."""
    record = {
        "stage": stage,
        "version": contract_for(stage).version if version is None else int(version),
        "settings": stable(dict(settings)),
        "input": stable(dict(input_id)),
        "upstream": {k: upstream[k] for k in sorted(upstream)},
    }
    record["digest"] = _digest(record)
    return record


def stage_fingerprint(
    stage: str,
    cfg: Mapping[str, Any],
    input_id: Mapping[str, Any],
    upstream_digests: Mapping[str, str | None],
) -> dict[str, Any]:
    """The fingerprint of ``stage`` run now, under ``cfg``, on these upstream artifacts."""
    contract = contract_for(stage)
    upstream = {u: upstream_digests.get(u) for u in contract.upstream}
    return fingerprint(stage, effective_settings(stage, cfg), input_id, upstream)


# -------------------------------------------------------------- the contracts

# The tracking frame plan: which frames get tracked. Geometry densifies cameras onto
# them, and dyn_mask / geometry select them when their frame_source is "tracking".
_TRACK_PLAN = {
    "tracking.frame_stride": 1,
    "tracking.frame_spacing_sec": None,
    "tracking.max_frames": None,
}
_TRACK_PLAN_PINNED = ("tracking.frame_stride", "tracking.max_frames")

_VGGT_LONG_ONLY = (
    "geometry.chunk_size",
    "geometry.overlap",
    "geometry.auto_tune*",
    "geometry.min_chunk_size",
    "geometry.min_resize_max_dim",
    "geometry.loop_enable",
    "geometry.use_dbow",
    "geometry.long_config",
    "geometry.delete_temp_files",
)


def _ingest_excluded(cfg):
    # With every frame a keyframe the stride and cap select nothing.
    return ("max_keyframes_per_stream", "keyframe_stride_sec") if cfg.get("keyframes_all") else ()


def _dyn_mask_excluded(cfg):
    source = str(get_path(cfg, "geometry.frame_source", "keyframes")).lower()
    return tuple(_TRACK_PLAN) if source != "tracking" else ()


def _geometry_excluded(cfg):
    backends = {
        str(get_path(cfg, key, "") or "").lower()
        for key in ("geometry.backend_name", "geometry.pose_backend", "geometry.depth_backend")
    }
    out: list[str] = []
    if "vggt-long" not in backends:
        out.extend(_VGGT_LONG_ONLY)
    if "moge" not in backends:
        out.append("env:MOGE_MODEL_ID")
    return out


def _pose_refine_excluded(cfg):
    # Disabled refinement is a pass-through: none of its other settings reach the output.
    if get_path(cfg, "pose_refine.enabled", False):
        return ()
    return tuple(f"pose_refine.{k}" for k in (cfg.get("pose_refine") or {}) if k != "enabled")


CONTRACTS: dict[str, StageContract] = {
    c.stage: c
    for c in (
        StageContract(
            "ingest",
            settings={
                "max_keyframes_per_stream": 8,
                "keyframe_stride_sec": 1.0,
                "keyframes_all": False,
                "default_fps": 30.0,
            },
            exclude_when=_ingest_excluded,
        ),
        StageContract("sync", upstream=("ingest",)),
        StageContract(
            "dyn_mask",
            settings={
                "dyn_mask": {},
                "geometry.frame_source": "keyframes",
                "geometry.frame_stride": None,
                "geometry.frame_spacing_sec": None,
                **_TRACK_PLAN,
            },
            exclude_when=_dyn_mask_excluded,
            pinned=_TRACK_PLAN_PINNED,
            upstream=("ingest", "sync"),
        ),
        StageContract(
            "geometry",
            settings={"geometry": {}, **_TRACK_PLAN, "env:MOGE_MODEL_ID": None},
            exclude=("geometry.save_depth_pngs",),
            exclude_when=_geometry_excluded,
            pinned=_TRACK_PLAN_PINNED,
            upstream=("ingest", "sync", "dyn_mask"),
        ),
        StageContract(
            "pose_refine",
            settings={"pose_refine": {}},
            exclude_when=_pose_refine_excluded,
            upstream=("ingest", "geometry"),
        ),
        StageContract(
            # Segment is configured by the objects section (detector, SAM 2, frame stride);
            # the embedder keys there belong to reid's floor rung.
            "segment",
            settings={"segment": {}, "objects": {}},
            exclude=("objects.embedder_*",),
            pinned=("objects.max_inference_side",),
            upstream=("ingest",),
        ),
        StageContract(
            "reid",
            settings={
                "reid": {},
                "objects.embedder_backend_mode": None,
                "objects.embedder_max_inference_side": None,
            },
            pinned=("objects.embedder_max_inference_side",),
            upstream=("ingest", "segment"),
        ),
        StageContract(
            "semantics", settings={"semantics": {}}, upstream=("ingest", "segment", "reid")
        ),
        # Objects runs no model: its output is a pure function of its inputs.
        StageContract("objects", upstream=("ingest", "segment", "reid", "semantics")),
        StageContract(
            # The dense ladder runs on segment's instances, so the humans section's own
            # detector / frame-selection / DensePose / SMPL keys do not reach its output.
            "humans",
            settings={"humans": {}},
            exclude=(
                "humans.frame_source",
                "humans.frame_stride",
                "humans.max_detections",
                "humans.densepose_*",
                "humans.smpl_*",
                "humans.backend_mode",
            ),
            pinned=("humans.max_inference_side", "humans.pose_max_inference_side"),
            upstream=("ingest", "segment", "reid"),
        ),
        StageContract(
            "tracking",
            settings={"tracking": {}, "lifting": {}},
            pinned=(
                "tracking.frame_stride",
                "tracking.max_frames",
                "tracking.resize_max_dim",
                "tracking.window_size",
                "tracking.window_overlap",
            ),
            upstream=("ingest", "pose_refine"),
        ),
        StageContract("graph", upstream=("pose_refine", "tracking", "humans", "objects")),
        StageContract(
            # The solver reads the ``optimizer`` section; ``optimization`` holds only
            # debug-viewer switches.
            "optimization",
            settings={"optimizer": {}},
            upstream=("sync", "pose_refine", "tracking", "humans", "objects", "graph"),
        ),
        StageContract(
            "scene",
            settings={"scene": {}, "env:TAPNET_CKPT_DIR": None},
            # timeline_source is only a metadata label; the human_* keys are legacy and
            # ignored by the assembler.
            exclude=("scene.timeline_source", "scene.human_*"),
            upstream=("pose_refine", "segment"),
        ),
        StageContract(
            # The run script's render presets (width, height, point_limit, ...) are legacy
            # keys the Gaussian backend never reads.
            "render",
            settings={"render": {}, "evaluation.masked_nvs": {}, "geometry.backend_name": "vggt"},
            exclude=(
                "render.width",
                "render.height",
                "render.point_limit",
                "render.frame_stride",
                "render.max_frames_per_path",
                "render.enabled_paths",
                "render.render_scale",
            ),
            upstream=("scene",),
        ),
    )
}


def contract_for(stage: str) -> StageContract:
    """The stage's contract; a stage outside the pipeline order fingerprints its own section."""
    return CONTRACTS.get(stage) or StageContract(stage, settings={stage: {}})

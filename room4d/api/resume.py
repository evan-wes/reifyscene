"""Resume helpers and compatibility checks for persisted stage artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from room4d.api.stages import STAGE_ORDER, StageSpec
from room4d.common.io import read_json
from room4d.common.paths import ensure_dir

VOLATILE_STAGE_KEYS: dict[str, set[str]] = {
    'geometry': {'device', 'system_report', 'frame_cache_dir', 'output_cache_dir', 'cpu_threads', 'debug_shapes', 'debug_max_points'},
    'tracking': {'device', 'cpu_threads', 'allow_github', 'debug', 'debug_max_points'},
    'humans': {'device'},
    'objects': {'device'},
    'render': set(),
}


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_stable_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_stable_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _video_signatures(video_paths: list[str]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for raw_path in video_paths:
        path = Path(raw_path).expanduser().resolve()
        stat = path.stat()
        signatures.append({
            'path': str(path),
            'name': path.name,
            'size_bytes': int(stat.st_size),
            'mtime_ns': int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))),
        })
    return signatures


def _stage_cfg_signature(cfg: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name == 'ingest':
        return {
            'max_keyframes_per_stream': int(cfg.get('max_keyframes_per_stream', 8)),
            'keyframe_stride_sec': float(cfg.get('keyframe_stride_sec', 1.0)),
        }
    stage_cfg = dict(cfg.get(stage_name, {}))
    for key in VOLATILE_STAGE_KEYS.get(stage_name, set()):
        stage_cfg.pop(key, None)
    return _stable_jsonable(stage_cfg)


def stage_compatibility_payload(cfg: dict[str, Any], stage_name: str, video_paths: list[str]) -> dict[str, Any]:
    """Build the current invocation signature for a stage."""
    return {
        'stage_name': str(stage_name),
        'video_signatures': _video_signatures(video_paths),
        'stage_config': _stage_cfg_signature(cfg, stage_name),
    }


def apply_stage_compatibility_metadata(artifact: Any, cfg: dict[str, Any], stage_name: str, video_paths: list[str]) -> None:
    """Stamp compatibility metadata onto an artifact before it is saved."""
    compat = stage_compatibility_payload(cfg, stage_name, video_paths)
    artifact.metadata = dict(getattr(artifact, 'metadata', {}) or {})
    artifact.metadata['compatibility'] = compat


def validate_loaded_artifact_compatibility(artifact: Any, cfg: dict[str, Any], stage_name: str, video_paths: list[str]) -> None:
    """Validate a loaded artifact against the current invocation."""
    expected = stage_compatibility_payload(cfg, stage_name, video_paths)
    actual = dict(getattr(artifact, 'metadata', {}) or {}).get('compatibility')
    if actual is None:
        raise ValueError(
            f"Artifact for stage '{stage_name}' is missing compatibility metadata and cannot be safely resumed."
        )
    if _stable_jsonable(actual) != _stable_jsonable(expected):
        raise ValueError(
            f"Stage '{stage_name}' from prior run is incompatible with the current invocation.\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"actual={json.dumps(_stable_jsonable(actual), sort_keys=True)}"
        )


def copy_stage_into_run(prior_run_dir: str | Path, current_run_dir: str | Path, spec: StageSpec) -> Path:
    """Copy one stage directory from a prior run into the current run."""
    src = Path(prior_run_dir) / spec.dir_name
    dst = Path(current_run_dir) / spec.dir_name
    if not src.exists():
        raise FileNotFoundError(f"Prior run does not contain required stage directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def materialize_stage_from_prior(prior_run_dir: str | Path, current_run_dir: str | Path, spec: StageSpec, cfg: dict[str, Any], video_paths: list[str]):
    """Copy, load, validate, and resave a stage artifact into the current run."""
    copied_dir = copy_stage_into_run(prior_run_dir, current_run_dir, spec)
    artifact = spec.load_fn(copied_dir)
    validate_loaded_artifact_compatibility(artifact, cfg, spec.name, video_paths)
    artifact.save_dir = copied_dir
    artifact.save()
    return artifact


def ensure_start_stage(stage_name: str) -> None:
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{stage_name}'. Expected one of: {', '.join(STAGE_ORDER)}")


def ensure_stop_stage(stage_name: str) -> None:
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{stage_name}'. Expected one of: {', '.join(STAGE_ORDER)}")


def resolve_stage_window(start_stage: str, stop_stage: str) -> tuple[int, int]:
    """Resolve and validate an inclusive ``[start, stop]`` stage window.

    Returns the ``(start_idx, stop_idx)`` pair into ``STAGE_ORDER``. Stages
    before ``start`` are loaded from a prior run; stages in ``[start, stop]``
    run; stages after ``stop`` are skipped. Raises ``ValueError`` if either
    name is unknown or if ``stop_stage`` precedes ``start_stage``.
    """
    ensure_start_stage(start_stage)
    ensure_stop_stage(stop_stage)
    start_idx = STAGE_ORDER.index(start_stage)
    stop_idx = STAGE_ORDER.index(stop_stage)
    if stop_idx < start_idx:
        raise ValueError(
            f"--stop-stage '{stop_stage}' precedes --start-stage '{start_stage}'; "
            f"the stage window must run forward through: {', '.join(STAGE_ORDER)}."
        )
    return start_idx, stop_idx

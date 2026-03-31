"""Deferred debug export utilities.

Debug viewers and other heavy artifacts are intentionally generated only
after the main pipeline stages have finished, so they do not add load while
geometry/tracking/perception are still running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room4d.common.io import read_yaml, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import FrameRef
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.optimize.optimization_artifact import OptimizationArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.tracking.track_artifact import TrackArtifact


def load_debug_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg = read_yaml(path) or {}
    cfg.setdefault('enabled', True)
    cfg.setdefault('shared_rgb_frames', {'enabled': True, 'dir_name': 'debug_rgb_frames'})
    cfg.setdefault('geometry', {})
    cfg.setdefault('tracking', {})
    cfg.setdefault('objects', {})
    cfg.setdefault('humans', {})
    cfg.setdefault('optimization', {})
    return cfg


def _frame_key(frame_ref: FrameRef) -> str:
    return f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}"


def plan_debug_frames(pipeline: Any, ingest_artifact: Any, debug_cfg: dict[str, Any]) -> list[FrameRef]:
    if not debug_cfg.get('enabled', False):
        return []
    refs: dict[tuple[str, int], FrameRef] = {}

    geom_cfg = debug_cfg.get('geometry', {})
    if geom_cfg.get('enabled', True):
        for fr in getattr(ingest_artifact, 'keyframes', []) or []:
            refs[(fr.stream_id, int(fr.frame_idx))] = fr

    needs_tracking_frames = any([
        debug_cfg.get('tracking', {}).get('enabled', True),
        debug_cfg.get('objects', {}).get('enabled', True),
        debug_cfg.get('humans', {}).get('enabled', True),
        debug_cfg.get('optimization', {}).get('enabled', True),
    ])
    if needs_tracking_frames:
        for stream in ingest_artifact.streams:
            for fr in pipeline.tracker.plan_frame_refs(stream):
                refs[(fr.stream_id, int(fr.frame_idx))] = fr

    return [refs[key] for key in sorted(refs.keys())]


def materialize_shared_debug_frames(
    pipeline: Any,
    ingest_artifact: Any,
    debug_cfg: dict[str, Any],
) -> dict[str, str]:
    if not debug_cfg.get('enabled', False):
        return {}
    shared_cfg = debug_cfg.get('shared_rgb_frames', {})
    if not shared_cfg.get('enabled', True):
        return {}

    frame_refs = plan_debug_frames(pipeline, ingest_artifact, debug_cfg)
    if not frame_refs:
        return {}

    root = ensure_dir(Path(pipeline.cfg['run_dir']) / str(shared_cfg.get('dir_name', 'debug_rgb_frames')))
    stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
    index: dict[str, str] = {}
    for fr in frame_refs:
        key = _frame_key(fr)
        rel = Path(fr.stream_id) / f"frame_{int(fr.frame_idx):06d}.png"
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            frame = pipeline.decoder.get_frame(stream_lookup[fr.stream_id], fr.frame_idx)
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(dst)
        index[key] = str(dst)

    write_json(root / 'index.json', {'frame_index': index})
    return index


def export_debug_artifacts(run_dir: str | Path, debug_cfg: dict[str, Any], shared_image_paths: dict[str, str]) -> None:
    if not debug_cfg.get('enabled', False):
        return
    run_dir = Path(run_dir)

    geom_cfg = debug_cfg.get('geometry', {})
    if geom_cfg.get('enabled', True) and (run_dir / 'geometry' / 'artifact.json').exists():
        artifact = GeometryArtifact.load(run_dir / 'geometry')
        artifact.metadata.update({
            'save_depth_pngs': bool(geom_cfg.get('depth_pngs', True)),
            'save_debug_assets': bool(geom_cfg.get('debug_assets', True)),
            'save_debug_html': bool(geom_cfg.get('viewer', True)),
            'debug_max_points': int(geom_cfg.get('max_points', 5000)),
            'debug_image_paths': dict(shared_image_paths),
        })
        artifact.save()
        del artifact

    trk_cfg = debug_cfg.get('tracking', {})
    if trk_cfg.get('enabled', True) and (run_dir / 'tracking' / 'artifact.json').exists():
        artifact = TrackArtifact.load(run_dir / 'tracking')
        artifact.metadata.update({
            'save_debug_assets': bool(trk_cfg.get('debug_assets', True)),
            'save_debug_html': bool(trk_cfg.get('viewer', True)),
            'debug_max_points': int(trk_cfg.get('max_points', 5000)),
            'debug_max_trails': int(trk_cfg.get('max_trails', 12)),
            'debug_trail_window': int(trk_cfg.get('trail_window', 12)),
            'debug_image_paths': dict(shared_image_paths),
        })
        artifact.save()
        del artifact

    obj_cfg = debug_cfg.get('objects', {})
    if obj_cfg.get('enabled', True) and (run_dir / 'objects' / 'artifact.json').exists():
        artifact = ObjectArtifact.load(run_dir / 'objects')
        artifact.metadata.update({
            'save_debug_assets': bool(obj_cfg.get('debug_assets', True)),
            'save_debug_html': bool(obj_cfg.get('viewer', True)),
            'debug_image_paths': dict(shared_image_paths),
        })
        artifact.save()
        del artifact

    hum_cfg = debug_cfg.get('humans', {})
    if hum_cfg.get('enabled', True) and (run_dir / 'humans' / 'artifact.json').exists():
        artifact = HumanArtifact.load(run_dir / 'humans')
        artifact.metadata.update({
            'save_debug_assets': bool(hum_cfg.get('debug_assets', True)),
            'save_debug_html': bool(hum_cfg.get('viewer', True)),
            'debug_image_paths': dict(shared_image_paths),
        })
        artifact.save()
        del artifact

    opt_cfg = debug_cfg.get('optimization', {})
    if opt_cfg.get('enabled', True) and (run_dir / 'optimization' / 'artifact.json').exists():
        artifact = OptimizationArtifact.load(run_dir / 'optimization')
        track_artifact = TrackArtifact.load(run_dir / 'tracking') if (run_dir / 'tracking' / 'artifact.json').exists() else None
        geometry_artifact = GeometryArtifact.load(run_dir / 'geometry') if (run_dir / 'geometry' / 'artifact.json').exists() else None
        if track_artifact is not None:
            track_artifact.metadata['debug_image_paths'] = dict(shared_image_paths)
        if geometry_artifact is not None:
            geometry_artifact.metadata['debug_image_paths'] = dict(shared_image_paths)
        artifact.metadata.update({
            'save_refinement_viewer': bool(opt_cfg.get('viewer', True)),
            'debug_max_points': int(opt_cfg.get('max_points', 5000)),
            'debug_max_trails': int(opt_cfg.get('max_trails', 12)),
            'debug_trail_window': int(opt_cfg.get('trail_window', 12)),
        })
        if artifact.metadata.get('save_refinement_viewer', False) and track_artifact is not None and geometry_artifact is not None:
            artifact.export_refinement_viewer(track_artifact, geometry_artifact)
        artifact.save()
        del artifact, track_artifact, geometry_artifact

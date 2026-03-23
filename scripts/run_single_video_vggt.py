#!/usr/bin/env python3
"""Run the single-video ingest -> sync -> VGGT geometry path.

This helper is intentionally narrow: it exists to validate that a single RGB
video can produce a persisted ``GeometryArtifact`` with actual per-frame camera,
depth, and point-map outputs when the real VGGT backend is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from room4d.api.pipeline import Room4DPipeline
from room4d.common.io import read_yaml


def build_config(args: argparse.Namespace) -> dict:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault('run_id', Path(args.video).stem)
    cfg['run_dir'] = str(Path(args.run_dir))
    geometry = cfg.setdefault('geometry', {})
    geometry['backend_mode'] = args.backend_mode
    if args.device is not None:
        geometry['device'] = args.device
    if args.frame_cache_dir is not None:
        geometry['frame_cache_dir'] = args.frame_cache_dir
    geometry['debug_shapes'] = bool(args.debug)
    geometry['save_depth_pngs'] = bool(args.debug)
    geometry['debug_max_points'] = int(args.debug_max_points)
    if args.max_keyframes is not None:
        cfg['max_keyframes_per_stream'] = args.max_keyframes
    if args.keyframe_stride_sec is not None:
        cfg['keyframe_stride_sec'] = args.keyframe_stride_sec
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Path to a single RGB video file.')
    parser.add_argument('--run-dir', required=True, help='Directory where artifacts will be written.')
    parser.add_argument('--config', default=None, help='Optional YAML config path.')
    parser.add_argument('--backend-mode', default='real', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--frame-cache-dir', default=None)
    parser.add_argument('--max-keyframes', type=int, default=8)
    parser.add_argument('--keyframe-stride-sec', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true', help='Enable verbose VGGT shape logging and save depth preview PNGs, camera JSON, PLYs, and a Plotly HTML viewer.')
    parser.add_argument('--debug-max-points', type=int, default=5000, help='Maximum point-cloud samples per keyframe in debug exports.')
    args = parser.parse_args()

    pipeline = Room4DPipeline(build_config(args))
    ingest = pipeline.preprocess([args.video])
    sync = pipeline.synchronize(ingest)
    geometry = pipeline.initialize_geometry(ingest, sync)

    print('Ingest summary:', ingest.summary())
    print('Sync summary:', sync.summary())
    print('Geometry summary:', geometry.summary())
    print('Geometry artifact dir:', geometry.save_dir)


if __name__ == '__main__':
    main()

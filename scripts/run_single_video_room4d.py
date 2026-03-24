#!/usr/bin/env python3
"""Run all currently implemented room4d components on a single RGB video.

This helper exercises the implemented vertical slice:

    ingest -> sync -> VGGT geometry -> CoTracker -> 3D lifting ->
    static/dynamic partition -> graph -> optimization -> render scaffold
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from rich.console import Console
from room4d.api.pipeline import Room4DPipeline
from room4d.common.io import read_yaml


def build_config(args: argparse.Namespace) -> dict:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault('run_id', Path(args.video).stem)
    cfg['run_dir'] = str(Path(args.run_dir))
    if args.max_keyframes is not None:
        cfg['max_keyframes_per_stream'] = int(args.max_keyframes)
    if args.keyframe_stride_sec is not None:
        cfg['keyframe_stride_sec'] = float(args.keyframe_stride_sec)

    geometry = cfg.setdefault('geometry', {})
    geometry['backend_mode'] = args.geometry_backend_mode
    if args.geometry_device is not None:
        geometry['device'] = args.geometry_device
    if args.frame_cache_dir is not None:
        geometry['frame_cache_dir'] = args.frame_cache_dir
    geometry['debug_shapes'] = bool(args.debug)
    geometry['save_depth_pngs'] = bool(args.debug)
    geometry['debug_max_points'] = int(args.debug_max_points)

    tracking = cfg.setdefault('tracking', {})
    tracking['backend_mode'] = args.tracking_backend_mode
    tracking['device'] = args.tracking_device or args.geometry_device or tracking.get('device', 'cpu')
    tracking['grid_size'] = int(args.tracking_grid_size)
    tracking['frame_stride'] = int(args.tracking_frame_stride)
    tracking['max_frames'] = None if args.tracking_max_frames is None else int(args.tracking_max_frames)
    tracking['resize_max_dim'] = None if args.tracking_resize_max_dim is None else int(args.tracking_resize_max_dim)
    tracking['debug'] = bool(args.debug)
    tracking['debug_max_points'] = int(args.debug_max_points)
    tracking['allow_github'] = bool(args.tracking_allow_github)

    lifting = cfg.setdefault('lifting', {})
    lifting['point_conf_threshold'] = float(args.point_conf_threshold)
    lifting['static_threshold_world'] = float(args.static_threshold_world)
    lifting['min_lifted_frames'] = int(args.min_lifted_frames)
    return cfg


def _debug_stage(console: Console, enabled: bool, label: str, *, details: str | None = None) -> float:
    if enabled:
        console.rule(f"[bold cyan]{label}")
        if details:
            console.print(details)
    return perf_counter()


def _debug_stage_done(console: Console, enabled: bool, label: str, started_at: float, *, save_dir: str | None = None) -> None:
    if not enabled:
        return
    elapsed = perf_counter() - started_at
    if save_dir is not None:
        console.print(f"[green]✓[/green] {label} complete in {elapsed:.2f}s → {save_dir}")
    else:
        console.print(f"[green]✓[/green] {label} complete in {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Path to a single RGB video file.')
    parser.add_argument('--run-dir', required=True, help='Directory where artifacts will be written.')
    parser.add_argument('--config', default=None, help='Optional YAML config path.')
    parser.add_argument('--frame-cache-dir', default=None)
    parser.add_argument('--max-keyframes', type=int, default=4)
    parser.add_argument('--keyframe-stride-sec', type=float, default=1.0)
    parser.add_argument('--geometry-backend-mode', default='real', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--geometry-device', default='cpu')
    parser.add_argument('--tracking-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--tracking-device', default=None)
    parser.add_argument('--tracking-grid-size', type=int, default=12)
    parser.add_argument('--tracking-frame-stride', type=int, default=1)
    parser.add_argument('--tracking-max-frames', type=int, default=None)
    parser.add_argument('--tracking-resize-max-dim', type=int, default=512)
    parser.add_argument('--tracking-allow-github', action='store_true', help='Allow torch.hub to fetch CoTracker from GitHub when no local checkout is present.')
    parser.add_argument('--point-conf-threshold', type=float, default=0.0)
    parser.add_argument('--static-threshold-world', type=float, default=0.05)
    parser.add_argument('--min-lifted-frames', type=int, default=2)
    parser.add_argument('--debug', action='store_true', help='Enable geometry and tracking debug exports.')
    parser.add_argument('--debug-max-points', type=int, default=5000)
    args = parser.parse_args()

    console = Console()
    if args.debug:
        console.print('[bold]room4d single-video end-to-end debug run[/bold]')
        console.print(f'video: {args.video}')
        console.print(f'run_dir: {args.run_dir}')
        console.print(f'geometry: {args.geometry_backend_mode} on {args.geometry_device}')
        console.print(f'tracking: {args.tracking_backend_mode} on {args.tracking_device or args.geometry_device}')
        console.print(f'max_keyframes: {args.max_keyframes}  keyframe_stride_sec: {args.keyframe_stride_sec}')
        console.print(f'tracking_grid_size: {args.tracking_grid_size}  resize_max_dim: {args.tracking_resize_max_dim}')

    pipeline = Room4DPipeline(build_config(args))

    outputs = {}

    stages = [
        ("Stage 1/7 · Ingest / preprocess", "ingest", lambda: pipeline.preprocess([args.video])),
        ("Stage 2/7 · Temporal sync", "sync", lambda: pipeline.synchronize(outputs["ingest"])),
        ("Stage 3/7 · VGGT geometry", "geometry", lambda: pipeline.initialize_geometry(outputs["ingest"], outputs["sync"])),
        ("Stage 4/7 · Tracking + lifting + static/dynamic", "tracks", lambda: pipeline.track(outputs["ingest"], geometry_artifact=outputs["geometry"])),
        ("Stage 5/7 · Scene graph", "graph", lambda: pipeline.build_graph(outputs["geometry"], outputs["tracks"])),
        ("Stage 6/7 · Optimization", "optimization", lambda: pipeline.optimize(outputs["sync"], outputs["geometry"], outputs["tracks"], graph_artifact=outputs["graph"])),
        ("Stage 7/7 · Render scaffold", "render", lambda: pipeline.fit_renderer(outputs["optimization"])),
    ]

    for label, key, fn in stages:
        started = _debug_stage(console, args.debug, label)
        artifact = fn()
        outputs[key] = artifact
        _debug_stage_done(console, args.debug, label.split('·', 1)[1].strip(), started, save_dir=str(artifact.save_dir))

    for name, artifact in outputs.items():
        print(f'{name}: {artifact.summary()}')
        print(f'  save_dir: {artifact.save_dir}')


if __name__ == '__main__':
    main()

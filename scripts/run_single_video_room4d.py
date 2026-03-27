#!/usr/bin/env python3
"""Run all currently implemented room4d components on a single RGB video.

This helper exercises the implemented vertical slice:

    ingest -> sync -> VGGT geometry -> CoTracker -> 3D lifting ->
    static/dynamic partition -> humans -> objects -> graph -> optimization -> render scaffold
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.table import Table
from room4d.api.pipeline import Room4DPipeline
from room4d.common.io import read_yaml
from room4d.common.system_report import collect_system_report, recommend_tracking_profile, save_system_report


def _coalesce(explicit, profile_value, fallback):
    if explicit is not None:
        return explicit
    if profile_value is not None:
        return profile_value
    return fallback


def build_config(args: argparse.Namespace, *, system_report: dict | None = None) -> tuple[dict, dict[str, object]]:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault('run_id', Path(args.video).stem)
    cfg['run_dir'] = str(Path(args.run_dir))
    if args.max_keyframes is not None:
        cfg['max_keyframes_per_stream'] = int(args.max_keyframes)
    if args.keyframe_stride_sec is not None:
        cfg['keyframe_stride_sec'] = float(args.keyframe_stride_sec)

    geometry = cfg.setdefault('geometry', {})
    geometry['backend_name'] = args.geometry_backend_name
    geometry['backend_mode'] = args.geometry_backend_mode
    if args.geometry_device is not None:
        geometry['device'] = args.geometry_device
    if args.frame_cache_dir is not None:
        geometry['frame_cache_dir'] = args.frame_cache_dir
    if args.vggt_long_output_cache_dir is not None:
        geometry['output_cache_dir'] = args.vggt_long_output_cache_dir
    geometry['chunk_size'] = int(args.vggt_long_chunk_size)
    geometry['overlap'] = int(args.vggt_long_overlap)
    geometry['loop_enable'] = bool(args.vggt_long_loop_enable)
    geometry['use_dbow'] = bool(args.vggt_long_use_dbow)
    geometry['delete_temp_files'] = bool(args.vggt_long_delete_temp_files)
    geometry['debug_shapes'] = bool(args.debug)
    geometry['save_depth_pngs'] = bool(args.debug)
    geometry['debug_max_points'] = int(args.debug_max_points)
    cpu_threads = getattr(args, 'cpu_threads', None)
    if cpu_threads is not None:
        geometry['cpu_threads'] = int(cpu_threads)

    tracking = cfg.setdefault('tracking', {})
    tracking_device = args.tracking_device or args.geometry_device or tracking.get('device', 'cpu')
    rec = {}
    if args.resource_profile != 'none':
        if args.resource_profile == 'auto':
            rec = recommend_tracking_profile(system_report or collect_system_report(), device=tracking_device)
        else:
            # Use the auto recommender and then override only the profile tag so the preset values stay in one place.
            rec = recommend_tracking_profile(system_report or collect_system_report(), device=tracking_device)
            manual_presets = {
                'lowmem': {'resize_max_dim': 256, 'frame_stride': 2, 'max_frames': 24, 'window_size': 12, 'window_overlap': 4, 'cpu_threads': 4},
                'balanced': {'resize_max_dim': 384, 'frame_stride': 2, 'max_frames': 48, 'window_size': 24, 'window_overlap': 8, 'cpu_threads': 6},
                'full': {'resize_max_dim': 512, 'frame_stride': 1, 'max_frames': 96, 'window_size': 48, 'window_overlap': 12, 'cpu_threads': 8},
            }
            rec.update(manual_presets[args.resource_profile])
            rec['profile'] = args.resource_profile

    tracking['backend_mode'] = args.tracking_backend_mode
    tracking['device'] = tracking_device
    tracking['grid_size'] = int(_coalesce(args.tracking_grid_size, None, 12))
    tracking['frame_stride'] = int(_coalesce(args.tracking_frame_stride, rec.get('frame_stride'), 1))
    tracking['max_frames'] = None if _coalesce(args.tracking_max_frames, rec.get('max_frames'), None) is None else int(_coalesce(args.tracking_max_frames, rec.get('max_frames'), None))
    tracking['resize_max_dim'] = None if _coalesce(args.tracking_resize_max_dim, rec.get('resize_max_dim'), 512) is None else int(_coalesce(args.tracking_resize_max_dim, rec.get('resize_max_dim'), 512))
    tracking['window_size'] = None if _coalesce(args.tracking_window_size, rec.get('window_size'), None) is None else int(_coalesce(args.tracking_window_size, rec.get('window_size'), None))
    tracking['window_overlap'] = int(_coalesce(args.tracking_window_overlap, rec.get('window_overlap'), 0))
    tracking['debug'] = bool(args.debug)
    tracking['debug_max_points'] = int(args.debug_max_points)
    tracking['allow_github'] = bool(args.tracking_allow_github)
    if cpu_threads is not None:
        tracking['cpu_threads'] = int(cpu_threads)
    elif rec.get('cpu_threads') is not None:
        tracking['cpu_threads'] = int(rec['cpu_threads'])
    humans = cfg.setdefault('humans', {})
    humans['backend_mode'] = args.humans_backend_mode
    humans['device'] = args.humans_device or args.tracking_device or args.geometry_device
    humans['score_threshold'] = float(args.humans_score_threshold)
    humans['max_detections'] = int(args.humans_max_detections)
    humans['debug'] = bool(args.debug)
    humans['frame_source'] = args.humans_frame_source
    humans['frame_stride'] = int(args.humans_frame_stride)
    humans['pose_backend_mode'] = args.humans_pose_backend_mode
    humans['densepose_backend_mode'] = args.humans_densepose_backend_mode
    humans['densepose_frame_stride'] = int(args.humans_densepose_frame_stride)
    humans['smpl_backend_mode'] = args.humans_smpl_backend_mode
    humans['smpl_frame_stride'] = int(args.humans_smpl_frame_stride)

    objects = cfg.setdefault('objects', {})
    objects['backend_mode'] = args.objects_backend_mode
    objects['device'] = args.objects_device or args.tracking_device or args.geometry_device
    objects['score_threshold'] = float(args.objects_score_threshold)
    objects['max_detections'] = int(args.objects_max_detections)
    objects['debug'] = bool(args.debug)
    objects['frame_source'] = args.objects_frame_source
    objects['frame_stride'] = int(args.objects_frame_stride)
    objects['embedder_backend_mode'] = args.objects_embedder_backend_mode

    lifting = cfg.setdefault('lifting', {})
    lifting['point_conf_threshold'] = float(args.point_conf_threshold)
    lifting['static_threshold_world'] = float(args.static_threshold_world)
    lifting['min_lifted_frames'] = int(args.min_lifted_frames)

    cfg.setdefault('runtime', {})['system_report'] = system_report or collect_system_report()
    return cfg, rec


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


def _print_system_report(console: Console, report: dict, recommendation: dict | None) -> None:
    table = Table(title='System preflight report')
    table.add_column('Item')
    table.add_column('Value')
    table.add_row('Platform', f"{report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})")
    table.add_row('Python', str(report['platform']['python']))
    table.add_row('Logical CPU cores', str(report['cpu'].get('logical_cores')))
    memory = report.get('memory', {})
    table.add_row('RAM total / available', f"{memory.get('total_gib', 0):.1f} GiB / {memory.get('available_gib', 0):.1f} GiB")
    table.add_row('Swap total / free', f"{memory.get('swap_total_gib', 0):.1f} GiB / {memory.get('swap_free_gib', 0):.1f} GiB")
    gpus = report.get('gpus', []) or []
    if gpus:
        gpu = max(gpus, key=lambda g: g.get('memory_free_mb', 0))
        table.add_row('GPU', f"{gpu.get('name')} (free {gpu.get('memory_free_mb', 0)/1024.0:.1f} / {gpu.get('memory_total_mb', 0)/1024.0:.1f} GiB)")
    else:
        table.add_row('GPU', 'None detected via nvidia-smi')
    if recommendation:
        table.add_row('Recommended profile', str(recommendation.get('profile')))
        table.add_row('Tracking preset', f"resize={recommendation.get('resize_max_dim')} max_frames={recommendation.get('max_frames')} window={recommendation.get('window_size')} overlap={recommendation.get('window_overlap')} stride={recommendation.get('frame_stride')}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Path to a single RGB video file.')
    parser.add_argument('--run-dir', required=True, help='Directory where artifacts will be written.')
    parser.add_argument('--config', default=None, help='Optional YAML config path.')
    parser.add_argument('--frame-cache-dir', default=None)
    parser.add_argument('--max-keyframes', type=int, default=4)
    parser.add_argument('--keyframe-stride-sec', type=float, default=1.0)
    parser.add_argument('--geometry-backend-name', default='vggt', choices=['vggt', 'vggt-long'])
    parser.add_argument('--geometry-backend-mode', default='real', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--geometry-device', default='cpu')
    parser.add_argument('--vggt-long-chunk-size', type=int, default=16)
    parser.add_argument('--vggt-long-overlap', type=int, default=4)
    parser.add_argument('--vggt-long-loop-enable', action='store_true')
    parser.add_argument('--vggt-long-use-dbow', action='store_true')
    parser.add_argument('--vggt-long-delete-temp-files', action='store_true')
    parser.add_argument('--vggt-long-output-cache-dir', default=None)
    parser.add_argument('--tracking-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--tracking-device', default=None)
    parser.add_argument('--tracking-grid-size', type=int, default=12)
    parser.add_argument('--tracking-frame-stride', type=int, default=None)
    parser.add_argument('--tracking-max-frames', type=int, default=None)
    parser.add_argument('--tracking-resize-max-dim', type=int, default=None)
    parser.add_argument('--tracking-window-size', type=int, default=None)
    parser.add_argument('--tracking-window-overlap', type=int, default=None)
    parser.add_argument('--tracking-allow-github', action='store_true', help='Allow torch.hub to fetch CoTracker from GitHub when no local checkout is present.')
    parser.add_argument('--humans-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--humans-device', default=None)
    parser.add_argument('--humans-score-threshold', type=float, default=0.5)
    parser.add_argument('--humans-max-detections', type=int, default=6)
    parser.add_argument('--humans-frame-source', default='tracking', choices=['tracking', 'keyframes'])
    parser.add_argument('--humans-frame-stride', type=int, default=1)
    parser.add_argument('--humans-pose-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--humans-densepose-backend-mode', default='auto', choices=['auto', 'heuristic'])
    parser.add_argument('--humans-densepose-frame-stride', type=int, default=1)
    parser.add_argument('--humans-smpl-backend-mode', default='auto', choices=['auto', 'heuristic'])
    parser.add_argument('--humans-smpl-frame-stride', type=int, default=1)
    parser.add_argument('--objects-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--objects-device', default=None)
    parser.add_argument('--objects-frame-source', default='tracking', choices=['tracking', 'keyframes'])
    parser.add_argument('--objects-frame-stride', type=int, default=1)
    parser.add_argument('--objects-score-threshold', type=float, default=0.5)
    parser.add_argument('--objects-max-detections', type=int, default=12)
    parser.add_argument('--objects-embedder-backend-mode', default='auto', choices=['real', 'auto', 'histogram'])
    parser.add_argument('--resource-profile', default='auto', choices=['none', 'auto', 'lowmem', 'balanced', 'full'], help='Apply a resource-aware tracking preset. Explicit tracking flags still win.')
    parser.add_argument('--cpu-threads', type=int, default=None, help='Limit CPU threads for torch/cv2 on CPU runs.')
    parser.add_argument('--preflight-only', action='store_true', help='Print/save the system report and exit.')
    parser.add_argument('--point-conf-threshold', type=float, default=0.0)
    parser.add_argument('--static-threshold-world', type=float, default=0.05)
    parser.add_argument('--min-lifted-frames', type=int, default=2)
    parser.add_argument('--debug', action='store_true', help='Enable geometry and tracking debug exports.')
    parser.add_argument('--debug-max-points', type=int, default=5000)
    args = parser.parse_args()

    console = Console()
    system_report = collect_system_report()
    recommended = None
    if args.resource_profile != 'none':
        recommended = recommend_tracking_profile(system_report, device=args.tracking_device or args.geometry_device)
        if args.resource_profile != 'auto':
            recommended['profile'] = args.resource_profile
    report_path = Path(args.run_dir) / 'preflight_system_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_system_report(report_path, {
        'report': system_report,
        'recommended_tracking_profile': recommended,
    })

    if args.debug or args.preflight_only:
        _print_system_report(console, system_report, recommended)
        console.print(f'saved preflight report: {report_path}')
    if args.preflight_only:
        return

    cfg, rec = build_config(args, system_report=system_report)
    if args.debug:
        console.print('[bold]room4d single-video end-to-end debug run[/bold]')
        console.print(f'video: {args.video}')
        console.print(f'run_dir: {args.run_dir}')
        console.print(f'geometry: {args.geometry_backend_mode} on {args.geometry_device}')
        console.print(f'tracking: {args.tracking_backend_mode} on {cfg["tracking"]["device"]}')
        console.print(f'max_keyframes: {args.max_keyframes}  keyframe_stride_sec: {args.keyframe_stride_sec}')
        console.print(
            f"tracking_grid_size: {cfg['tracking']['grid_size']}  resize_max_dim: {cfg['tracking']['resize_max_dim']} "
            f"max_frames: {cfg['tracking']['max_frames']}  frame_stride: {cfg['tracking']['frame_stride']}"
        )
        console.print(
            f"tracking_window_size: {cfg['tracking']['window_size']}  tracking_window_overlap: {cfg['tracking']['window_overlap']}  "
            f"cpu_threads: {cfg['tracking'].get('cpu_threads')}  resource_profile: {(rec or {}).get('profile', 'none')}"
        )
        console.print(
            f"humans: {cfg['humans']['backend_mode']} on {cfg['humans']['device']}  score_threshold: {cfg['humans']['score_threshold']}  "
            f"max_detections: {cfg['humans']['max_detections']}  frame_source: {cfg['humans']['frame_source']} stride: {cfg['humans']['frame_stride']}  "
            f"pose: {cfg['humans']['pose_backend_mode']}  densepose: {cfg['humans']['densepose_backend_mode']} stride: {cfg['humans']['densepose_frame_stride']}  smpl: {cfg['humans']['smpl_backend_mode']} stride: {cfg['humans']['smpl_frame_stride']}"
        )
        console.print(
            f"objects: {cfg['objects']['backend_mode']} on {cfg['objects']['device']}  score_threshold: {cfg['objects']['score_threshold']}  "
            f"max_detections: {cfg['objects']['max_detections']}  frame_source: {cfg['objects']['frame_source']} stride: {cfg['objects']['frame_stride']}  embedder: {cfg['objects']['embedder_backend_mode']}"
        )

    pipeline = Room4DPipeline(cfg)

    outputs = {}

    stages = [
        ("Stage 1/9 · Ingest / preprocess", "ingest", lambda: pipeline.preprocess([args.video])),
        ("Stage 2/9 · Temporal sync", "sync", lambda: pipeline.synchronize(outputs["ingest"])),
        ("Stage 3/9 · VGGT geometry", "geometry", lambda: pipeline.initialize_geometry(outputs["ingest"], outputs["sync"])),
        ("Stage 4/9 · Tracking + lifting + static/dynamic", "tracks", lambda: pipeline.track(outputs["ingest"], geometry_artifact=outputs["geometry"])),
        ("Stage 5/9 · Human perception", "humans", lambda: pipeline.perceive_humans(outputs["ingest"], track_artifact=outputs["tracks"])),
        ("Stage 6/9 · Object perception", "objects", lambda: pipeline.perceive_objects(outputs["ingest"], track_artifact=outputs["tracks"])),
        ("Stage 7/9 · Scene graph", "graph", lambda: pipeline.build_graph(outputs["geometry"], outputs["tracks"], human_artifact=outputs["humans"], object_artifact=outputs["objects"])),
        ("Stage 8/9 · Optimization", "optimization", lambda: pipeline.optimize(outputs["sync"], outputs["geometry"], outputs["tracks"], human_artifact=outputs["humans"], object_artifact=outputs["objects"], graph_artifact=outputs["graph"])),
        ("Stage 9/9 · Render scaffold", "render", lambda: pipeline.fit_renderer(outputs["optimization"])),
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

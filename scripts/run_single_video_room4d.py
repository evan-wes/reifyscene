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
from room4d.api.resume import ensure_start_stage, materialize_stage_from_prior
from room4d.api.stages import STAGE_ORDER, write_run_manifest
from room4d.common.io import read_yaml
from room4d.debug_exports import apply_debug_config_to_runtime_cfg, export_debug_artifacts, load_debug_config, materialize_shared_debug_frames
from room4d.common.memory import release_memory
from room4d.common.system_report import collect_system_report, recommend_tracking_profile, recommend_vggt_long_profile, save_system_report


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
    cfg['_resume_video_paths'] = [str(Path(args.video).expanduser().resolve())]
    if args.max_keyframes is not None:
        cfg['max_keyframes_per_stream'] = int(args.max_keyframes)
    if args.keyframe_stride_sec is not None:
        cfg['keyframe_stride_sec'] = float(args.keyframe_stride_sec)

    geometry = cfg.setdefault('geometry', {})
    geometry['backend_name'] = args.geometry_backend_name
    geometry['backend_mode'] = args.geometry_backend_mode
    geometry['frame_source'] = str(args.geometry_frame_source).lower()
    geometry_frame_stride = getattr(args, 'geometry_frame_stride', None)
    geometry_frame_spacing_sec = getattr(args, 'geometry_frame_spacing_sec', None)
    if geometry_frame_stride is not None:
        geometry['frame_stride'] = max(1, int(geometry_frame_stride))
    if geometry_frame_spacing_sec is not None:
        geometry['frame_spacing_sec'] = float(geometry_frame_spacing_sec)
    geometry['system_report'] = system_report or collect_system_report()
    if args.geometry_device is not None:
        geometry['device'] = args.geometry_device
    if args.frame_cache_dir is not None:
        geometry['frame_cache_dir'] = args.frame_cache_dir
    if args.geometry_resize_max_dim is not None:
        geometry['resize_max_dim'] = int(args.geometry_resize_max_dim)
    if args.vggt_long_output_cache_dir is not None:
        geometry['output_cache_dir'] = args.vggt_long_output_cache_dir
    geometry['chunk_size'] = int(args.vggt_long_chunk_size)
    geometry['overlap'] = int(args.vggt_long_overlap)
    geometry['auto_tune'] = bool(args.vggt_long_auto_tune)
    geometry['min_chunk_size'] = int(args.vggt_long_min_chunk_size)
    if args.vggt_long_min_resize_max_dim is not None:
        geometry['min_resize_max_dim'] = int(args.vggt_long_min_resize_max_dim)
    geometry['loop_enable'] = bool(args.vggt_long_loop_enable)
    geometry['use_dbow'] = bool(args.vggt_long_use_dbow)
    geometry['delete_temp_files'] = bool(args.vggt_long_delete_temp_files)
    geometry['debug_shapes'] = bool(args.debug)
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
    tracking_frame_spacing_sec = getattr(args, 'tracking_frame_spacing_sec', None)
    if tracking_frame_spacing_sec is not None:
        tracking['frame_spacing_sec'] = float(tracking_frame_spacing_sec)
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
    perception_side_presets = {
        "lowmem": {"detector": 512, "pose": 512, "embedder": 256},
        "balanced": {"detector": 768, "pose": 640, "embedder": 384},
        "full": {"detector": 1024, "pose": 768, "embedder": 512},
    }
    side_preset = perception_side_presets.get((rec or {}).get("profile", ""), perception_side_presets["lowmem"])

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
    humans.setdefault('max_inference_side', side_preset['detector'])
    humans.setdefault('pose_max_inference_side', side_preset['pose'])
    humans.setdefault('oom_retry_on_cpu', True)

    objects = cfg.setdefault('objects', {})
    objects['backend_mode'] = args.objects_backend_mode
    objects['device'] = args.objects_device or args.tracking_device or args.geometry_device
    objects['score_threshold'] = float(args.objects_score_threshold)
    objects['max_detections'] = int(args.objects_max_detections)
    objects['debug'] = bool(args.debug)
    objects['frame_source'] = args.objects_frame_source
    objects['frame_stride'] = int(args.objects_frame_stride)
    objects['embedder_backend_mode'] = args.objects_embedder_backend_mode
    objects.setdefault('max_inference_side', side_preset['detector'])
    objects.setdefault('embedder_max_inference_side', side_preset['embedder'])
    objects.setdefault('oom_retry_on_cpu', True)

    scene_cfg = cfg.setdefault('scene', {})
    if args.scene_timeline_source is not None:
        scene_cfg['timeline_source'] = str(args.scene_timeline_source).lower()
    else:
        scene_cfg.setdefault('timeline_source', str(scene_cfg.get('timeline_source', 'geometry')).lower())

    render_cfg = cfg.setdefault('render', {})
    render_presets = {
        'lowmem': {'width': 800, 'height': 450, 'point_limit': 10000, 'frame_stride': 1, 'max_frames_per_path': 32},
        'balanced': {'width': 960, 'height': 540, 'point_limit': 16000, 'frame_stride': 1, 'max_frames_per_path': 48},
        'full': {'width': 1280, 'height': 720, 'point_limit': 30000, 'frame_stride': 1, 'max_frames_per_path': None},
    }
    render_preset = render_presets.get((rec or {}).get('profile', 'lowmem'), render_presets['lowmem'])
    render_cfg.setdefault('backend_name', 'gaussian4d')
    render_cfg.setdefault('debug_triptych', True)
    render_cfg.setdefault('render_scale', 1)
    render_cfg.setdefault('width', render_preset['width'])
    render_cfg.setdefault('height', render_preset['height'])
    render_cfg.setdefault('point_limit', render_preset['point_limit'])
    render_cfg.setdefault('frame_stride', render_preset['frame_stride'])
    render_cfg.setdefault('max_frames_per_path', render_preset['max_frames_per_path'])
    # The renderer is gsplat-backed and GPU-only; there is no CPU render path.
    render_cfg.setdefault('device', 'cuda')
    render_cfg.setdefault('enabled_paths', {
        'orbit': True,
        'overhead': True,
        'replay': True,
        'follow_human': True,
        'follow_object': True,
    })

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


def _print_system_report(console: Console, report: dict, recommendation: dict | None, geometry_recommendation: dict | None = None) -> None:
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
    if geometry_recommendation:
        table.add_row('VGGT-Long seed', f"profile={geometry_recommendation.get('profile')} chunk={geometry_recommendation.get('initial_chunk_size')} overlap={geometry_recommendation.get('initial_overlap')} resize={geometry_recommendation.get('initial_resize_max_dim')}")
    console.print(table)


def _sorted_unique_frame_refs(frames):
    unique = {}
    for fr in frames or []:
        unique[(fr.stream_id, int(fr.frame_idx))] = fr
    return [unique[k] for k in sorted(unique)]


def _sample_rate_hz(frames) -> str:
    frames = _sorted_unique_frame_refs(frames)
    if len(frames) < 2:
        return "n/a"
    deltas = []
    for a, b in zip(frames[:-1], frames[1:]):
        dt = float(b.timestamp_sec) - float(a.timestamp_sec)
        if dt > 1e-6:
            deltas.append(dt)
    if not deltas:
        return "n/a"
    hz = 1.0 / (sum(deltas) / len(deltas))
    return f"{hz:.2f} Hz"


def _frame_set_from_states(states):
    frames = []
    for state in states or []:
        fr = getattr(state, 'frame_ref', None)
        if fr is not None:
            frames.append(fr)
    return _sorted_unique_frame_refs(frames)


def _render_frame_count(render_artifact) -> int:
    path_index = getattr(render_artifact, 'path_index', {}) or {}
    if not path_index:
        return 0
    return max((len(v) for v in path_index.values()), default=0)


def _build_frame_summary_rows(outputs: dict, cfg: dict) -> list[dict[str, str]]:
    geometry_frames = _sorted_unique_frame_refs(getattr(outputs.get('geometry'), 'selected_frames', []))
    tracking_frames = _sorted_unique_frame_refs(getattr(outputs.get('tracks'), 'selected_frames', []))
    human_frames = _frame_set_from_states(getattr(outputs.get('humans'), 'humans', []))
    object_frames = _frame_set_from_states(getattr(outputs.get('objects'), 'objects', []))
    scene_frames = _sorted_unique_frame_refs(getattr(outputs.get('scene'), 'frame_refs', []))
    render_count = _render_frame_count(outputs.get('render')) if outputs.get('render') is not None else 0

    frame_sets = {
        'geometry': geometry_frames,
        'tracking': tracking_frames,
        'humans': human_frames,
        'objects': object_frames,
        'scene': scene_frames,
    }
    normalized = {name: tuple((fr.stream_id, int(fr.frame_idx)) for fr in frames) for name, frames in frame_sets.items()}

    def share_comment(stage_name: str) -> str:
        current = normalized.get(stage_name, ())
        if not current:
            return 'none'
        shared = [other for other, vals in normalized.items() if other != stage_name and vals == current and vals]
        return ', '.join(shared) if shared else 'none'

    rows = [
        {
            'stage': 'Geometry',
            'source': 'geometry',
            'count': str(len(geometry_frames)),
            'rate': _sample_rate_hz(geometry_frames),
            'shares': share_comment('geometry'),
        },
        {
            'stage': 'Tracking',
            'source': 'tracking',
            'count': str(len(tracking_frames)),
            'rate': _sample_rate_hz(tracking_frames),
            'shares': share_comment('tracking'),
        },
        {
            'stage': 'Humans',
            'source': str(getattr(outputs.get('humans'), 'metadata', {}).get('frame_source', cfg.get('humans', {}).get('frame_source', 'tracking'))),
            'count': str(len(human_frames)),
            'rate': _sample_rate_hz(human_frames),
            'shares': share_comment('humans'),
        },
        {
            'stage': 'Objects',
            'source': str(getattr(outputs.get('objects'), 'metadata', {}).get('frame_source', cfg.get('objects', {}).get('frame_source', 'tracking'))),
            'count': str(len(object_frames)),
            'rate': _sample_rate_hz(object_frames),
            'shares': share_comment('objects'),
        },
        {
            'stage': 'Scene',
            'source': str(getattr(outputs.get('scene'), 'timeline', {}).get('source', cfg.get('scene', {}).get('timeline_source', 'geometry'))),
            'count': str(len(scene_frames)),
            'rate': _sample_rate_hz(scene_frames),
            'shares': share_comment('scene'),
        },
        {
            'stage': 'Render',
            'source': str(getattr(outputs.get('scene'), 'timeline', {}).get('source', cfg.get('scene', {}).get('timeline_source', 'geometry'))),
            'count': str(render_count),
            'rate': _sample_rate_hz(scene_frames) if render_count else 'n/a',
            'shares': 'scene',
        },
    ]
    return rows


def _print_frame_summary(console: Console, outputs: dict, cfg: dict) -> None:
    rows = _build_frame_summary_rows(outputs, cfg)
    table = Table(title='Frame timeline summary')
    table.add_column('Stage')
    table.add_column('Source timeline')
    table.add_column('Frame count', justify='right')
    table.add_column('Sample rate')
    table.add_column('Shares frame set with')
    for row in rows:
        table.add_row(row['stage'], row['source'], row['count'], row['rate'], row['shares'])
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Path to a single RGB video file.')
    parser.add_argument('--run-dir', required=True, help='Directory where artifacts will be written.')
    parser.add_argument('--prior-run-dir', default=None, help='Optional prior run directory to load earlier stage artifacts from.')
    parser.add_argument('--start-stage', default='ingest', choices=list(STAGE_ORDER), help='Stage to start running in the current invocation. Earlier stages are loaded from --prior-run-dir.')
    parser.add_argument('--config', default=None, help='Optional YAML config path.')
    parser.add_argument('--debug-config', default=None, help='Optional YAML debug-export config path.')
    parser.add_argument('--frame-cache-dir', default=None)
    parser.add_argument('--max-keyframes', type=int, default=4)
    parser.add_argument('--keyframe-stride-sec', type=float, default=1.0)
    parser.add_argument('--geometry-backend-name', default='vggt', choices=['vggt', 'vggt-long', 'dataset'])
    parser.add_argument('--geometry-backend-mode', default='real', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--geometry-frame-source', default='keyframes', choices=['keyframes', 'tracking', 'all'], help='Which timeline geometry should run on. Use tracking or all to process the full sequence with VGGT-Long chunking.')
    parser.add_argument('--geometry-frame-stride', type=int, default=None, help='Subsample the geometry frame set by original-frame stride after selecting the source timeline.')
    parser.add_argument('--geometry-frame-spacing-sec', type=float, default=None, help='Subsample the geometry frame set by approximate time spacing in seconds. Overrides --geometry-frame-stride when set.')
    parser.add_argument('--geometry-device', default='cpu')
    parser.add_argument('--geometry-resize-max-dim', type=int, default=None, help='Optionally downscale materialized geometry frames before backend execution.')
    parser.add_argument('--vggt-long-chunk-size', type=int, default=16)
    parser.add_argument('--vggt-long-overlap', type=int, default=4)
    parser.add_argument('--vggt-long-auto-tune', action='store_true', help='Automatically back off VGGT-Long chunk/overlap/resize settings on OOM.')
    parser.add_argument('--vggt-long-min-chunk-size', type=int, default=2)
    parser.add_argument('--vggt-long-min-resize-max-dim', type=int, default=None)
    parser.add_argument('--vggt-long-loop-enable', action='store_true')
    parser.add_argument('--vggt-long-use-dbow', action='store_true')
    parser.add_argument('--vggt-long-delete-temp-files', action='store_true')
    parser.add_argument('--vggt-long-output-cache-dir', default=None)
    parser.add_argument('--tracking-backend-mode', default='auto', choices=['real', 'auto', 'heuristic'])
    parser.add_argument('--tracking-device', default=None)
    parser.add_argument('--tracking-grid-size', type=int, default=12)
    parser.add_argument('--tracking-frame-stride', type=int, default=None)
    parser.add_argument('--tracking-frame-spacing-sec', type=float, default=None, help='Sample tracking frames by approximate time spacing in seconds. Overrides the effective tracking frame stride when set.')
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
    parser.add_argument('--scene-timeline-source', default=None, choices=['geometry', 'tracking'], help='Timeline source for scene assembly and rendering.',)
    parser.add_argument('--resource-profile', default='auto', choices=['none', 'auto', 'lowmem', 'balanced', 'full'], help='Apply a resource-aware tracking preset. Explicit tracking flags still win.')
    parser.add_argument('--cpu-threads', type=int, default=None, help='Limit CPU threads for torch/cv2 on CPU runs.')
    parser.add_argument('--preflight-only', action='store_true', help='Print/save the system report and exit.')
    parser.add_argument('--point-conf-threshold', type=float, default=0.0)
    parser.add_argument('--static-threshold-world', type=float, default=0.05)
    parser.add_argument('--min-lifted-frames', type=int, default=2)
    parser.add_argument('--debug', action='store_true', help='Enable geometry and tracking debug exports.')
    parser.add_argument('--debug-max-points', type=int, default=5000)
    args = parser.parse_args()

    ensure_start_stage(args.start_stage)
    console = Console()
    system_report = collect_system_report()
    debug_cfg = load_debug_config(args.debug_config)
    recommended = None
    geometry_recommended = None
    if args.resource_profile != 'none':
        recommended = recommend_tracking_profile(system_report, device=args.tracking_device or args.geometry_device)
        if args.resource_profile != 'auto':
            recommended['profile'] = args.resource_profile
    if args.geometry_backend_name == 'vggt-long':
        geometry_recommended = recommend_vggt_long_profile(system_report, device=args.geometry_device)
    report_path = Path(args.run_dir) / 'preflight_system_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_system_report(report_path, {
        'report': system_report,
        'recommended_tracking_profile': recommended,
        'recommended_vggt_long_profile': geometry_recommended,
    })

    if args.debug or args.preflight_only:
        _print_system_report(console, system_report, recommended, geometry_recommendation=geometry_recommended)
        console.print(f'saved preflight report: {report_path}')
    if args.preflight_only:
        return

    cfg, rec = build_config(args, system_report=system_report)
    cfg = apply_debug_config_to_runtime_cfg(cfg, debug_cfg)
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
            f"geometry_frame_source: {cfg['geometry']['frame_source']}  "
            f"geometry_resize_max_dim: {cfg['geometry'].get('resize_max_dim')}  vggt_long_auto_tune: {cfg['geometry'].get('auto_tune', False)}  "
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
        console.print(f"scene timeline source: {cfg.get('scene', {}).get('timeline_source', 'geometry')}")

    write_run_manifest(
        args.run_dir,
        {
            'video_paths': cfg.get('_resume_video_paths', [str(Path(args.video).expanduser().resolve())]),
            'run_dir': str(Path(args.run_dir).expanduser().resolve()),
            'prior_run_dir': None if args.prior_run_dir is None else str(Path(args.prior_run_dir).expanduser().resolve()),
            'start_stage': args.start_stage,
            'config': cfg,
        },
    )

    pipeline = Room4DPipeline(cfg)
    registry = pipeline.stage_registry()

    outputs = {}
    stage_labels = {
        'ingest': 'Stage 1/10 · Ingest / preprocess',
        'sync': 'Stage 2/10 · Temporal sync',
        'geometry': 'Stage 3/10 · VGGT geometry',
        'tracking': 'Stage 4/10 · Tracking + lifting + static/dynamic',
        'humans': 'Stage 5/10 · Human perception',
        'objects': 'Stage 6/10 · Object perception',
        'graph': 'Stage 7/10 · Scene graph',
        'optimization': 'Stage 8/10 · Optimization',
        'scene': 'Stage 9/10 · 4D scene assembly',
        'render': 'Stage 10/10 · Novel-view render',
    }

    stage_release_names = {
        'geometry': 'geometry',
        'tracking': 'tracking',
        'humans': 'humans',
        'objects': 'objects',
        'render': 'render',
    }

    start_idx = list(STAGE_ORDER).index(args.start_stage)
    if start_idx > 0 and not args.prior_run_dir:
        raise ValueError('--prior-run-dir is required when --start-stage is later than ingest.')

    for idx, stage_name in enumerate(STAGE_ORDER):
        spec = registry[stage_name]
        label = stage_labels[stage_name]
        if idx < start_idx:
            started = _debug_stage(console, args.debug, f"{label} [load]")
            artifact = materialize_stage_from_prior(args.prior_run_dir, args.run_dir, spec, cfg, [args.video])
            outputs[stage_name] = artifact
            if stage_name == 'tracking':
                outputs['tracks'] = artifact
            _debug_stage_done(console, args.debug, f"{label.split('·', 1)[1].strip()} loaded", started, save_dir=str(artifact.save_dir))
            continue

        missing = [dep for dep in spec.dependencies if dep not in outputs]
        if missing:
            raise RuntimeError(f"Cannot execute stage '{stage_name}' because dependencies are missing: {missing}")
        started = _debug_stage(console, args.debug, label)
        artifact = spec.run_fn(pipeline, outputs, [args.video])
        outputs[stage_name] = artifact
        if stage_name == 'tracking':
            outputs['tracks'] = artifact
        _debug_stage_done(console, args.debug, label.split('·', 1)[1].strip(), started, save_dir=str(artifact.save_dir))
        release_name = stage_release_names.get(stage_name)
        if release_name is not None:
            pipeline.release_stage_resources(release_name)

    if debug_cfg.get('enabled', False) and 'ingest' in outputs:
        export_started = _debug_stage(console, args.debug, 'Deferred debug export')
        shared_debug_images = materialize_shared_debug_frames(pipeline, outputs['ingest'], debug_cfg)
        export_debug_artifacts(args.run_dir, debug_cfg, shared_debug_images)
        release_memory()
        _debug_stage_done(console, args.debug, 'Deferred debug export', export_started, save_dir=args.run_dir)

    _print_frame_summary(console, outputs, cfg)

    for name in STAGE_ORDER:
        artifact = outputs[name]
        print(f'{name}: {artifact.summary()}')
        print(f'  save_dir: {artifact.save_dir}')


if __name__ == '__main__':
    main()

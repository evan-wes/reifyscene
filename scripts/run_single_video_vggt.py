#!/usr/bin/env python3
"""Run the single-video ingest -> sync -> geometry path.

This helper is intentionally narrow: it exists to validate that a single RGB
video can produce a persisted ``GeometryArtifact`` with actual per-frame camera,
depth, and point-map outputs when the selected real geometry backend is
installed.

Supported geometry backends:
- vggt
- vggt-long
- vggt-omega
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common import data_dir
from reifyscene.common.io import read_yaml
from reifyscene.debug_exports import export_debug_artifacts, load_debug_config, materialize_shared_debug_frames
from reifyscene.common.memory import release_memory
from reifyscene.common.system_report import collect_system_report, recommend_vggt_long_profile, save_system_report


def build_config(args: argparse.Namespace, *, system_report: dict | None = None) -> dict:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault("run_id", Path(args.video).stem)
    cfg["run_dir"] = str(Path(args.run_dir))

    geometry = cfg.setdefault("geometry", {})
    geometry["backend_name"] = getattr(args, "geometry_backend_name", "vggt")
    geometry["backend_mode"] = args.backend_mode
    geometry["frame_source"] = str(getattr(args, "geometry_frame_source", "keyframes")).lower()
    geometry["system_report"] = system_report or collect_system_report()

    if args.device is not None:
        geometry["device"] = args.device
    if args.frame_cache_dir is not None:
        geometry["frame_cache_dir"] = args.frame_cache_dir
    geometry_resize_max_dim = getattr(args, "geometry_resize_max_dim", None)
    if geometry_resize_max_dim is not None:
        geometry["resize_max_dim"] = int(geometry_resize_max_dim)

    geometry["debug_shapes"] = bool(args.debug)
    geometry["save_depth_pngs"] = bool(args.debug)
    geometry["debug_max_points"] = int(args.debug_max_points)

    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        geometry["cpu_threads"] = int(cpu_threads)

    if args.max_keyframes is not None:
        cfg["max_keyframes_per_stream"] = args.max_keyframes
    if args.keyframe_stride is not None:
        cfg["keyframe_stride"] = args.keyframe_stride
    if args.keyframe_stride_sec is not None:
        cfg["keyframe_stride_sec"] = args.keyframe_stride_sec

    # VGGT-Long-specific geometry options
    if getattr(args, "geometry_backend_name", "vggt") == "vggt-long":
        if args.vggt_long_chunk_size is not None:
            geometry["chunk_size"] = int(args.vggt_long_chunk_size)
        if args.vggt_long_overlap is not None:
            geometry["overlap"] = int(args.vggt_long_overlap)
        geometry["auto_tune"] = bool(getattr(args, "vggt_long_auto_tune", False))
        geometry["min_chunk_size"] = int(getattr(args, "vggt_long_min_chunk_size", 2))
        vggt_long_min_resize_max_dim = getattr(args, "vggt_long_min_resize_max_dim", None)
        if vggt_long_min_resize_max_dim is not None:
            geometry["min_resize_max_dim"] = int(vggt_long_min_resize_max_dim)
        if args.vggt_long_loop_enable is not None:
            geometry["loop_enable"] = bool(args.vggt_long_loop_enable)
        if args.vggt_long_use_dbow is not None:
            geometry["use_dbow"] = bool(args.vggt_long_use_dbow)
        if args.vggt_long_delete_temp_files is not None:
            geometry["delete_temp_files"] = bool(args.vggt_long_delete_temp_files)
        if args.vggt_long_output_cache_dir is not None:
            geometry["output_cache_dir"] = args.vggt_long_output_cache_dir

    if getattr(args, "geometry_backend_name", "vggt") == "vggt-omega":
        geometry["omega_mode"] = str(getattr(args, "vggt_omega_mode", "balanced"))
        geometry["omega_resolution"] = int(getattr(args, "vggt_omega_resolution", 512))
        geometry["omega_checkpoint"] = str(
            getattr(args, "vggt_omega_checkpoint", "vggt_omega_1b_512.pt"))

    cfg.setdefault("runtime", {})["system_report"] = system_report or collect_system_report()
    return cfg


def _debug_stage(console: Console, enabled: bool, label: str) -> float:
    if enabled:
        console.rule(f"[bold cyan]{label}")
    return perf_counter()


def _debug_stage_done(
    console: Console,
    enabled: bool,
    label: str,
    started_at: float,
    *,
    save_dir: str | None = None,
) -> None:
    if not enabled:
        return
    elapsed = perf_counter() - started_at
    if save_dir is not None:
        console.print(f"[green]✓[/green] {label} complete in {elapsed:.2f}s → {save_dir}")
    else:
        console.print(f"[green]✓[/green] {label} complete in {elapsed:.2f}s")


def _print_system_report(console: Console, report: dict, geometry_recommendation: dict | None = None) -> None:
    table = Table(title="System preflight report")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row(
        "Platform",
        f"{report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})",
    )
    table.add_row("Python", str(report["platform"]["python"]))
    table.add_row("Logical CPU cores", str(report["cpu"].get("logical_cores")))
    memory = report.get("memory", {})
    table.add_row(
        "RAM total / available",
        f"{memory.get('total_gib', 0):.1f} GiB / {memory.get('available_gib', 0):.1f} GiB",
    )
    table.add_row(
        "Swap total / free",
        f"{memory.get('swap_total_gib', 0):.1f} GiB / {memory.get('swap_free_gib', 0):.1f} GiB",
    )
    gpus = report.get("gpus", []) or []
    if gpus:
        gpu = max(gpus, key=lambda g: g.get("memory_free_mb", 0))
        table.add_row(
            "GPU",
            f"{gpu.get('name')} (free {gpu.get('memory_free_mb', 0)/1024.0:.1f} / {gpu.get('memory_total_mb', 0)/1024.0:.1f} GiB)",
        )
    else:
        table.add_row("GPU", "None detected via nvidia-smi")
    if geometry_recommendation:
        resize_candidates = geometry_recommendation.get("resize_candidates") or []
        resize_preview = ", ".join(str(v) for v in resize_candidates[:6])
        if len(resize_candidates) > 6:
            resize_preview += ", ..."
        table.add_row(
            "VGGT-Long seed",
            f"profile={geometry_recommendation.get('profile')} chunk={geometry_recommendation.get('initial_chunk_size')} overlap={geometry_recommendation.get('initial_overlap')} resize={geometry_recommendation.get('initial_resize_max_dim')}",
        )
        if resize_preview:
            table.add_row("Resize ladder", resize_preview)
    console.print(table)


def _print_run_header(console: Console, args: argparse.Namespace, geometry_recommendation: dict | None) -> None:
    body = Text()
    body.append(f"video: {args.video}\n")
    body.append(f"run_dir: {args.run_dir}\n")
    body.append(
        f"geometry backend: {args.geometry_backend_name}  mode: {args.backend_mode}  frame_source: {args.geometry_frame_source}  device: {args.device or 'auto'}\n"
    )
    body.append(
        f"max_keyframes: {args.max_keyframes}  keyframe_stride: {args.keyframe_stride}  keyframe_stride_sec: {args.keyframe_stride_sec}  cpu_threads: {args.cpu_threads}\n"
    )
    body.append(f"geometry_resize_max_dim: {args.geometry_resize_max_dim}\n")
    if args.geometry_backend_name == "vggt-long":
        body.append(
            f"vggt-long: chunk_size={args.vggt_long_chunk_size} overlap={args.vggt_long_overlap} auto_tune={args.vggt_long_auto_tune} min_chunk={args.vggt_long_min_chunk_size} min_resize={args.vggt_long_min_resize_max_dim}\n"
        )
        if geometry_recommendation:
            resize_candidates = geometry_recommendation.get("resize_candidates") or []
            resize_preview = ", ".join(str(v) for v in resize_candidates[:6])
            if len(resize_candidates) > 6:
                resize_preview += ", ..."
            body.append(
                f"recommended seed: chunk={geometry_recommendation.get('initial_chunk_size')} overlap={geometry_recommendation.get('initial_overlap')} resize={geometry_recommendation.get('initial_resize_max_dim')}  ladder=[{resize_preview}]"
            )
    console.print(Panel(body, title="reifyscene single-video geometry run", expand=False))


def _print_stage_summary(console: Console, ingest, sync, geometry) -> None:
    table = Table(title="Run outputs")
    table.add_column("Artifact")
    table.add_column("Summary")
    table.add_column("Path")
    table.add_row("Ingest", str(ingest.summary()), str(ingest.save_dir))
    table.add_row("Sync", str(sync.summary()), str(sync.save_dir))
    table.add_row("Geometry", str(geometry.summary()), str(geometry.save_dir))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True,
                        help="A single RGB video, by name in the data directory's datasets/.")
    parser.add_argument("--run-dir", required=True,
                        help="Where artifacts are written, by name in runs/.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--debug-config", default=None, help="Optional YAML debug-export config path.")

    parser.add_argument(
        "--geometry-backend-name",
        default="vggt",
        choices=["vggt", "vggt-long", "vggt-omega"],
        help="Which geometry backend to use.",
    )
    parser.add_argument(
        "--backend-mode",
        default="real",
        choices=["real", "auto", "heuristic"],
        help="Geometry backend mode.",
    )
    parser.add_argument(
        "--geometry-frame-source",
        default="keyframes",
        choices=["keyframes", "tracking", "all"],
        help="Which frame timeline geometry should run on. Use 'all' for the full sequence or 'tracking' to mirror the tracker cadence from config.",
    )
    parser.add_argument("--device", default=None, help="Geometry device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit torch CPU threads when running geometry on CPU.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print/save the system report and exit.",
    )
    parser.add_argument("--frame-cache-dir", default=None)
    parser.add_argument("--geometry-resize-max-dim", type=int, default=None, help="Optionally downscale materialized geometry frames before backend execution.")

    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--keyframe-stride", type=int, default=None,
                        help="Keyframe spacing in frames (30 when neither stride is given).")
    parser.add_argument("--keyframe-stride-sec", type=float, default=None,
                        help="Keyframe spacing in seconds; needs a source that records its fps.")

    # VGGT-Long-specific flags
    parser.add_argument("--vggt-long-chunk-size", type=int, default=None)
    parser.add_argument("--vggt-long-overlap", type=int, default=None)
    parser.add_argument("--vggt-long-auto-tune", action="store_true", help="Automatically back off VGGT-Long chunk/overlap/resize settings on OOM.")
    parser.add_argument("--vggt-long-min-chunk-size", type=int, default=2)
    parser.add_argument("--vggt-long-min-resize-max-dim", type=int, default=None)
    parser.add_argument(
        "--vggt-long-loop-enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable loop closure inside VGGT-Long.",
    )
    parser.add_argument(
        "--vggt-long-use-dbow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DBoW retrieval inside VGGT-Long.",
    )
    parser.add_argument(
        "--vggt-long-delete-temp-files",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Delete VGGT-Long temporary files after processing.",
    )
    parser.add_argument(
        "--vggt-long-output-cache-dir",
        default=None,
        help="Optional VGGT-Long output cache directory.",
    )
    parser.add_argument(
        "--vggt-omega-mode",
        default="balanced",
        choices=["balanced", "max_size"],
        help="VGGT-Omega's resize: balanced keeps about resolution^2 pixels; max_size takes the "
             "long side to the resolution and needs less GPU memory.",
    )
    parser.add_argument(
        "--vggt-omega-resolution", type=int, default=512, help="VGGT-Omega's input resolution (px)."
    )
    parser.add_argument(
        "--vggt-omega-checkpoint",
        default="vggt_omega_1b_512.pt",
        help="The checkpoint file in facebook/VGGT-Omega on Hugging Face (gated: request access "
             "first).",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose geometry shape logging and save depth preview PNGs, "
            "camera JSON, PLYs, and a Plotly HTML viewer."
        ),
    )
    parser.add_argument(
        "--debug-max-points",
        type=int,
        default=5000,
        help="Maximum point-cloud samples per keyframe in debug exports.",
    )
    args = parser.parse_args()
    args.video = str(data_dir.path(data_dir.DATASETS, args.video))
    args.run_dir = str(data_dir.path(data_dir.RUNS, args.run_dir))

    console = Console()
    system_report = collect_system_report()
    debug_cfg = load_debug_config(args.debug_config)
    geometry_recommended = None
    if args.geometry_backend_name == "vggt-long":
        geometry_recommended = recommend_vggt_long_profile(system_report, device=args.device)
    report_path = Path(args.run_dir) / "preflight_system_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_system_report(report_path, {"report": system_report, "recommended_vggt_long_profile": geometry_recommended})

    _print_system_report(console, system_report, geometry_recommendation=geometry_recommended)
    console.print(f"saved preflight report: {report_path}")
    if args.preflight_only:
        return

    _print_run_header(console, args, geometry_recommended)
    if args.debug and args.geometry_backend_name == "vggt-long":
        console.print(
            f"loop_enable={args.vggt_long_loop_enable}  use_dbow={args.vggt_long_use_dbow}  "
            f"delete_temp_files={args.vggt_long_delete_temp_files}  output_cache_dir={args.vggt_long_output_cache_dir}"
        )

    pipeline = ReifyScenePipeline(build_config(args, system_report=system_report))

    started = _debug_stage(console, args.debug, "Stage 1/3 · Ingest / preprocess")
    ingest = pipeline.preprocess([args.video])
    _debug_stage_done(console, args.debug, "Ingest / preprocess", started, save_dir=str(ingest.save_dir))
    shared_debug_images = materialize_shared_debug_frames(pipeline, ingest, debug_cfg) if debug_cfg.get('enabled', False) else {}
    pipeline.release_stage_resources('ingest')

    started = _debug_stage(console, args.debug, "Stage 2/3 · Temporal sync")
    sync = pipeline.synchronize(ingest)
    _debug_stage_done(console, args.debug, "Temporal sync", started, save_dir=str(sync.save_dir))
    pipeline.release_stage_resources('sync')

    stage_label = (
        "Stage 3/3 · VGGT-Long geometry"
        if args.geometry_backend_name == "vggt-long"
        else "Stage 3/3 · VGGT geometry"
    )
    started = _debug_stage(console, args.debug, stage_label)
    geometry = pipeline.initialize_geometry(ingest, sync)
    _debug_stage_done(console, args.debug, stage_label, started, save_dir=str(geometry.save_dir))
    pipeline.release_stage_resources('geometry')

    if debug_cfg.get('enabled', False):
        export_started = _debug_stage(console, args.debug, 'Deferred debug export')
        export_debug_artifacts(args.run_dir, debug_cfg, shared_debug_images)
        release_memory()
        _debug_stage_done(console, args.debug, 'Deferred debug export', export_started, save_dir=args.run_dir)

    _print_stage_summary(console, ingest, sync, geometry)


if __name__ == "__main__":
    main()

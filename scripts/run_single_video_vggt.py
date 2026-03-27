#!/usr/bin/env python3
"""Run the single-video ingest -> sync -> geometry path.

This helper is intentionally narrow: it exists to validate that a single RGB
video can produce a persisted ``GeometryArtifact`` with actual per-frame camera,
depth, and point-map outputs when the selected real geometry backend is
installed.

Supported geometry backends:
- vggt
- vggt-long
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.table import Table

from room4d.api.pipeline import Room4DPipeline
from room4d.common.io import read_yaml
from room4d.common.system_report import collect_system_report, save_system_report


def build_config(args: argparse.Namespace, *, system_report: dict | None = None) -> dict:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault("run_id", Path(args.video).stem)
    cfg["run_dir"] = str(Path(args.run_dir))

    geometry = cfg.setdefault("geometry", {})
    geometry["backend_name"] = args.geometry_backend_name
    geometry["backend_mode"] = args.backend_mode

    if args.device is not None:
        geometry["device"] = args.device
    if args.frame_cache_dir is not None:
        geometry["frame_cache_dir"] = args.frame_cache_dir

    geometry["debug_shapes"] = bool(args.debug)
    geometry["save_depth_pngs"] = bool(args.debug)
    geometry["debug_max_points"] = int(args.debug_max_points)

    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        geometry["cpu_threads"] = int(cpu_threads)

    if args.max_keyframes is not None:
        cfg["max_keyframes_per_stream"] = args.max_keyframes
    if args.keyframe_stride_sec is not None:
        cfg["keyframe_stride_sec"] = args.keyframe_stride_sec

    # VGGT-Long-specific geometry options
    if args.geometry_backend_name == "vggt-long":
        if args.vggt_long_chunk_size is not None:
            geometry["chunk_size"] = int(args.vggt_long_chunk_size)
        if args.vggt_long_overlap is not None:
            geometry["overlap"] = int(args.vggt_long_overlap)
        if args.vggt_long_loop_enable is not None:
            geometry["loop_enable"] = bool(args.vggt_long_loop_enable)
        if args.vggt_long_use_dbow is not None:
            geometry["use_dbow"] = bool(args.vggt_long_use_dbow)
        if args.vggt_long_delete_temp_files is not None:
            geometry["delete_temp_files"] = bool(args.vggt_long_delete_temp_files)
        if args.vggt_long_output_cache_dir is not None:
            geometry["output_cache_dir"] = args.vggt_long_output_cache_dir

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


def _print_system_report(console: Console, report: dict) -> None:
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
    gpus = report.get("gpus", []) or []
    if gpus:
        gpu = max(gpus, key=lambda g: g.get("memory_free_mb", 0))
        table.add_row(
            "GPU",
            f"{gpu.get('name')} (free {gpu.get('memory_free_mb', 0)/1024.0:.1f} / {gpu.get('memory_total_mb', 0)/1024.0:.1f} GiB)",
        )
    else:
        table.add_row("GPU", "None detected via nvidia-smi")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a single RGB video file.")
    parser.add_argument("--run-dir", required=True, help="Directory where artifacts will be written.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")

    parser.add_argument(
        "--geometry-backend-name",
        default="vggt",
        choices=["vggt", "vggt-long"],
        help="Which geometry backend to use.",
    )
    parser.add_argument(
        "--backend-mode",
        default="real",
        choices=["real", "auto", "heuristic"],
        help="Geometry backend mode.",
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

    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--keyframe-stride-sec", type=float, default=1.0)

    # VGGT-Long-specific flags
    parser.add_argument("--vggt-long-chunk-size", type=int, default=None)
    parser.add_argument("--vggt-long-overlap", type=int, default=None)
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

    console = Console()
    system_report = collect_system_report()
    report_path = Path(args.run_dir) / "preflight_system_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_system_report(report_path, {"report": system_report})

    if args.debug or args.preflight_only:
        _print_system_report(console, system_report)
        console.print(f"saved preflight report: {report_path}")
    if args.preflight_only:
        return

    if args.debug:
        console.print("[bold]room4d single-video geometry debug run[/bold]")
        console.print(f"video: {args.video}")
        console.print(f"run_dir: {args.run_dir}")
        console.print(
            f"geometry backend: {args.geometry_backend_name}  "
            f"mode: {args.backend_mode}  "
            f"device: {args.device or 'auto'}"
        )
        console.print(
            f"max_keyframes: {args.max_keyframes}  "
            f"keyframe_stride_sec: {args.keyframe_stride_sec}"
        )
        console.print(f"cpu_threads: {args.cpu_threads}")
        if args.geometry_backend_name == "vggt-long":
            console.print(
                "vggt-long:"
                f" chunk_size={args.vggt_long_chunk_size}"
                f" overlap={args.vggt_long_overlap}"
                f" loop_enable={args.vggt_long_loop_enable}"
                f" use_dbow={args.vggt_long_use_dbow}"
                f" delete_temp_files={args.vggt_long_delete_temp_files}"
                f" output_cache_dir={args.vggt_long_output_cache_dir}"
            )

    pipeline = Room4DPipeline(build_config(args, system_report=system_report))

    started = _debug_stage(console, args.debug, "Stage 1/3 · Ingest / preprocess")
    ingest = pipeline.preprocess([args.video])
    _debug_stage_done(console, args.debug, "Ingest / preprocess", started, save_dir=str(ingest.save_dir))

    started = _debug_stage(console, args.debug, "Stage 2/3 · Temporal sync")
    sync = pipeline.synchronize(ingest)
    _debug_stage_done(console, args.debug, "Temporal sync", started, save_dir=str(sync.save_dir))

    stage_label = (
        "Stage 3/3 · VGGT-Long geometry"
        if args.geometry_backend_name == "vggt-long"
        else "Stage 3/3 · VGGT geometry"
    )
    started = _debug_stage(console, args.debug, stage_label)
    geometry = pipeline.initialize_geometry(ingest, sync)
    _debug_stage_done(console, args.debug, stage_label, started, save_dir=str(geometry.save_dir))

    print("Ingest summary:", ingest.summary())
    print("Sync summary:", sync.summary())
    print("Geometry summary:", geometry.summary())
    print("Geometry artifact dir:", geometry.save_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run phase-1 hloc + COLMAP calibration on one or more video streams."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.table import Table

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.io import read_yaml
from reifyscene.common.system_report import collect_system_report, save_system_report


def build_config(args: argparse.Namespace, *, system_report: dict | None = None) -> dict:
    cfg = read_yaml(args.config) if args.config else {}
    cfg.setdefault("run_id", Path(args.videos[0]).stem if args.videos else Path(args.run_dir).name)
    cfg["run_dir"] = str(Path(args.run_dir))
    calibration = cfg.setdefault("calibration", {})
    sampling = calibration.setdefault("sampling", {})
    if args.sample_spacing_sec is not None:
        sampling["sample_spacing_sec"] = float(args.sample_spacing_sec)
    if args.max_frames_per_stream is not None:
        sampling["max_frames_per_stream"] = int(args.max_frames_per_stream)
    if args.image_max_dim is not None:
        sampling["image_max_dim"] = int(args.image_max_dim)
    if args.disable_static_filter:
        sampling["prefer_static_frames"] = False
    if args.static_diff_threshold is not None:
        sampling["static_diff_threshold"] = float(args.static_diff_threshold)
    hloc_cfg = calibration.setdefault("hloc_colmap", {})
    if args.retrieval_top_k is not None:
        hloc_cfg["retrieval_top_k"] = int(args.retrieval_top_k)
    if args.camera_model is not None:
        hloc_cfg["camera_model"] = str(args.camera_model)
    if args.camera_mode is not None:
        hloc_cfg["camera_mode"] = str(args.camera_mode)
    if args.fix_intrinsics:
        hloc_cfg["fix_intrinsics"] = True
    if args.verbose_hloc:
        hloc_cfg["verbose"] = True
    cfg.setdefault("runtime", {})["system_report"] = system_report or collect_system_report()
    return cfg


def _print_system_report(console: Console, report: dict) -> None:
    table = Table(title="Calibration preflight report")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Platform", f"{report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})")
    table.add_row("Python", str(report['platform']['python']))
    table.add_row("Logical CPU cores", str(report['cpu'].get('logical_cores')))
    memory = report.get('memory', {})
    table.add_row('RAM total / available', f"{memory.get('total_gib', 0):.1f} GiB / {memory.get('available_gib', 0):.1f} GiB")
    gpus = report.get('gpus', []) or []
    if gpus:
        gpu = max(gpus, key=lambda g: g.get('memory_free_mb', 0))
        table.add_row('GPU', f"{gpu.get('name')} (free {gpu.get('memory_free_mb', 0)/1024.0:.1f} / {gpu.get('memory_total_mb', 0)/1024.0:.1f} GiB)")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", dest="videos", action="append", required=True, help="Input video path. Repeat for multiple streams.")
    parser.add_argument("--run-dir", required=True, help="Output directory for the calibration artifact.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "calibration" / "calibration.yaml"), help="Calibration config YAML.")
    parser.add_argument("--sample-spacing-sec", type=float, default=None)
    parser.add_argument("--max-frames-per-stream", type=int, default=None)
    parser.add_argument("--image-max-dim", type=int, default=None)
    parser.add_argument("--disable-static-filter", action="store_true")
    parser.add_argument("--static-diff-threshold", type=float, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--camera-model", default=None)
    parser.add_argument("--camera-mode", default=None, choices=["AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"])
    parser.add_argument("--fix-intrinsics", action="store_true")
    parser.add_argument("--verbose-hloc", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    console = Console()
    report = collect_system_report()
    _print_system_report(console, report)
    save_system_report(report, Path(args.run_dir) / "preflight_system_report.json")

    started = perf_counter()
    pipeline = ReifyScenePipeline(build_config(args, system_report=report))
    artifact = pipeline.calibrate(args.videos)
    elapsed = perf_counter() - started
    console.print(f"[green]Calibration complete[/green] in {elapsed:.2f}s → {artifact.save_dir}")
    summary = Table(title="Calibration outputs")
    summary.add_column("Artifact")
    summary.add_column("Value")
    summary.add_row("Sampled frames", str(len(artifact.sampled_frames)))
    summary.add_row("Registered images", str(int(artifact.reprojection_stats.get('registered_images', 0))))
    summary.add_row("3D points", str(int(artifact.reprojection_stats.get('num_points3d', 0))))
    summary.add_row("Reprojection mean", f"{float(artifact.reprojection_stats.get('mean', 0.0)):.3f}")
    summary.add_row("Connected components", str(len(artifact.connected_components)))
    summary.add_row("Viewer", str(Path(artifact.save_dir) / artifact.metadata.get('viewer_html', 'calibration_viewer.html')))
    summary.add_row("Report", str(Path(artifact.save_dir) / artifact.metadata.get('report_html', 'calibration_report.html')))
    console.print(summary)


if __name__ == "__main__":
    main()

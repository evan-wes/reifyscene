#!/usr/bin/env python3
"""Run a reference 4D reconstruction benchmark and emit a normalized result.

This is the room4d-side wrapper around the vendored Shape of Motion repo
(third_party/shape-of-motion, installed into an isolated conda env by
scripts/setup_reference_tool.sh). It turns a processed dataset sequence into a
normalized ``reference_benchmark.json`` that room4d's own renderer results can
be compared against.

Pipeline:
  1. validate the sequence directory layout (iPhone/DyCheck format)
  2. train Shape of Motion in the reference env (skipped when the run dir
     already holds a checkpoint; --retrain forces)
  3. run its evaluation script and parse the printed metrics
  4. write ``reference_benchmark.json`` into the run directory

The JSON schema (``reference_benchmark_v1``) is the comparison contract:
room4d's renderer evaluation should emit the same shape so that
``--compare a.json b.json`` can score any two methods on the same sequence.

This script needs only the Python stdlib and may run from any environment;
all heavy work happens in the reference conda env via ``conda run``.

Examples:
  # benchmark (reuses an existing trained run when present)
  python scripts/run_reference_benchmark.py \
    --data-dir .../iphone/paper-windmill \
    --run-dir  .../runs/paper-windmill-full

  # compare two results (e.g. room4d vs the reference)
  python scripts/run_reference_benchmark.py --compare room4d.json reference.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOM_DIR = REPO_ROOT / "third_party" / "shape-of-motion"
DEFAULT_CONDA_ENV = "room4d_som_ref"
SCHEMA = "reference_benchmark_v1"

# evaluate_iphone.py prints "mean <name>: <float>" lines; these patterns are
# stable at the pinned SoM commit (see setup_reference_tool.sh SOM_COMMIT).
METRIC_PATTERNS: dict[str, str] = {
    "masked_mpsnr": r"mean NV mPSNR:\s*([0-9.]+)",
    "masked_mssim": r"mean NV mSSIM:\s*([0-9.]+)",
    "masked_mlpips": r"mean NV mLPIPS:\s*([0-9.]+)",
    "track3d_epe": r"mean 3D tracking EPE:\s*([0-9.]+)",
    "track3d_pck_10cm": r"mean 3D tracking PCK \(10cm\):\s*([0-9.]+)",
    "track3d_pck_5cm": r"mean 3D tracking PCK \(5cm\):\s*([0-9.]+)",
    "track2d_aj": r"mean 2D tracking AJ:\s*([0-9.]+)",
    "track2d_avg_pck": r"mean 2D tracking avg PCK:\s*([0-9.]+)",
    "track2d_occ_acc": r"mean 2D tracking occlusion accuracy:\s*([0-9.]+)",
}

# Direction of "better" per metric, used by --compare output.
HIGHER_IS_BETTER = {
    "masked_mpsnr": True,
    "masked_mssim": True,
    "masked_mlpips": False,
    "track3d_epe": False,
    "track3d_pck_10cm": True,
    "track3d_pck_5cm": True,
    "track2d_aj": True,
    "track2d_avg_pck": True,
    "track2d_occ_acc": True,
}

IPHONE_REQUIRED_DIRS = ["rgb", "depth", "camera", "splits", "flow3d_preprocessed"]
# NOTE: flow3d_preprocessed/scene_depth.npy is intentionally NOT required —
# it only exists for depth_type="monst3r"; the shipped sequences use
# aligned_depth_anything_colmap/. 2d_tracks/ is needed for every depth type.
IPHONE_REQUIRED_FILES = [
    "dataset.json",
    "splits/train.json",
    "flow3d_preprocessed/2d_tracks",
]


def find_conda() -> str:
    """Locate the conda executable without requiring an activated shell."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe
    found = shutil.which("conda")
    if found:
        return found
    fallback = Path.home() / "miniconda3" / "condabin" / "conda"
    if fallback.is_file():
        return str(fallback)
    sys.exit("conda executable not found (checked CONDA_EXE, PATH, $HOME/miniconda3).")


def conda_run_cmd(conda: str, env: str, *args: str) -> list[str]:
    return [conda, "run", "--no-capture-output", "-n", env, *args]


def validate_iphone_sequence(data_dir: Path) -> None:
    problems = []
    for d in IPHONE_REQUIRED_DIRS:
        if not (data_dir / d).is_dir():
            problems.append(f"missing dir:  {d}/")
    for f in IPHONE_REQUIRED_FILES:
        if not (data_dir / f).exists():
            problems.append(f"missing file: {f}")
    if problems:
        listing = "\n  ".join(problems)
        sys.exit(
            f"Sequence at {data_dir} does not look like a processed iPhone/DyCheck "
            f"sequence:\n  {listing}\n"
            "Re-check the download (see scripts/verify_reference_data.sh)."
        )


def run_training(
    conda: str, env: str, som_dir: Path, data_dir: Path, run_dir: Path,
    num_epochs: int | None,
) -> float:
    """Train SoM on the sequence; returns wall-clock minutes."""
    cmd = conda_run_cmd(
        conda, env, "python", "run_training.py",
        "--work-dir", str(run_dir),
    )
    if num_epochs is not None:
        cmd += ["--num-epochs", str(num_epochs)]
    cmd += ["data:iphone", "--data.data-dir", str(data_dir)]
    print(f"[train] {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    subprocess.run(cmd, cwd=som_dir, check=True)
    return (time.monotonic() - start) / 60.0


def run_eval(
    conda: str, env: str, som_dir: Path, data_dir: Path, run_dir: Path,
    seq_name: str,
) -> tuple[dict[str, float], str]:
    """Run evaluate_iphone.py, return (parsed metrics, full output text)."""
    cmd = conda_run_cmd(
        conda, env, "python", "scripts/evaluate_iphone.py",
        "--data_dir", str(data_dir),
        "--result_dir", str(run_dir),
        "--seq_names", seq_name,
    )
    print(f"[eval] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=som_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "."},
    )
    output = proc.stdout or ""
    (run_dir / "benchmark_eval.log").write_text(output)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-25:])
        sys.exit(f"evaluate_iphone.py failed (exit {proc.returncode}). Tail:\n{tail}")

    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
        else:
            print(f"[warn] metric not found in eval output: {key}", flush=True)
    if not any(k.startswith("masked_") for k in metrics):
        sys.exit(
            "No novel-view metrics parsed from eval output — the print format "
            "may have drifted from the pinned SoM commit. See "
            f"{run_dir / 'benchmark_eval.log'}"
        )
    return metrics, output


def capture_env_versions(conda: str, env: str) -> dict[str, str]:
    snippet = (
        "import json, torch, gsplat; "
        "print(json.dumps({'torch': torch.__version__, "
        "'torch_cuda': torch.version.cuda or '', "
        "'gsplat': getattr(gsplat, '__version__', 'unknown')}))"
    )
    try:
        out = subprocess.run(
            conda_run_cmd(conda, env, "python", "-c", snippet),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
        return json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - version capture is best-effort
        return {}


def latest_rgb_video(run_dir: Path) -> str | None:
    candidates = sorted(run_dir.glob("videos/epoch_*/rgbs.mp4"))
    return str(candidates[-1]) if candidates else None


def som_pinned_commit(som_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(som_dir), "rev-parse", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def benchmark(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    som_dir = Path(args.som_dir).expanduser().resolve()
    seq_name = args.seq_name or data_dir.name

    if not som_dir.is_dir():
        sys.exit(
            f"Shape of Motion checkout not found at {som_dir}. "
            "Run scripts/setup_reference_tool.sh first."
        )
    validate_iphone_sequence(data_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    conda = find_conda()

    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    train_minutes: float | None = None
    if checkpoint.exists() and not args.retrain:
        print(f"[train] reusing existing run ({checkpoint})", flush=True)
    else:
        train_minutes = run_training(
            conda, args.conda_env, som_dir, data_dir, run_dir, args.num_epochs
        )

    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        sys.exit(
            f"{results_dir} missing — training did not produce eval renders "
            "(interrupted run?). Re-run with --retrain."
        )

    metrics, _ = run_eval(conda, args.conda_env, som_dir, data_dir, run_dir, seq_name)

    result = {
        "schema": SCHEMA,
        "method": "shape-of-motion",
        "method_commit": som_pinned_commit(som_dir),
        "sequence": seq_name,
        "dataset_format": "iphone",
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "videos": {"rgb": latest_rgb_video(run_dir)},
        "env": capture_env_versions(conda, args.conda_env),
        "train_minutes": round(train_minutes, 1) if train_minutes is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = Path(args.output) if args.output else run_dir / "reference_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nReference benchmark written to: {out_path}\n")
    width = max(len(k) for k in metrics)
    for key in METRIC_PATTERNS:
        if key in metrics:
            arrow = "↑" if HIGHER_IS_BETTER.get(key, True) else "↓"
            print(f"  {key:<{width}}  {metrics[key]:>10.4f}  {arrow}")


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    for blob, path in ((a, path_a), (b, path_b)):
        if blob.get("schema") != SCHEMA:
            sys.exit(f"{path}: schema {blob.get('schema')!r} != {SCHEMA!r}")

    name_a = f"{a.get('method', '?')}"
    name_b = f"{b.get('method', '?')}"
    seq = a.get("sequence", "?")
    if b.get("sequence") != seq:
        print(
            f"[warn] comparing different sequences: {seq} vs {b.get('sequence')}",
            flush=True,
        )

    keys = [k for k in METRIC_PATTERNS if k in a.get("metrics", {}) or k in b.get("metrics", {})]
    width = max(len(k) for k in keys) if keys else 10
    print(f"\nSequence: {seq}")
    print(f"  {'metric':<{width}}  {name_a:>14}  {name_b:>14}  {'better':>8}")
    for key in keys:
        va = a.get("metrics", {}).get(key)
        vb = b.get("metrics", {}).get(key)
        sa = f"{va:.4f}" if va is not None else "—"
        sb = f"{vb:.4f}" if vb is not None else "—"
        better = ""
        if va is not None and vb is not None and va != vb:
            hib = HIGHER_IS_BETTER.get(key, True)
            better = name_a if (va > vb) == hib else name_b
        print(f"  {key:<{width}}  {sa:>14}  {sb:>14}  {better:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Processed sequence dir (iPhone/DyCheck format)")
    parser.add_argument("--run-dir", help="Training/eval output dir (reused when it has a checkpoint)")
    parser.add_argument("--seq-name", help="Sequence name (default: data-dir basename)")
    parser.add_argument("--som-dir", default=str(DEFAULT_SOM_DIR), help="Shape of Motion checkout")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Reference conda env name")
    parser.add_argument("--num-epochs", type=int, help="Override training epochs (smoke runs)")
    parser.add_argument("--retrain", action="store_true", help="Train even if a checkpoint exists")
    parser.add_argument("--output", help="Write the result JSON here instead of run-dir")
    parser.add_argument(
        "--compare", nargs=2, metavar=("A.json", "B.json"),
        help="Compare two benchmark JSONs and exit (no training/eval)",
    )
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.data_dir or not args.run_dir:
        parser.error("--data-dir and --run-dir are required (unless using --compare)")
    benchmark(args)


if __name__ == "__main__":
    main()

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
import math
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
# Must match the CONDA_ENV_NAME default in scripts/setup_reference_tool.sh.
DEFAULT_CONDA_ENV = "room4d_som_ref"
SCHEMA = "reference_benchmark_v1"
# Written into the run dir by this wrapper after run_training exits 0. SoM
# itself writes checkpoints/last.ckpt at epoch 0 and every 200 steps, so a
# bare checkpoint does NOT prove training completed — only this sentinel
# (or final-epoch video detection, for runs predating it) does.
TRAINING_COMPLETE_SENTINEL = ".room4d_training_complete"

# evaluate_iphone.py prints "mean <name>: <float>" lines; these patterns are
# stable at the pinned SoM commit (see setup_reference_tool.sh SOM_COMMIT).
# The value group accepts scientific notation — '([0-9.]+)' would silently
# truncate '8.21e-05' to 8.21, a 100,000x error in a recorded metric.
_FLOAT = r"([-+0-9.eE]+)"
METRIC_PATTERNS: dict[str, str] = {
    "masked_mpsnr": rf"mean NV mPSNR:\s*{_FLOAT}",
    "masked_mssim": rf"mean NV mSSIM:\s*{_FLOAT}",
    "masked_mlpips": rf"mean NV mLPIPS:\s*{_FLOAT}",
    "track3d_epe": rf"mean 3D tracking EPE:\s*{_FLOAT}",
    "track3d_pck_10cm": rf"mean 3D tracking PCK \(10cm\):\s*{_FLOAT}",
    "track3d_pck_5cm": rf"mean 3D tracking PCK \(5cm\):\s*{_FLOAT}",
    "track2d_aj": rf"mean 2D tracking AJ:\s*{_FLOAT}",
    "track2d_avg_pck": rf"mean 2D tracking avg PCK:\s*{_FLOAT}",
    "track2d_occ_acc": rf"mean 2D tracking occlusion accuracy:\s*{_FLOAT}",
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

# Required layout verified against the pinned SoM commit (579753e):
# - the iphone dataloader opens dataset.json, extra.json, splits/{train,val}.json
#   unconditionally, plus flow3d_preprocessed/2d_tracks for every depth type;
# - evaluate_iphone.py additionally needs flow3d_preprocessed/colmap/scale.npy,
#   flow3d_preprocessed/covisible/ and keypoint/ — checked here so a missing
#   eval input fails BEFORE spending ~an hour training, not after.
# NOTE: flow3d_preprocessed/scene_depth.npy is intentionally NOT required —
# it only exists for depth_type="monst3r"; the shipped sequences use
# aligned_depth_anything_colmap/. metadata.json ships with the data but is
# referenced nowhere in SoM, so it is not required.
IPHONE_REQUIRED_DIRS = [
    "rgb",
    "depth",
    "camera",
    "splits",
    "flow3d_preprocessed",
    "flow3d_preprocessed/2d_tracks",
    "flow3d_preprocessed/covisible",
    "keypoint",
]
IPHONE_REQUIRED_FILES = [
    "dataset.json",
    "extra.json",
    "splits/train.json",
    "splits/val.json",
    "flow3d_preprocessed/colmap/scale.npy",
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
            problems.append(f"missing dir:   {d}/")
    for f in IPHONE_REQUIRED_FILES:
        if not (data_dir / f).exists():
            problems.append(f"missing entry: {f}")
    if problems:
        listing = "\n  ".join(problems)
        sys.exit(
            f"Sequence at {data_dir} does not look like a processed iPhone/DyCheck "
            f"sequence:\n  {listing}\n"
            "Re-check the download (see scripts/verify_reference_data.sh)."
        )


def _epochs_label(num_epochs: int | None) -> str:
    """Canonical label for a training schedule ('default' = SoM's built-in)."""
    return "default" if num_epochs is None else str(num_epochs)


def completed_epochs(run_dir: Path) -> str | None:
    """The epoch label a completed run was trained with, or None if incomplete.

    SoM writes checkpoints/last.ckpt at initialization and every 200 steps,
    so an interrupted run leaves a checkpoint plus stale results/ behind —
    reusing those would silently present a half-trained model as the
    reference baseline. Completion is proven by either:
    - the sentinel this wrapper writes after run_training exits 0 (its body
      records the epoch schedule, so a 5-epoch smoke run can never be reused
      as a full baseline), or
    - (runs predating the sentinel) the final-epoch video dir existing, with
      the epoch count read from the run's own cfg.yaml — reported as the
      label of the schedule the run actually used.
    """
    sentinel = run_dir / TRAINING_COMPLETE_SENTINEL
    if sentinel.exists():
        return sentinel.read_text().strip() or "default"
    cfg = run_dir / "cfg.yaml"
    if cfg.is_file():
        match = re.search(r"^num_epochs:\s*(\d+)", cfg.read_text(), re.M)
        if match:
            epochs = int(match.group(1))
            if (run_dir / "videos" / f"epoch_{epochs - 1:04d}").is_dir():
                # 500 is SoM's default schedule; label accordingly so reuse
                # against a default-schedule request matches.
                return "default" if epochs == 500 else str(epochs)
    return None


def clear_run_state(run_dir: Path) -> None:
    """Remove training outputs so --retrain actually retrains.

    SoM auto-resumes from checkpoints in the work dir (init_from_checkpoint
    reads the saved epoch and trains range(start_epoch, num_epochs)), so
    without this a --retrain against a finished run trains ~0 epochs and
    re-scores the old renders while looking like a fresh run. Prior result
    artifacts are cleared too — a crashed retrain must not leave the OLD
    run's reference_benchmark.json looking current.
    """
    for entry in (
        "checkpoints", "results", "videos",
        TRAINING_COMPLETE_SENTINEL, "reference_benchmark.json", "benchmark_eval.log",
    ):
        path = run_dir / entry
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


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
    # Written only on success (check=True raises first) and records the
    # schedule, so completion claims are schedule-specific.
    (run_dir / TRAINING_COMPLETE_SENTINEL).write_text(_epochs_label(num_epochs) + "\n")
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
        if not match:
            print(f"[warn] metric not found in eval output: {key}", flush=True)
            continue
        try:
            metrics[key] = float(match.group(1))
        except ValueError:
            print(
                f"[warn] unparseable value for {key}: {match.group(1)!r}",
                flush=True,
            )
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
    """HEAD of the vendored checkout, with '+dirty' when the tree is patched.

    setup_reference_tool.sh deliberately sed-patches the checkout (device
    placement, gsplat radii), so a bare SHA would claim pristine-upstream
    provenance for results produced by modified code.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(som_dir), "rev-parse", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(som_dir), "status", "--porcelain"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        return f"{sha}+dirty" if status else sha
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

    train_minutes: float | None = None
    requested = _epochs_label(args.num_epochs)
    recorded = completed_epochs(run_dir)
    if args.retrain:
        print("[train] --retrain: clearing checkpoints/results/videos", flush=True)
        clear_run_state(run_dir)
        train_minutes = run_training(
            conda, args.conda_env, som_dir, data_dir, run_dir, args.num_epochs
        )
    elif recorded is not None:
        if recorded != requested:
            sys.exit(
                f"{run_dir} completed with epochs={recorded}, but epochs="
                f"{requested} was requested. Reusing would mislabel the result "
                "(and SoM resume would train 0 epochs). Use --retrain, or a "
                "different --run-dir."
            )
        print(f"[train] reusing completed run ({run_dir}, epochs={recorded})", flush=True)
    else:
        if (run_dir / "checkpoints" / "last.ckpt").exists():
            # Checkpoint without completion proof = interrupted run. SoM will
            # auto-resume from it — but only toward the schedule already in
            # the run's cfg.yaml. An explicit --num-epochs on a resume is
            # ambiguous (smaller values train 0 epochs and would then be
            # stamped complete), so require --retrain for that.
            if args.num_epochs is not None:
                sys.exit(
                    f"{run_dir} holds an interrupted run; resuming ignores "
                    "--num-epochs (SoM continues its saved schedule) and a "
                    "smaller value would train 0 epochs yet be stamped "
                    "complete. Drop --num-epochs to resume, or use --retrain."
                )
            print("[train] incomplete run detected — resuming training", flush=True)
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
        "epochs": completed_epochs(run_dir) or requested,
        "metrics": metrics,
        "videos": {"rgb": latest_rgb_video(run_dir)},
        "env": capture_env_versions(conda, args.conda_env),
        "train_minutes": round(train_minutes, 1) if train_minutes is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "reference_benchmark.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    # Union of both sides' metrics: known keys first (in canonical order),
    # then any method-specific extras — dropped keys would silently hide a
    # mismatch between two methods' emitters.
    metrics_a, metrics_b = a.get("metrics", {}), b.get("metrics", {})
    known = [k for k in METRIC_PATTERNS if k in metrics_a or k in metrics_b]
    extras = sorted((set(metrics_a) | set(metrics_b)) - set(known))
    keys = known + extras
    width = max(len(k) for k in keys) if keys else 10
    print(f"\nSequence: {seq}")
    print(f"  {'metric':<{width}}  {name_a:>14}  {name_b:>14}  {'better':>8}")

    def _usable(v) -> bool:
        # json.loads accepts NaN AND Infinity literals; bool is an int
        # subclass. Any of them would silently win/lose a direction
        # comparison, so only finite real numbers count.
        return (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
        )

    for key in keys:
        va, vb = metrics_a.get(key), metrics_b.get(key)
        sa = f"{va:.4f}" if _usable(va) else "—"
        sb = f"{vb:.4f}" if _usable(vb) else "—"
        better = ""
        if _usable(va) and _usable(vb) and va != vb:
            if key in HIGHER_IS_BETTER:
                better = name_a if (va > vb) == HIGHER_IS_BETTER[key] else name_b
            else:
                better = "?"  # unknown metric — direction not defined
        print(f"  {key:<{width}}  {sa:>14}  {sb:>14}  {better:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Processed sequence dir (iPhone/DyCheck format)")
    parser.add_argument("--run-dir", help="Training/eval output dir (reused when training completed there)")
    parser.add_argument("--seq-name", help="Sequence name (default: data-dir basename)")
    parser.add_argument("--som-dir", default=str(DEFAULT_SOM_DIR), help="Shape of Motion checkout")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Reference conda env name")
    parser.add_argument("--num-epochs", type=int, help="Override training epochs (smoke runs)")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Clear checkpoints/results/videos and train from scratch (a plain "
        "re-run RESUMES an incomplete run instead)",
    )
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

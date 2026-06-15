#!/usr/bin/env python3
"""Run a reference 4D reconstruction benchmark and emit a normalized result.

This is the room4d-side wrapper around the vendored Shape of Motion repo
(third_party/shape-of-motion, installed into an isolated conda env by
scripts/setup_reference_tool.sh). It turns a processed dataset sequence into a
normalized ``reference_benchmark.json`` that room4d's own renderer results can
be compared against.

Pipeline:
  1. validate the sequence directory layout (--dataset iphone|nvidia)
  2. train Shape of Motion in the reference env under an exclusive run-dir
     lock (skipped only when the run dir proves completion — sentinel or
     final-epoch videos; an interrupted run resumes toward the schedule
     recorded in its cfg.yaml; --retrain starts over)
  3. collect metrics — iphone: run evaluate_iphone.py and parse its stdout;
     nvidia: SoM ships no evaluate_nvidia.py, so read the trainer's
     final-epoch ``val/*`` scalars from the run dir's TensorBoard event files.
     NOTE the protocols differ: iphone eval scores 1x renders against 1x
     covisible masks, while the nvidia dataloader is hardcoded to factor=2
     (half resolution). The result JSON records this in its ``eval`` block so
     a future emitter can detect a protocol mismatch before comparing.
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

  # NVIDIA Dynamic Scenes sequence (monocular protocol: trains on camera 0,
  # scores novel views on held-out cameras 1-3)
  python scripts/run_reference_benchmark.py --dataset nvidia \
    --data-dir .../nvidia/Balloon1 \
    --run-dir  .../runs/balloon1-full

  # compare two results (e.g. room4d vs the reference)
  python scripts/run_reference_benchmark.py --compare room4d.json reference.json
"""

from __future__ import annotations

import argparse
import fcntl
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

# The reference_benchmark_v1 contract (schema id, metric directions, compare)
# lives in room4d so this Shape-of-Motion emitter and room4d's own renderer eval
# share one source of truth.
sys.path.insert(0, str(REPO_ROOT))
from room4d.eval.benchmarks.covisibility_masked_novel_view_synthesis import (  # noqa: E402
    HIGHER_IS_BETTER,
    SCHEMA,
    compare,
)
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

# SoM has no evaluate_nvidia.py; for --dataset nvidia the metrics come from
# the training-time validator instead, which run_training.py always invokes at
# the final epoch and logs only to TensorBoard event files in the work dir.
# Tag -> schema key. val/psnr|ssim|lpips map onto the masked_m* keys because
# they measure the same quantity as evaluate_iphone.py's "mean NV m*" lines:
# covisibility-masked novel-view metrics over the val split, computed with the
# same flow3d.metrics classes (the validator multiplies covisible_masks into
# valid_mask, and the nvidia val split loads covisible/2x/val/). fg_/bg_
# variants split that by the track_anything dynamic-object mask.
NVIDIA_TB_METRICS: dict[str, str] = {
    "val/psnr": "masked_mpsnr",
    "val/ssim": "masked_mssim",
    "val/lpips": "masked_mlpips",
    "val/fg_psnr": "fg_psnr",
    "val/fg_ssim": "fg_ssim",
    "val/fg_lpips": "fg_lpips",
    "val/bg_psnr": "bg_psnr",
    "val/bg_ssim": "bg_ssim",
    "val/bg_lpips": "bg_lpips",
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

# Verified against the pinned SoM nvidia dataloader and the downloaded
# Balloon1/Jumping scenes. The dataloader hardcodes factor=2, so the 2x
# variants are what training opens (plus rgb/1x for the intrinsics rescale);
# depth and 2d_tracks are train-split inputs, covisible/2x/val is what the
# val split loads for masked metrics. dataset.json/scene.json/colmap/ ship
# with the data but the nvidia loader never reads them — not required.
NVIDIA_REQUIRED_DIRS = [
    "rgb/1x",
    "rgb/2x",
    "camera",
    "splits",
    "depth/2x",
    "flow3d_preprocessed/2d_tracks/2x",
    "flow3d_preprocessed/track_anything/2x",
    "covisible/2x/val",
]
NVIDIA_REQUIRED_FILES = [
    "splits/train.json",
    "splits/val.json",
]

DATASET_LAYOUTS = {
    "iphone": ("iPhone/DyCheck", IPHONE_REQUIRED_DIRS, IPHONE_REQUIRED_FILES),
    "nvidia": ("NVIDIA Dynamic Scenes", NVIDIA_REQUIRED_DIRS, NVIDIA_REQUIRED_FILES),
}


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


def validate_sequence(data_dir: Path, dataset: str) -> None:
    label, required_dirs, required_files = DATASET_LAYOUTS[dataset]
    problems = []
    for d in required_dirs:
        if not (data_dir / d).is_dir():
            problems.append(f"missing dir:   {d}/")
    for f in required_files:
        if not (data_dir / f).exists():
            problems.append(f"missing entry: {f}")
    if problems:
        listing = "\n  ".join(problems)
        sys.exit(
            f"Sequence at {data_dir} does not look like a processed {label} "
            f"sequence:\n  {listing}\n"
            "Re-check the download (see scripts/verify_reference_data.sh)."
        )
    if dataset == "nvidia":
        # SoM's has_validation is content-gated, not existence-gated: an
        # empty val split trains fine but never logs val/* scalars, so the
        # harvest would fail only AFTER the full training spend.
        val_frames = _split_frame_names(data_dir, "val.json")
        if not val_frames:
            sys.exit(
                f"{data_dir / 'splits/val.json'} has an empty frame_names "
                "list — with no val split SoM logs no val/* metrics, and the "
                "benchmark would only fail after the full training run."
            )
        # The dataloader lazily np.loads pairwise track files for RANDOM
        # frame pairs per batch, so a partial download passes startup and
        # crashes mid-training at an arbitrary epoch. The full N^2 set is
        # checkable up front with one listdir + set membership.
        train_frames = _split_frame_names(data_dir, "train.json")
        tracks_dir = data_dir / "flow3d_preprocessed" / "2d_tracks" / "2x"
        present = {p.name for p in tracks_dir.iterdir()}
        missing_pairs = [
            f"{fi}_{fj}.npy"
            for fi in train_frames
            for fj in train_frames
            if f"{fi}_{fj}.npy" not in present
        ]
        if missing_pairs:
            sys.exit(
                f"{tracks_dir} is missing {len(missing_pairs)} of "
                f"{len(train_frames) ** 2} pairwise track files (e.g. "
                f"{missing_pairs[0]}) — the dataloader loads RANDOM pairs "
                "per batch, so training would crash mid-run. Re-download "
                "the sequence."
            )


def _split_frame_names(data_dir: Path, split_file: str) -> list[str]:
    """frame_names from splits/<split_file>, exiting on malformed content."""
    path = data_dir / "splits" / split_file
    try:
        split = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        sys.exit(f"{path} is unreadable or invalid JSON: {exc}")
    if not isinstance(split, dict):
        sys.exit(f"{path} top level is not an object — re-download the sequence.")
    return split.get("frame_names", [])


# SoM's built-in num_epochs (run_training.py TrainConfig) — the schedule a
# run trains toward when no --num-epochs is passed.
SOM_DEFAULT_EPOCHS = 500


def _epochs_label(num_epochs: int | None) -> str:
    """Canonical label for a training schedule ('default' = SoM's built-in).

    An explicit --num-epochs 500 and a plain run train the SAME schedule, so
    they must share one label — otherwise a resume (which stamps 'default')
    followed by an explicit-500 rerun would spuriously abort on a schedule
    mismatch that doesn't exist.
    """
    if num_epochs is None or num_epochs == SOM_DEFAULT_EPOCHS:
        return "default"
    return str(num_epochs)


def trained_num_epochs(run_dir: Path) -> int | None:
    """The schedule a run dir's cfg.yaml records, or None when unavailable.

    cfg.yaml is the only durable record of an interrupted run's schedule:
    SoM writes it at training start and never re-reads it, so resume must
    recall the value from here and pass it back explicitly.
    """
    cfg = run_dir / "cfg.yaml"
    if not cfg.is_file():
        return None
    try:
        text = cfg.read_text()
    except OSError as exc:
        print(f"[warn] could not read {cfg}: {exc}", flush=True)
        return None
    match = re.search(r"^num_epochs:\s*(\d+)", text, re.M)
    return int(match.group(1)) if match else None


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
    epochs = trained_num_epochs(run_dir)
    if epochs is not None and (run_dir / "videos" / f"epoch_{epochs - 1:04d}").is_dir():
        # Label SoM's default schedule accordingly so reuse against a
        # default-schedule request matches.
        return "default" if epochs == SOM_DEFAULT_EPOCHS else str(epochs)
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
        # cfg.yaml is the OLD run's record; run_training rewrites it only
        # after the long dataset load, so leaving it would false-positive the
        # provenance guard if a retrain dies during load. code/ is SoM's
        # per-invocation source backup — unbounded disk growth across retrains.
        "cfg.yaml", "code",
        TRAINING_COMPLETE_SENTINEL, "reference_benchmark.json", "benchmark_eval.log",
    ):
        path = run_dir / entry
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    # TensorBoard event files too: a retrain restarts global_step at 0, so a
    # stale event file's final step would outrank the new run's and the
    # nvidia metric harvest would resurrect the OLD run's val/* values.
    for event_file in run_dir.glob("events.out.tfevents.*"):
        event_file.unlink()


RUN_LOCK_NAME = ".room4d_benchmark.lock"


def acquire_run_lock(run_dir: Path) -> int:
    """Exclusive per-run-dir flock; exits when another process holds it.

    Two concurrent invocations sharing a run dir interleave checkpoints and
    TensorBoard event files into plausible-looking corruption (the harvest's
    same-step check detects it only after the fact). The lock is an
    fcntl.flock on the returned fd — kernel-owned, so it releases on ANY
    process death (kill -9 included) with no PID-liveness or stale-reclaim
    protocol to race. The training child inherits the fd (pass_fds), so an
    orphaned trainer keeps the run dir locked until it actually exits. The
    lock FILE persists on disk (never unlinked — unlinking would detach
    waiters from the locked inode); its PID content is diagnostics only.
    """
    lock_path = run_dir / RUN_LOCK_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = ""
        try:
            holder = lock_path.read_text().strip()
        except OSError:
            pass
        os.close(fd)
        sys.exit(
            f"{lock_path} is locked by another running benchmark"
            + (f" (pid {holder})" if holder else "")
            + " — wait for it to finish or use a different --run-dir."
        )
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def run_training(
    conda: str, env: str, som_dir: Path, data_dir: Path, run_dir: Path,
    num_epochs: int | None, dataset: str, lock_fd: int | None = None,
) -> float:
    """Train SoM on the sequence; returns wall-clock minutes."""
    cmd = conda_run_cmd(
        conda, env, "python", "run_training.py",
        "--work-dir", str(run_dir),
    )
    if num_epochs is not None:
        cmd += ["--num-epochs", str(num_epochs)]
    cmd += [f"data:{dataset}", "--data.data-dir", str(data_dir)]
    print(f"[train] {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    # The child inherits the run lock's fd: if the wrapper dies (kill -9)
    # the orphaned trainer keeps the run dir locked until it exits, so a
    # second invocation cannot interleave with it.
    extra = {"pass_fds": (lock_fd,)} if lock_fd is not None else {}
    subprocess.run(cmd, cwd=som_dir, check=True, **extra)
    # Written only on success (check=True raises first) and records the
    # schedule, so completion claims are schedule-specific. Atomic replace:
    # a crash mid-write must not leave an empty sentinel that reads as a
    # completed default-schedule run.
    sentinel_tmp = run_dir / (TRAINING_COMPLETE_SENTINEL + ".tmp")
    sentinel_tmp.write_text(_epochs_label(num_epochs) + "\n")
    os.replace(sentinel_tmp, run_dir / TRAINING_COMPLETE_SENTINEL)
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


# Runs inside the reference env (torch lives there, not in the caller's env).
# weights_only=True only: SoM checkpoints are plain dicts of tensors/ints
# (trainer.save_checkpoint), and a weights_only=False fallback would hand
# pickle code-execution to anything dropped into a run dir.
_CKPT_EPOCH_SNIPPET = """\
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(ckpt["epoch"]))
"""


def checkpoint_epoch(conda: str, env: str, ckpt: Path) -> int | None:
    """The epoch a SoM checkpoint was saved at, or None when unreadable.

    SoM resumes training from this value (trainer.init_from_checkpoint), so
    it is what decides whether a resume has any epochs left to train.
    """
    try:
        proc = subprocess.run(
            conda_run_cmd(conda, env, "python", "-c", _CKPT_EPOCH_SNIPPET, str(ckpt)),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=120,  # a metadata read must not wedge the pipeline
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


# Runs inside the reference env (tensorboard is a SoM dependency there).
# Prints one JSON object mapping every scalar tag to its highest-step event —
# train/* tags included, for provenance in benchmark_eval.log. size_guidance
# scalars=0 keeps ALL events; the default caps at 10k per tag and the cap
# discards a RANDOM subsample, which could silently drop the final-epoch one.
_TB_HARVEST_SNIPPET = """\
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
out = {}
for tag in acc.Tags()["scalars"]:
    last = max(acc.Scalars(tag), key=lambda e: e.step)
    out[tag] = {"value": last.value, "step": last.step}
print(json.dumps(out))
"""


def harvest_nvidia_metrics(conda: str, env: str, run_dir: Path) -> dict[str, float]:
    """Final-epoch val metrics from the run dir's TensorBoard event files.

    The highest step per tag is the final-epoch value: run_training.py always
    validates at epoch == num_epochs - 1, and clear_run_state removes event
    files on --retrain so a stale file can never outrank the current run.
    EventAccumulator merges every event file in the dir, which is what makes
    resumed runs (one event file per process) read correctly.
    """
    cmd = conda_run_cmd(
        conda, env, "python", "-c", _TB_HARVEST_SNIPPET, str(run_dir)
    )
    print(f"[eval] harvesting TensorBoard scalars from {run_dir}", flush=True)
    proc = subprocess.run(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    log = stdout + (f"\n--- stderr ---\n{stderr}" if stderr else "")
    (run_dir / "benchmark_eval.log").write_text(log)
    if proc.returncode != 0:
        tail = "\n".join((stderr or stdout).splitlines()[-25:])
        sys.exit(f"TensorBoard harvest failed (exit {proc.returncode}). Tail:\n{tail}")
    try:
        tags = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        sys.exit(
            "TensorBoard harvest printed no JSON — see "
            f"{run_dir / 'benchmark_eval.log'}"
        )

    # The validator logs all val/* scalars in one log_dict call at a single
    # global_step, so a step disagreement means the event files mix more than
    # one training run (e.g. two processes shared the run dir) — plausible-
    # looking but corrupt numbers that nothing downstream could detect.
    steps = {tag: tags[tag]["step"] for tag in NVIDIA_TB_METRICS if tag in tags}
    if len(set(steps.values())) > 1:
        sys.exit(
            f"val/* scalars disagree on their final step ({steps}) — the "
            "event files mix more than one training run. Use --retrain to "
            "clear the run dir."
        )

    metrics: dict[str, float] = {}
    for tag, key in NVIDIA_TB_METRICS.items():
        if tag not in tags:
            print(f"[warn] scalar tag not found in event files: {tag}", flush=True)
            continue
        value = float(tags[tag]["value"])
        if not math.isfinite(value):
            # A NaN/Inf scalar (degenerate validation) must not be recorded
            # as the baseline: json.dumps would emit a non-RFC literal that
            # compare() then renders as missing anyway. Treat as absent.
            print(f"[warn] non-finite value for {tag}: {value!r}", flush=True)
            continue
        metrics[key] = value
    if not any(k.startswith("masked_") for k in metrics):
        sys.exit(
            "No val/* scalars found in the TensorBoard event files — training "
            "may not have reached its final-epoch validation (or the val split "
            f"is empty). See {run_dir / 'benchmark_eval.log'}"
        )
    return metrics


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


def trained_data_dir(run_dir: Path) -> str | None:
    """The data.data-dir a run dir was trained against, from its cfg.yaml.

    SoM writes cfg.yaml at training start; ``data_dir`` is nested under the
    ``data:`` block (the only indented data_dir key — work_dir is top-level).
    None when no cfg.yaml exists (pre-training run dir) or the key is absent.
    """
    cfg = run_dir / "cfg.yaml"
    if not cfg.is_file():
        return None
    try:
        text = cfg.read_text()
    except OSError as exc:
        print(f"[warn] could not read {cfg}: {exc} — provenance check skipped", flush=True)
        return None
    match = re.search(r"^\s+data_dir:\s*(.+)$", text, re.M)
    if not match:
        # Fail OPEN but never silently: a cfg.yaml format change must not
        # quietly disarm the run-relabeling protection.
        print(f"[warn] no data_dir key in {cfg} — provenance check skipped", flush=True)
        return None
    # yaml.dump quotes paths containing special characters
    return match.group(1).strip().strip("'\"")


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
    validate_sequence(data_dir, args.dataset)
    run_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = acquire_run_lock(run_dir)
    try:
        _locked_benchmark(args, data_dir, run_dir, som_dir, seq_name, lock_fd)
    finally:
        # sys.exit paths unwind through here too. Closing the fd releases
        # the flock (the kernel also releases it on any death) — the lock
        # FILE stays on disk by design.
        os.close(lock_fd)


def _locked_benchmark(
    args: argparse.Namespace, data_dir: Path, run_dir: Path, som_dir: Path,
    seq_name: str, lock_fd: int,
) -> None:
    """The run-dir-mutating part of benchmark(); caller holds the run lock."""
    conda = find_conda()

    if not args.retrain:
        # A run dir only proves metrics for the sequence it was trained on.
        # Without this, reusing a finished run with the wrong --data-dir (or
        # --dataset) would silently relabel its metrics as the new sequence —
        # the harvest path reads only the run dir and could never notice.
        recorded_data_dir = trained_data_dir(run_dir)
        if recorded_data_dir is not None and Path(recorded_data_dir) != data_dir:
            sys.exit(
                f"{run_dir} was trained on {recorded_data_dir}, but --data-dir "
                f"is {data_dir}. Reusing would relabel that run's metrics as "
                "this sequence. Use a different --run-dir, or --retrain to "
                "overwrite."
            )

    train_minutes: float | None = None
    resumed = False
    resumed_from_epoch: int | None = None
    requested = _epochs_label(args.num_epochs)
    recorded = completed_epochs(run_dir)
    if args.retrain:
        print(
            "[train] --retrain: clearing checkpoints/results/videos, event "
            "files, and prior benchmark artifacts",
            flush=True,
        )
        clear_run_state(run_dir)
        train_minutes = run_training(
            conda, args.conda_env, som_dir, data_dir, run_dir, args.num_epochs,
            args.dataset, lock_fd,
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
        epochs_for_training = args.num_epochs
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
        if ckpt_path.exists():
            # Checkpoint without completion proof = interrupted run. SoM
            # resumes from the checkpoint's epoch toward whatever schedule
            # THIS invocation passes — cfg.yaml is written but never re-read
            # (run_training.py trains range(ckpt_epoch, num_epochs) with the
            # CLI/default value) — so the wrapper recalls the recorded
            # schedule itself and pre-checks the checkpoint epoch against
            # it: a resume with zero epochs left must never be stamped
            # complete.
            if args.num_epochs is not None:
                sys.exit(
                    f"{run_dir} holds an interrupted run; resume recalls its "
                    "recorded schedule from cfg.yaml, so an explicit "
                    "--num-epochs is ambiguous. Drop --num-epochs to resume, "
                    "or use --retrain for a clean run of the requested "
                    "schedule."
                )
            schedule = trained_num_epochs(run_dir)
            if schedule is None:
                # Fail CLOSED: guessing a schedule could silently retarget
                # the run (e.g. an interrupted 1000-epoch run resumed toward
                # an assumed 500 ships as a completed 'default' baseline).
                sys.exit(
                    f"{run_dir} holds an interrupted run but its cfg.yaml is "
                    "missing or records no schedule — cannot tell what the "
                    "run was training toward. Use --retrain for a clean run."
                )
            ckpt_epoch = checkpoint_epoch(conda, args.conda_env, ckpt_path)
            if ckpt_epoch is None:
                print(
                    "[warn] could not read the checkpoint's epoch — resume "
                    "proceeds unchecked. If SoM also fails to load it, the "
                    "checkpoint is likely corrupt (a killed torch.save); "
                    "recover with --retrain.",
                    flush=True,
                )
            elif ckpt_epoch >= schedule:
                sys.exit(
                    f"{run_dir} checkpoint is at epoch {ckpt_epoch}, already "
                    f">= its recorded {schedule}-epoch schedule — resuming "
                    "would train zero epochs yet stamp the run complete. The "
                    "run likely finished training but died before being "
                    "stamped; use --retrain for a clean run, or a different "
                    "--run-dir."
                )
            print(
                "[train] incomplete run detected — resuming from epoch "
                f"{'?' if ckpt_epoch is None else ckpt_epoch} toward its "
                f"recorded {schedule}-epoch schedule",
                flush=True,
            )
            resumed = True
            resumed_from_epoch = ckpt_epoch
            # The recalled schedule is passed back explicitly; passing None
            # for SoM's built-in value keeps the training command minimal
            # (_epochs_label folds 500 and None into 'default' either way).
            epochs_for_training = None if schedule == SOM_DEFAULT_EPOCHS else schedule
        train_minutes = run_training(
            conda, args.conda_env, som_dir, data_dir, run_dir,
            epochs_for_training, args.dataset, lock_fd,
        )

    # Written by the validator during final-epoch validation for BOTH formats
    # (evaluate_iphone.py additionally reads the renders from it).
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        sys.exit(
            f"{results_dir} missing — training did not produce eval renders "
            "(interrupted run?). Re-run with --retrain."
        )

    if args.dataset == "nvidia":
        metrics = harvest_nvidia_metrics(conda, args.conda_env, run_dir)
    else:
        metrics, _ = run_eval(
            conda, args.conda_env, som_dir, data_dir, run_dir, seq_name
        )

    result = {
        "schema": SCHEMA,
        "method": "shape-of-motion",
        "method_commit": som_pinned_commit(som_dir),
        "sequence": seq_name,
        "dataset_format": args.dataset,
        # Protocols differ per format; recorded so a cross-emitter compare
        # can detect a mismatch (nvidia: SoM's dataloader hardcodes factor=2,
        # half resolution; iphone: evaluate_iphone.py scores at 1x).
        "eval": (
            {"source": "training-validator-tensorboard", "resolution_factor": 2}
            if args.dataset == "nvidia"
            else {"source": "evaluate_iphone", "resolution_factor": 1}
        ),
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "epochs": completed_epochs(run_dir) or requested,
        "metrics": metrics,
        "videos": {"rgb": latest_rgb_video(run_dir)},
        "env": capture_env_versions(conda, args.conda_env),
        # When resumed, train_minutes covers only the final segment — the
        # resume fields keep the provenance honest.
        "resumed": resumed,
        "resumed_from_epoch": resumed_from_epoch,
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
    # Canonical order first, then format-specific extras (nvidia fg_/bg_ keys).
    ordered = list(METRIC_PATTERNS) + [k for k in metrics if k not in METRIC_PATTERNS]
    for key in ordered:
        if key in metrics:
            arrow = "↑" if HIGHER_IS_BETTER.get(key, True) else "↓"
            print(f"  {key:<{width}}  {metrics[key]:>10.4f}  {arrow}")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Processed sequence dir (see --dataset for the expected format)")
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_LAYOUTS), default="iphone",
        help="Processed dataset format of --data-dir (default: iphone)",
    )
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
    if args.num_epochs is not None and args.num_epochs < 1:
        parser.error("--num-epochs must be >= 1")
    benchmark(args)


if __name__ == "__main__":
    main()

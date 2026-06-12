# TODOS

## Reference Comparison Tool

### Proper resume handling in run_reference_benchmark.py

**What:** Make resume of an interrupted run recall the original schedule
(read cfg.yaml's `num_epochs` and pass it explicitly), and refuse to stamp
the completion sentinel when a resume trained zero epochs.

**Why:** Verified during the dataset-backends ship review: SoM never re-reads
cfg.yaml — resume trains `range(ckpt_epoch, cli num_epochs)` with the CLI
default of 500. A plain resume of an interrupted smoke run silently retargets
to 500 epochs, and a resume whose checkpoint epoch exceeds the requested
schedule trains 0 epochs, exits 0, and gets stamped complete — publishing
mid-training metrics as a finished baseline. The wrapper's guard text was
corrected (2026-06-11) but the underlying semantics remain.

**Context:** Found by the /ship red-team review of feature/dataset-backends.
The interim guard: explicit `--num-epochs` on a resume aborts with a
corrected message recommending --retrain.

**Effort:** S-M
**Priority:** P1
**Depends on:** None

### Lockfile guard against concurrent runs sharing a run dir

**What:** Take an exclusive lockfile (or pidfile) in the run dir for the
duration of training + harvest in `run_reference_benchmark.py`; exit early
with a clear message when held.

**Why:** Two simultaneous invocations against the same run dir interleave
checkpoints and TensorBoard event files; the nvidia harvest's max-step
selection would then mix the two runs' scalars nondeterministically into a
plausible-looking but corrupt result.

**Context:** Found by the /ship red-team review of feature/dataset-backends.
Plausible solo-workflow accident (forgotten background run + re-run in a new
shell); recovery is --retrain. The harvest now aborts when val/* scalars
disagree on their final step (mixed-run detector), but prevention needs the
lock. Related: SoM's dataloader also writes caches into the DATA dir
(`flow3d_preprocessed/cache/scene_norm_dict.pth`), so two concurrent runs of
the same sequence into different run dirs race on that file too — a run-dir
lock alone won't cover it (and the data dir must be writable, which nothing
documents).

**Effort:** S
**Priority:** P2
**Depends on:** None

### Score hloc+COLMAP calibration against dataset GT extrinsics

**What:** Run the phase-1 calibration (`room4d/calibration/`) on DyCheck/NVIDIA
sequences and score recovered extrinsics against the datasets' ground-truth
camera poses.

**Why:** Both benchmark datasets ship GT extrinsics, which turns the calibration
stage from "feels stubbed" into something with a real correctness test, for free.

**Context:** Deferred from plan (design doc "bonus" item) at ship time of the
reference comparison tool. Needs pose-error metrics (rotation/translation deltas
after alignment) and, for NVIDIA, the dataset download above.

**Effort:** M
**Priority:** P1
**Depends on:** None (NVIDIA Balloon1 + Jumping downloaded and verified 2026-06-11; DyCheck worked already)

## Renderer (Track 2)

### Emit reference_benchmark_v1 from room4d's renderer evaluation

**What:** Make room4d's renderer eval (`room4d/eval/rendering_metrics.py`) emit
the `reference_benchmark_v1` JSON (masked mPSNR/mSSIM/mLPIPS + tracking
metrics) so `run_reference_benchmark.py --compare` can score room4d against
the Shape of Motion baseline.

**Why:** The schema is the comparison contract; until room4d emits it, the
documented cross-method compare flow has only one real emitter.

**Context:** room4d's RenderingMetrics computes raw PSNR only today (no SSIM/
LPIPS/co-visibility masking). Adopt the schema when the real gsplat renderer
lands (design doc Track 2).

**Effort:** M
**Priority:** P2
**Depends on:** room4d gsplat renderer (Track 2)

## Completed

### Download NVIDIA Dynamic Scenes dataset + extend benchmark wrapper

**What:** Download the processed NVIDIA Dynamic Scenes data (Drive link in
`scripts/verify_reference_data.sh`) and extend `scripts/run_reference_benchmark.py`
beyond iphone-format (`data:nvidia` plumbing).

**How it landed:** Balloon1 + Jumping downloaded manually and verified
(layout, per-frame consistency, full N² pairwise track sets). The wrapper
gained `--dataset {iphone,nvidia}`; nvidia metrics are harvested from the
trainer's final-epoch TensorBoard `val/*` scalars (SoM has no
`evaluate_nvidia.py`). Proven on GPU with a 5-epoch Jumping smoke run.
Note: the processed download ships 4 of the rig's 12 cameras with a fixed
1-train/3-val split — the N=2/4/8 input sweep from the original "Why" needs
custom splits later (Track 2).

**Completed:** v0.1.3.0 (2026-06-11)

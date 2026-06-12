# TODOS

## Reference Comparison Tool

### Shared data-dir cache race between concurrent runs of one sequence

**What:** SoM's dataloader writes caches into the DATA dir
(`flow3d_preprocessed/cache/scene_norm_dict.pth`), so two concurrent
benchmarks of the same sequence into *different* run dirs race
writer-vs-reader on that file (torch.load of a half-written file crashes;
worst case the val split normalizes with the other process's transform —
silently skewed metrics). The run-dir flock (v0.1.3.1) does not cover this.
Also: the data dir must be writable, which nothing documents or validates.

**Why:** Residual risk split out of the lockfile work — the run-dir lock
covers same-run-dir contention but not same-data-dir contention.

**Context:** Found by the /ship adversarial review of
feature/dataset-backends; deferred when the run-dir lock shipped.

**Effort:** S
**Priority:** P3
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

### Proper resume handling in run_reference_benchmark.py

**What:** Make resume of an interrupted run recall the original schedule and
refuse zero-epoch completion stamps.

**How it landed:** Resume reads cfg.yaml's `num_epochs` and passes it back
explicitly (SoM never re-reads it); a missing schedule fails closed; the
checkpoint's epoch is read via the reference env (`weights_only=True`,
120s timeout) and a resume with zero epochs left aborts before training.
`--num-epochs 500` and plain runs share the canonical `default` label; the
sentinel is written atomically; resumed runs record
`resumed`/`resumed_from_epoch` in the result JSON. GPU-verified (cfg-bump
resume trained 2→4 and stamped "4").

**Completed:** v0.1.3.1 (2026-06-12)

### Lockfile guard against concurrent runs sharing a run dir

**What:** Exclusive per-run-dir lock for the duration of training + harvest.

**How it landed:** `fcntl.flock` held on an fd for the wrapper's lifetime —
kernel-owned, releases on any death (kill -9 included), no PID/reclaim
protocol to race (the hand-rolled O_EXCL+PID design was rejected in review
for TOCTOU reclaim races). The training child inherits the fd via
`pass_fds`, so an orphaned trainer keeps the run dir locked until it exits.
Live-verified: held lock refuses a second invocation; released lock
proceeds. Residual same-data-dir cache race split into its own P3 entry.

**Completed:** v0.1.3.1 (2026-06-12)

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

# TODOS

## Reference Comparison Tool

### Download NVIDIA Dynamic Scenes dataset + extend benchmark wrapper

**What:** Download the processed NVIDIA Dynamic Scenes data (Drive link in
`scripts/verify_reference_data.sh`) and extend `scripts/run_reference_benchmark.py`
beyond iphone-format (`data:nvidia` plumbing).

**Why:** NVIDIA's 12-camera rig is the multi-view input testbed — feed N of 12
cameras as input streams, hold out the rest as GT, sweep N (2/4/8) to measure
how novel-view quality climbs with more streams. Directly exercises room4d's
per-stream pipeline (multi-cam roadmap).

**Context:** Deferred from plan (design doc, Track 1) at ship time of the
reference comparison tool — its consumers are Track 2 work (renderer multi-view
sweep) and the calibration-GT bonus below. Validator + Drive links already in
place; data download is manual (browser).

**Effort:** S (download + dataloader flag)
**Priority:** P1
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
**Depends on:** NVIDIA Dynamic Scenes dataset (for the multi-cam case; DyCheck works today)

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

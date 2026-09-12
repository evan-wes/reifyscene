# Co-visibility-masked novel-view-synthesis benchmark

A cross-method benchmark for **dynamic-scene novel-view synthesis**: render a
reconstruction from held-out cameras and score it against ground truth **only
over the co-visible pixels**, then compare methods (reifyscene vs. Shape-of-Motion
vs. published numbers) on identical sequences.

## What it measures (the DyCheck protocol)

This is the evaluation protocol introduced by **DyCheck** — *"Monocular Dynamic
View Synthesis: A Reality Check"* (Gao, Li, Tulsiani, Russell, Kanazawa,
NeurIPS 2022) — and implemented in Shape-of-Motion's `scripts/evaluate_iphone.py`
/ `flow3d/metrics.py`. The defining idea: a monocular reconstruction can only be
fairly scored where the held-out view's content was actually observed by the
training camera, so each held-out frame ships a **co-visibility mask** and all
metrics are computed over the `1`-region of that mask (hence the `m` prefix —
*masked*):

- **`masked_mpsnr`** (↑) — per-frame masked MSE pooled to PSNR, averaged over
  frames.
- **`masked_mssim`** (↑) — mask-aware Gaussian-window SSIM, averaged over frames.
- **`masked_mlpips`** (↓) — spatial LPIPS (AlexNet) over masked images, averaged
  over frames.

Optional 3D/2D tracking metrics (`track3d_*`, `track2d_*`) are part of the same
DyCheck suite; reifyscene's renderer does not emit them yet (left `null`), so a
compare shows them only for methods that do.

reifyscene's masked-metric math is ported verbatim from `flow3d/metrics.py` into
`reifyscene/eval/metrics/image.py` (`MaskedPSNR`/`MaskedSSIM`/`MaskedLPIPS`), so the
numbers are directly comparable to the SoM/published baselines. Only torch +
torchmetrics are required.

## Datasets

Both shipped benchmark datasets provide the co-visibility masks under
`covisible/{factor}x/val/` and a `splits/val.json`:

- **DyCheck iPhone** (`dataset_format: "iphone"`) — casual monocular sequences
  (e.g. `paper-windmill`); the moving training camera + 2 held-out static
  cameras. The hard v1 benchmark.
- **NVIDIA Dynamic Scenes** (`dataset_format: "nvidia"`) — multi-camera rig
  (e.g. `Balloon1`); a subset of cameras held out as novel views.

## The record contract (`reference_benchmark_v1`)

`schema.py` is the single source of truth. A record:

```json
{
  "schema": "reference_benchmark_v1",
  "method": "reifyscene" | "shape-of-motion" | ...,
  "sequence": "paper-windmill",
  "dataset_format": "iphone" | "nvidia",
  "eval": {"source": "...", "resolution_factor": 1 | 2},
  "metrics": {"masked_mpsnr": ..., "masked_mssim": ..., "masked_mlpips": ...},
  "...": "method-specific provenance (run_dir, env, timestamp, ...)"
}
```

`eval` records the measurement protocol so a compare can warn when two records
were produced under different protocols (e.g. 1× vs 2× resolution), whose
numbers are not directly comparable.

## Who emits it

- **Shape-of-Motion** — `scripts/run_reference_benchmark.py` trains/evaluates SoM
  in its reference conda env and harvests the metrics. Pre-computed baselines
  live under `data/shape_of_motion/runs/*/reference_benchmark.json` (e.g.
  `paper-windmill-full`: masked_mpsnr 19.53 / mssim 0.559 / mlpips 0.198).
- **reifyscene** — `scripts/run_render_proof.py` (the `train` flow) renders the
  trained Gaussian field at the held-out cameras and emits a record via
  `build_record(...)`.

## Comparing

```bash
python scripts/run_reference_benchmark.py --compare \
    data/shape_of_motion/runs/paper-windmill-full/reference_benchmark.json \
    <reifyscene_run>/reference_benchmark.json
```

`compare()` prints a metric-by-metric table with a per-row "better" verdict,
warning on sequence or eval-protocol mismatch.

## Caveats

- **Resolution.** Co-visibility masks ship only at `2x` for these sequences;
  score at `factor: 2` for an apples-to-apples comparison.
- **Static vs. dynamic.** A static reconstruction is scored on the same masked
  region that includes moving content it cannot reconstruct, so its
  `masked_mpsnr` is motion-capped and will sit below a dynamic method's. That
  gap is the expected signal, not a bug.

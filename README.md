# reifyscene

**Dynamic 4D reconstruction from casually captured, uncalibrated video** — one clip or
several, of a moving scene, turned into a renderable, novel-view-capable 4D (dynamic
Gaussian) representation.

*To reify* is to make something abstract concrete and manipulable. The aim is a scene
representation you can **interpret, query, and semantically edit** — not an opaque
neural blob. ([Renamed from `room4d`.](CHANGELOG.md#0201---2026-09-10))

Licensed under Apache 2.0.

reifyscene is a **research / learning project** (not a product): the aim is to build the whole
reconstruction pipeline end to end and understand every stage. Today the single-video
dynamic-4D path works end to end on the pipeline's own reconstructed geometry; the
queryable-scene-graph back-half and multi-stream fusion are designed but not yet built
(see [Status](#status)).

## Quick start

```bash
scripts/setup_reifyscene.sh --with-dev     # editable install + pinned deps
reifyscene --help
```

## Pipeline

A single-video run flows through 15 stages (`reifyscene/api/stages.py`), driven by
`scripts/run_single_video_reifyscene.py`:

```
ingest → sync → dyn_mask → geometry → pose_refine → segment → reid → semantics
→ objects → humans → tracking → graph → optimization → scene → render
```

- **geometry** — VGGT / VGGT-Long camera + depth + point-map reconstruction (or a
  ground-truth "oracle" backend for isolating downstream quality).
- **pose_refine** — robust bundle adjustment of the camera poses (gauge re-anchored to
  the input scale).
- **perception band** — SAM-2 segmentation, re-identification, semantics, and object +
  human perception.
- **tracking** — CoTracker point tracks lifted to 3D.
- **scene → render** — assemble a dynamic Gaussian scene (moving foreground + rigid-motion
  bases + static background), train it, and render a novel-view flythrough, with a
  covisibility-masked novel-view (mPSNR / SSIM / LPIPS) evaluation.

## Status

reifyscene reconstructs a **single-video dynamic 4D scene end to end** and scores its novel
views. As of **v0.2.0.0** the dynamic-4D path holds on the pipeline's own (non-oracle)
geometry at full frame count.

**Works today:**
- End-to-end single-video pipeline → a trained dynamic Gaussian scene + novel-view
  flythrough + masked-NVS metrics.
- Real backends for geometry (VGGT / VGGT-Long), pose refinement (BA), segmentation
  (SAM-2), re-id, object/human perception, point tracking (CoTracker), and dynamic 4D
  rendering.

**Partial / scaffold:**
- The scene **graph** stage builds camera + per-stream track nodes, but relation and
  uncertainty inference and cross-view entity association are still placeholders.
- The joint **optimization** stage is a staged scaffold — `pose_refine` does the real
  bundle adjustment today.

**Designed, not yet built:**
- The **queryable scene graph** back-half — natural-language query → plan → render an
  answer → diff the re-perceived result against the original.
- **Multi-stream** capture (cross-view synchronization + fusion).
- Live-camera ingest directly into the pipeline.


## Reproducible setup

### Core package

```bash
scripts/setup_reifyscene.sh --with-dev
```

This installs the local `reifyscene` package in editable mode and pins the current
Python dependencies used by the pipeline.

### Optional RealSense SDK

```bash
scripts/setup_realsense_sdk.sh
```

This is **not required** for early RGB-only testing. If your D435/D435i shows up
as a standard camera device on Linux, you can capture RGB clips with OpenCV via:

```bash
python scripts/capture_uvc_video.py --device 0 --output data/test_rgb.mp4 --show
```

If `pyrealsense2` is available, you can also capture through the RealSense SDK:

```bash
python scripts/capture_realsense_rgb.py --output data/test_rgb.mp4 --show
```

## Running a single-video reconstruction

```bash
scripts/run_single_video_reifyscene.py \
  --frames-dir path/to/frames \        # or --video path/to/clip.mp4
  --geometry-backend-name vggt-long \  # vggt | vggt-long | dataset (GT oracle)
  --geometry-device cuda \
  --keyframes-all \                    # train on every frame (benchmark parity)
  --pose-refine-enable \
  --config configs/base.yaml \
  --run-dir artifacts/runs/my_run
```

Each stage persists a self-describing artifact under the run dir (reloadable, e.g.
`GeometryArtifact.load(...)`). `--start-stage` / `--stop-stage` + `--prior-run-dir` resume
from a prior run's cached stages — e.g. re-run only `scene` / `render` without recomputing
geometry. The final render writes `render/trained_dynamic.npz` + a novel-view flythrough,
and the run records a covisibility-masked novel-view score.

### Results (v0.2.0.0)

Masked novel-view PSNR on the DyCheck "spin" sequence (full held-out set):

| geometry | frames | masked mPSNR |
|---|---|---|
| VGGT-Long + pose_refine (native) | 426 (all) | 12.56 |
| GT-oracle (upper bound) | 426 (all) | ~19 |
| Shape-of-Motion reference | all | 17.42 |

Held-out quality is limited by the reconstruction's geometry (native chunk-stitch pose
drift), not the dynamic modeling; the remaining gap to the oracle is the cost of
self-reconstructed vs. ground-truth geometry.


## Reference comparison tool (Shape of Motion)

To benchmark reifyscene's renderer against a known-good 4D reconstruction method,
the repo can vendor [Shape of Motion](https://github.com/vye16/shape-of-motion)
into `third_party/shape-of-motion` and run it in an **isolated** conda
environment (it pins different torch/CUDA than the main `reifyscene` env and must
never share it).

### Setup

```bash
scripts/setup_reference_tool.sh            # creates conda env reifyscene_som_ref
```

This installs an sm_120-capable torch (cu128), a matching CUDA toolkit,
Shape of Motion at a pinned commit (with small patches for modern torch/gsplat,
applied reproducibly and verified by sentinel checks), and gsplat built from
source at a pinned commit. Safety guards refuse to target the `reifyscene` or
`base` envs.

Both project envs install their pip trees from full lockfiles in
`scripts/locks/` by default (pass `--unlocked` to re-resolve; regenerate after
deliberate upgrades with `scripts/gen_env_locks.sh`).

### Data

Datasets are downloaded manually in a browser (Google Drive links are in the
Shape of Motion README and in `scripts/verify_reference_data.sh`). Validate an
extracted sequence with:

```bash
scripts/verify_reference_data.sh --dataset iphone --seq-dir /path/to/paper-windmill
scripts/verify_reference_data.sh --dataset nvidia --seq-dir /path/to/Balloon1
```

For nvidia sequences the check is deep: full directory layout, a non-empty
val split (an empty one trains for hours and yields no metrics), and the
complete N² pairwise track set (the dataloader reads random pairs per batch,
so a partial download would otherwise crash mid-training).

### Benchmark

```bash
# DyCheck iPhone sequence (default: --dataset iphone)
python scripts/run_reference_benchmark.py \
  --data-dir /path/to/iphone/paper-windmill \
  --run-dir  /path/to/runs/paper-windmill-full

# NVIDIA Dynamic Scenes sequence
python scripts/run_reference_benchmark.py --dataset nvidia \
  --data-dir /path/to/nvidia/Balloon1 \
  --run-dir  /path/to/runs/balloon1-full
```

Trains Shape of Motion, evaluates, and writes a normalized
`reference_benchmark.json` (`reference_benchmark_v1` schema: masked
mPSNR/mSSIM/mLPIPS, 2D/3D tracking metrics where the dataset's eval provides
them, env versions, video paths). The metric source differs per dataset:
iphone runs SoM's `evaluate_iphone.py`; SoM ships no `evaluate_nvidia.py`, so
nvidia metrics are harvested from the trainer's final-epoch `val/*`
TensorBoard scalars (same covisibility-masked novel-view quantities, plus
`fg_`/`bg_` splits). Each result records its eval protocol in an `eval` block
(source and resolution factor — nvidia trains/evaluates at the dataloader's
hardcoded 2x, iphone at 1x).
Run handling: a run dir whose training demonstrably completed (completion
sentinel, or final-epoch videos for older runs) is reused — but only for the
sequence it was trained on (reusing it with a different `--data-dir` aborts);
an interrupted run resumes toward the schedule recorded in its `cfg.yaml`
(recalled and passed back explicitly — SoM itself never re-reads it), and a
resume whose checkpoint is already at or past that schedule aborts rather
than stamping a zero-epoch run complete; `--retrain` clears all prior run
state (checkpoints, results, videos, TensorBoard event files) and starts
from scratch. Each invocation holds an exclusive kernel lock (`fcntl.flock`)
on `.reifyscene_benchmark.lock` in its run dir, so a forgotten concurrent run
can't interleave checkpoints and event files into corrupt metrics. The lock
releases on any process death (`kill -9` included), and the training child
inherits it — an orphaned trainer keeps the run dir locked until it exits.
The lock file itself persists on disk; its contents are diagnostics only.

### Compare

```bash
python scripts/run_reference_benchmark.py --compare reifyscene.json reference.json
```

Prints a side-by-side table with per-metric better-direction (unknown metrics
are shown with an undefined direction rather than dropped), and warns when
the two results' sequences or eval protocols differ (e.g. nvidia's 2x vs
iphone's 1x) instead of silently declaring winners on incomparable numbers. The
`reference_benchmark_v1` schema is the common contract for scoring any two
results on the same sequence with one command. reifyscene's render stage now
computes masked novel-view metrics (mPSNR / SSIM / LPIPS) directly (see the
Results table above); emitting them in this exact schema for a one-command
`--compare` against a reference run is the remaining wiring.

Reference baseline on DyCheck iPhone `paper-windmill` (RTX 5090, 500 epochs,
~69 min): masked mPSNR 19.53 / mSSIM 0.559 / mLPIPS 0.198, 3D-track PCK@10cm
0.996 — consistent with the published Shape of Motion results.

MoSca as a second (pose-free) reference is deliberately deferred;
`--with-mosca` in the setup script errors until that port is scoped.

## Calibration GT scoring

Both benchmark datasets ship ground-truth camera poses, so the hloc+COLMAP
calibration stage (`reifyscene/calibration/`) can be scored against them
directly — no reference training run needed:

```bash
# NVIDIA static rig (default: 5 timesteps per camera)
python scripts/score_calibration_gt.py --dataset nvidia \
  --data-dir /path/to/nvidia/Balloon1 \
  --run-dir  /path/to/runs/balloon1-calib

# Known-intrinsics protocol: inject the shared GT intrinsics into COLMAP
# and disable refinement (refuses rigs whose intrinsics differ per camera,
# e.g. DyCheck's per-frame ARKit focals)
python scripts/score_calibration_gt.py --dataset nvidia --fix-intrinsics \
  --data-dir /path/to/nvidia/Balloon1 \
  --run-dir  /path/to/runs/balloon1-calib-fixed
```

`--dataset iphone` scores a DyCheck capture the same way (24 frames sampled
along the trajectory by default). The scorer runs in the main `reifyscene` env
(calibration extras install by default in `scripts/setup_reifyscene.sh`, which
also patches the vendored hloc for pycolmap >= 3.10; `--no-calibration`
skips them).

The collector copies the selected frames into the run dir and writes
`gt_manifest.json` (image filename → GT pose) beside the image dir; after
SfM the scorer joins recovered poses against that manifest and writes
`calibration_gt_score.json` (`calibration_gt_score_v1` schema). Primary
metrics are gauge-free pairwise-relative errors (per-pair rotation
geodesic, baseline direction, baseline ratio); Sim(3)-aligned APE on camera
centers is secondary, guarded for ill-conditioned rigs, because fitting 7
DOF to a handful of cameras absorbs a flattering share of the true error.
Fragmented reconstructions are scored on their largest alignable component
with coverage reported, and `--fix-intrinsics` runs record whether the
injection actually held (`intrinsics_match_injected`), so a silent
fallback can't pose as a fixed-intrinsics result.

Measured so far: Balloon1 cross-camera pairwise rotation error mean 0.176°
(free intrinsics) / 0.136° (known-intrinsics protocol); DyCheck
`paper-windmill` 1.89° across 3 cameras against ARKit odometry.

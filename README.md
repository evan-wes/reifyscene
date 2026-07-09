# room4d

Starter codebase scaffold for a room-scale dynamic 4D reconstruction system.

Licensed under Apache 2.0

This repository is intentionally a skeleton:
- package structure is in place
- artifacts and dataclasses are defined
- wrappers, solvers, and renderers are stubs with docstrings
- CLI entrypoints are wired for future implementation

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
room4d --help
```

## Repository goals

The scaffold is organized around these stages:

1. ingest and preprocessing
2. synchronization
3. geometry initialization
4. tracking
5. human and object perception
6. graph construction
7. joint spatiotemporal optimization
8. 4D rendering
9. promptable query and camera planning

## Status

This is a prototype engineering skeleton, not a working reconstruction system yet.


## Reproducible setup

### Core package

```bash
scripts/setup_room4d.sh --with-dev
```

This installs the local `room4d` package in editable mode and pins the current
Python dependencies used by the scaffold.

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

## Current single-video capability

With one RGB video source, the current code can:
- decode the video and probe metadata with OpenCV
- sample keyframes
- run the VGGT adapter path for coarse camera/depth/point-map initialization
  when the official VGGT package is installed, or fall back to a heuristic stub
- emit a `TrackArtifact` with placeholder tracks
- build a minimal graph with camera and per-stream track nodes
- run a staged optimization scaffold that records factor diagnostics but does not
  yet solve a real bundle-adjustment problem
- initialize a renderer artifact shell

It cannot yet:
- ingest a live camera stream directly into the pipeline
- run a real CoTracker backend in the tracker wrapper
- recover a trustworthy final 4D reconstruction from one video alone
- use RealSense depth in the pipeline
- do object/human perception or real graph-conditioned reconstruction end to end


## Single-video VGGT geometry path

To validate the current real-geometry slice on a single RGB video, run:

```bash
python scripts/run_single_video_vggt.py \
  --video path/to/clip.mp4 \
  --run-dir artifacts/runs/my_single_video \
  --backend-mode real
```

If the official `vggt` package is not installed locally, switch `--backend-mode auto` to fall back gracefully. A successful real run writes:

- `artifacts/runs/.../geometry/artifact.json`
- `artifacts/runs/.../geometry/depthmaps/*.npz`
- `artifacts/runs/.../geometry/pointmaps/*.npz`

The persisted `GeometryArtifact` can then be reloaded with `GeometryArtifact.load(...)` for downstream tracking, graph building, or optimization.


## Reference comparison tool (Shape of Motion)

To benchmark room4d's renderer against a known-good 4D reconstruction method,
the repo can vendor [Shape of Motion](https://github.com/vye16/shape-of-motion)
into `third_party/shape-of-motion` and run it in an **isolated** conda
environment (it pins different torch/CUDA than the main `room4d` env and must
never share it).

### Setup

```bash
scripts/setup_reference_tool.sh            # creates conda env room4d_som_ref
```

This installs an sm_120-capable torch (cu128), a matching CUDA toolkit,
Shape of Motion at a pinned commit (with small patches for modern torch/gsplat,
applied reproducibly and verified by sentinel checks), and gsplat built from
source at a pinned commit. Safety guards refuse to target the `room4d` or
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
on `.room4d_benchmark.lock` in its run dir, so a forgotten concurrent run
can't interleave checkpoints and event files into corrupt metrics. The lock
releases on any process death (`kill -9` included), and the training child
inherits it — an orphaned trainer keeps the run dir locked until it exits.
The lock file itself persists on disk; its contents are diagnostics only.

### Compare

```bash
python scripts/run_reference_benchmark.py --compare room4d.json reference.json
```

Prints a side-by-side table with per-metric better-direction (unknown metrics
are shown with an undefined direction rather than dropped), and warns when
the two results' sequences or eval protocols differ (e.g. nvidia's 2x vs
iphone's 1x) instead of silently declaring winners on incomparable numbers. The
`reference_benchmark_v1` schema is the contract room4d's own renderer
evaluation **will emit** when the real renderer lands (today `room4d/eval/`
computes raw PSNR only — the emitter is tracked in TODOS.md), so any two
results on the same sequence can be scored with one command.

Reference baseline on DyCheck iPhone `paper-windmill` (RTX 5090, 500 epochs,
~69 min): masked mPSNR 19.53 / mSSIM 0.559 / mLPIPS 0.198, 3D-track PCK@10cm
0.996 — consistent with the published Shape of Motion results.

MoSca as a second (pose-free) reference is deliberately deferred;
`--with-mosca` in the setup script errors until that port is scoped.

## Calibration GT scoring

Both benchmark datasets ship ground-truth camera poses, so the hloc+COLMAP
calibration stage (`room4d/calibration/`) can be scored against them
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
along the trajectory by default). The scorer runs in the main `room4d` env
(calibration extras install by default in `scripts/setup_room4d.sh`, which
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

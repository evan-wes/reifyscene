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
applied reproducibly), and gsplat built from source. Safety guards refuse to
target the `room4d` or `base` envs.

### Data

Datasets are downloaded manually in a browser (Google Drive links are in the
Shape of Motion README and in `scripts/verify_reference_data.sh`). Validate an
extracted sequence with:

```bash
scripts/verify_reference_data.sh --dataset iphone --seq-dir /path/to/paper-windmill
```

### Benchmark

```bash
python scripts/run_reference_benchmark.py \
  --data-dir /path/to/iphone/paper-windmill \
  --run-dir  /path/to/runs/paper-windmill-full
```

Trains Shape of Motion (skipped automatically when the run dir already holds a
checkpoint; `--retrain` forces), evaluates, and writes a normalized
`reference_benchmark.json` (`reference_benchmark_v1` schema: masked
mPSNR/mSSIM/mLPIPS, 2D/3D tracking metrics, env versions, video paths).

### Compare

```bash
python scripts/run_reference_benchmark.py --compare room4d.json reference.json
```

Prints a side-by-side table with per-metric better-direction. The
`reference_benchmark_v1` schema is the contract that room4d's own renderer
evaluation emits against, so any two results on the same sequence can be
scored with one command.

Reference baseline on DyCheck iPhone `paper-windmill` (RTX 5090, 500 epochs,
~69 min): masked mPSNR 19.53 / mSSIM 0.559 / mLPIPS 0.198, 3D-track PCK@10cm
0.996 — consistent with the published Shape of Motion results.

MoSca as a second (pose-free) reference is deliberately deferred;
`--with-mosca` in the setup script errors until that port is scoped.

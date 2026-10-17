# reifyscene

**Dynamic 4D reconstruction from casually captured, uncalibrated video** — one clip or
several, of a moving scene, turned into a renderable, novel-view-capable 4D (dynamic
Gaussian) representation.

*To reify* is to make something abstract concrete and manipulable. The aim is a scene
representation you can **interpret, query, and semantically edit** — not an opaque
neural blob. ([Renamed from `room4d`.](CHANGELOG.md#0201---2026-09-10))

Licensed under Apache 2.0.

reifyscene is a **research / learning project** (not a product): the aim is to build the whole
reconstruction pipeline end to end and understand every stage. Today one video goes end to end,
on the pipeline's own reconstructed geometry: a dynamic 4D Gaussian scene, a scene graph of the
people and objects in it, structured queries over that graph, and removing an entity from the
scene. Multi-stream fusion and natural-language queries are not built yet (see
[Status](#status)).

## Quick start

```bash
scripts/setup_reifyscene.sh --with-dev     # editable install + pinned deps
conda env config vars set -n reifyscene REIFYSCENE_DATA_DIR=/path/to/reifyscene_data
conda activate reifyscene                  # again, so the variable is set
reifyscene --help
```

## The data directory

Everything the pipeline reads or writes lives in one folder, named by the `REIFYSCENE_DATA_DIR`
environment variable; nothing in the repository names a folder on your machine. Its layout:

```
datasets/     inputs, each laid out as it ships (davis2017/, dycheck_iphone/,
              nvidia_dynamic/, panoptic/), and clips of your own
checkpoints/  model weights the pipeline fetches (tapnet/)
runs/         pipeline runs
reference/    what the reference tools produce (Shape-of-Motion runs, ConceptGraphs exports)
cache/        inputs a stage derives and rebuilds when missing
archive/      kept, read by nothing
```

Commands name things inside it rather than by path: `--frames-dir dycheck_iphone/spin/rgb/1x`
is a folder in `datasets/`, and `--run-dir my_run` is `runs/my_run`. A full path is refused.
A run records its inputs by these names too, so the whole folder can be moved or renamed and
every run in it still opens.

## Pipeline

A single-video run flows through 16 stages (`reifyscene/api/stages.py`), driven by
`scripts/run_single_video_reifyscene.py`:

```
ingest → sync → dyn_mask → geometry → pose_refine → segment → tracking → reid
→ semantics → objects → humans → graph → optimization → scene → render → handles
```

- **geometry** — VGGT / VGGT-Long camera + depth + point-map reconstruction (or a
  ground-truth "oracle" backend for isolating downstream quality).
- **pose_refine** — robust bundle adjustment of the camera poses (gauge re-anchored to
  the input scale).
- **perception band** — SAM-2 segmentation, re-identification, semantics, and object +
  human perception.
- **tracking** — CoTracker point tracks lifted to 3D.
- **graph** — the scene's entities (people, objects, background), their positions over time,
  and the relations between them in each frame.
- **scene → render** — assemble a dynamic Gaussian scene (moving foreground + rigid-motion
  bases + static background), train it, and render a novel-view flythrough, with a
  covisibility-masked novel-view (mPSNR / SSIM / LPIPS) evaluation.
- **handles** — which of the trained scene's Gaussians belong to each entity, so an entity can
  be rendered alone or removed.

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

- A **queryable scene graph**: its entities own their Gaussians in the trained scene, and
  relations between them are computed per frame with VLA-3D's relation functions. Structured
  queries (`reifyscene query`: select, count, when, relate) and removing an entity
  (`reifyscene edit remove`) work on a finished run, and a local web app (`reifyscene serve`)
  shows the graph, its relations over time and the scene in 3D, and runs removals.

**Known limits (measured):**
- **Removing a person leaves a visible copy of them.** On DAVIS `lab-coat`, the removal's
  floater filter keeps a coarse layer of large Gaussians that draws about a third of the
  person's pixels, so the "after" render still shows them. The cause is that training never
  separates one entity's Gaussians from another's; a fix that gives every Gaussian a fixed owner
  from the moment it is created is designed, not built.
- Learned per-Gaussian identity is built but off by default, and the body-volume bound for
  people is experimental and off.

**Partial / scaffold:**
- The joint **optimization** stage is a staged scaffold — `pose_refine` does the real
  bundle adjustment today.

**Not built yet:**
- **Natural-language** queries (queries are structured templates) and edits other than removal.
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
python scripts/capture_uvc_video.py --device 0 --output my_clips/test_rgb.mp4 --show
```

If `pyrealsense2` is available, you can also capture through the RealSense SDK:

```bash
python scripts/capture_realsense_rgb.py --output my_clips/test_rgb.mp4 --show
```

## Running a single-video reconstruction

```bash
python scripts/run_single_video_reifyscene.py \
  --frames-dir dycheck_iphone/spin/rgb/1x \   # or --video my_clips/clip.mp4 (in datasets/)
  --geometry-backend-name vggt-long \         # vggt | vggt-long | dataset (GT oracle)
  --geometry-device cuda \
  --keyframes-all \                           # train on every frame (benchmark parity)
  --pose-refine-enable \
  --config configs/base.yaml \
  --run-dir my_run                            # runs/my_run
```

Each stage persists a self-describing artifact under the run dir (reloadable, e.g.
`GeometryArtifact.load(...)`). `--start-stage` / `--stop-stage` + `--prior-run-dir` resume
from a prior run's cached stages — e.g. re-run only `scene` / `render` without recomputing
geometry. The final render writes the trained scene + a novel-view flythrough, and the run
records a covisibility-masked novel-view score.

A resume loads a prior stage only when it is what this run would make: every artifact records
the settings its stage used (from whichever config sections it reads), the input frames it read,
its contract version and the artifacts it was built from, and each stage lists what a loaded
artifact must carry. The whole resume is checked before anything is written. What can be derived
from what is loaded is (a moved input or dataset folder, missing photos, a crop box, re-lifted
tracks), cheap stages that run no model are re-run, and anything else is refused with the stage
to restart from. Add `--check-resume` to see that report — every prior stage's verdict and the
latest `--start-stage` the prior run supports — without running or writing anything.

`configs/base.yaml` builds a **static** scene (`render/trained_gaussians.npz`). The dynamic
4D scene (`render/trained_dynamic.npz`) needs `scene: {kind: dynamic}` in the config, as in
`configs/experiment/spin_native_full.yaml`; `configs/experiment/davis_graph_loop.yaml` is the
configuration the scene graph and removal are developed on.

### Exploring a finished run

```bash
reifyscene serve                                  # the local app over runs/, at http://127.0.0.1:8765
reifyscene query my_run select category=person    # also: count, when, relate
reifyscene edit remove my_run 8                   # an entity's node id or its tracklet number
```

### Results

Masked novel-view PSNR on the DyCheck "spin" sequence (all 722 held-out views):

| geometry | scene | frames | masked mPSNR | mLPIPS |
|---|---|---|---|---|
| VGGT-Long + pose_refine (native) | dynamic | 426 (all) | 13.97 ¹ | 0.53 |
| VGGT-Long + pose_refine (native) | static | 426 (all) | 13.34 ¹ | 0.62 |
| GT-oracle (upper bound) | dynamic | 426 (all) | 19.65 | 0.34 |
| GT-oracle | static | 426 (all) | 16.22 ¹ | 0.32 |
| Shape-of-Motion reference | dynamic | all | 17.42 | — |

¹ Held-out cameras are placed through a similarity (rotation, translation, scale) fitted on the
training cameras, whose poses are known in both frames — the usual way to score a method that
estimates its own cameras. A self-reconstructed scene lives in its own frame, and a static scene
in its geometry's ground-plane-rotated one (for the GT oracle the fit is exact). The fit absorbs
one global offset; per-camera drift still counts (median rotation left after the fit: 0.60° for
the native rows), and so do parts of the room a held-out camera sees that VGGT-Long's centre crop
never did.

Both native rows come from one fresh run of `configs/experiment/spin_native_full.yaml`, scored
before pose refinement was made repeatable; the geometry a refinement lands on moves the score,
and another converged run of the same config scored 14.96 dynamic / 14.56 static. Pose refinement
now repeats: its tracker always did (three runs, byte-identical seed models), and the bundle
adjustment does too once it stops refining cameras the observations cannot constrain and runs
single-threaded. A run whose bundle adjustment *diverges* is refused — it keeps the unrefined
poses and says so — rather than quietly scoring 12.67.

Held-out quality is limited by the reconstruction's geometry (native chunk-stitch pose
drift), not the dynamic modeling; the remaining gap to the oracle is the cost of
self-reconstructed vs. ground-truth geometry.

Reproduce the native row with `configs/experiment/spin_native_full.yaml` (the full command is
in its header). To see a trained scene rather than only its score, export held-out views, a
replay and the score with `scripts/export_checkpoint_views.py RUN --dataset-root DATASET`.


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
scripts/verify_reference_data.sh --dataset iphone \
  --seq-dir "$REIFYSCENE_DATA_DIR"/datasets/dycheck_iphone/paper-windmill
scripts/verify_reference_data.sh --dataset nvidia \
  --seq-dir "$REIFYSCENE_DATA_DIR"/datasets/nvidia_dynamic/Balloon1
```

For nvidia sequences the check is deep: full directory layout, a non-empty
val split (an empty one trains for hours and yields no metrics), and the
complete N² pairwise track set (the dataloader reads random pairs per batch,
so a partial download would otherwise crash mid-training).

### Benchmark

```bash
# DyCheck iPhone sequence (default: --dataset iphone); the run goes to reference/som_runs/
python scripts/run_reference_benchmark.py \
  --data-dir dycheck_iphone/paper-windmill \
  --run-dir  som_runs/paper-windmill-full

# NVIDIA Dynamic Scenes sequence
python scripts/run_reference_benchmark.py --dataset nvidia \
  --data-dir nvidia_dynamic/Balloon1 \
  --run-dir  som_runs/balloon1-full
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
python scripts/run_reference_benchmark.py --compare \
  runs/my_run/reference_benchmark.json reference/som_runs/paper-windmill-full/reference_benchmark.json
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
  --data-dir nvidia_dynamic/Balloon1 \
  --run-dir  balloon1-calib

# Known-intrinsics protocol: inject the shared GT intrinsics into COLMAP
# and disable refinement (refuses rigs whose intrinsics differ per camera,
# e.g. DyCheck's per-frame ARKit focals)
python scripts/score_calibration_gt.py --dataset nvidia --fix-intrinsics \
  --data-dir nvidia_dynamic/Balloon1 \
  --run-dir  balloon1-calib-fixed
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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(loose variant), and this project uses a 4-digit `MAJOR.MINOR.PATCH.MICRO`
version scheme (canonical version in the `VERSION` file, read by
`pyproject.toml` at build time).

## [0.1.17.0] - 2026-06-25

### Added
- **F0a — renderable core: a self-contained, renderable `SceneArtifact` + a
  trainer-driven render backend** (first phase of the full-pipeline architecture,
  design doc `docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`).
  The render stage now **trains a real 3DGS scene and emits real novel views**
  in-pipeline, replacing the no-train placeholder, with a clean separation of
  concerns: the **scene stage** turns the reconstruction into a self-contained
  renderable scene; the **render stage consumes only that artifact**.
  - `room4d/scene/scene_artifact.py`: re-cut to a self-contained renderable
    contract — a dense Gaussian `render_seed` (npz), per-frame `train_views` with
    the **GT frame persisted into the artifact** (portable/reproducible), a
    novel-view `render_trajectory`, a `kind` (static/dynamic), and a
    `dynamic_bundle` placeholder (F0b). The toy `render_contract` /
    `layered_scene_v2` / `static_scene` / `dynamic_entities` schema is gone.
  - `room4d/scene/assembler.py`: re-cut (1046 → ~210 lines). Builds the dense seed
    (`seed_arrays_from_geometry`, the new numpy-only seed path factored out of
    `seed_cloud_from_geometry`) + train views + an orbit trajectory. Sheds the
    toy sprite/template/entity machinery (entity world-anchoring is a graph-stage
    concern, deferred to P3). Tolerant `**cfg` constructor (legacy `scene:` keys
    are ignored).
  - `room4d/render/gaussian4d/render_backend.py` (new) `GaussianRenderBackend`:
    seed → `GaussianField` → `StaticGaussianTrainer.fit(train_views)` → trajectory
    render + a trained-Gaussian checkpoint → `RenderArtifact`. A degenerate scene
    (no depth / GT frames) emits an empty render (`scene_state['empty']`) instead
    of crashing, so the pipeline stays robust on smoke/degenerate inputs.
  - `room4d/render/eval/masked_nvs_benchmark.py` (new): in-pipeline masked-NVS
    (DyCheck protocol) — scores the trained checkpoint against a benchmark
    dataset's held-out val cameras + covisibility masks, reusing the proof
    harness's `score_masked_novel_views` path. Dataset-coupled: runs only when an
    eval dataset source is configured (`cfg['evaluation']['masked_nvs']`), returns
    `{}` on arbitrary video. Wired into `Room4DPipeline.fit_renderer`.

### Changed
- **Render stage rewired** to the new contract: `render/registry.py`,
  `render/__init__.py`, and `Room4DPipeline.fit_renderer` now drive
  `GaussianRenderBackend` (the proven `StaticGaussianTrainer`) instead of the
  deleted toy.

### Removed
- The no-train toy renderer — `render/gaussian4d/scene_model.py`
  (`Gaussian4DScene`) and `render/gaussian4d/trainer.py` (`Gaussian4DTrainer`).

### Tests
- `tests/test_scene_render.py` rewritten against the new contract (assembler
  builds a self-contained renderable scene; save/load round-trip persists GT
  frames; render backend trains+renders+checkpoints, gsplat-gated; dynamic kind
  rejected until F0b). `tests/test_eval_metrics.py` rewritten to the masked-NVS
  graceful-empty contract. Full suite: 260 passed, 1 skipped.

### Validated
- In-pipeline masked mPSNR **18.54** on DyCheck paper-windmill (static; 277 train
  frames, 7000 iters, 419 val views) **reproduces the standalone
  `run_render_proof.py` static 18.51** — F0a static parity proven. The dynamic
  branch (the +2.2 dB toward the 20.7 dynamic number) is F0b.

## [0.1.16.0] - 2026-06-20

### Added
- **Pose-refinement stage (`pose_refine`)** — a new, backend-agnostic pipeline
  stage that bundle-adjusts the geometry camera poses, inserted between
  `geometry` and `tracking` so every downstream stage consumes the refined
  poses. It consumes the `GeometryArtifact` contract (poses + intrinsics +
  per-pixel world points) plus the source frames, runs the vendored VGGSfM
  tracker, seeds a `pycolmap` reconstruction with the backend's poses / K /
  world points and the visible tracks, and runs COLMAP's iterative robust BA
  (Cauchy loss + reprojection-outlier filtering, **refine shared focal, principal
  point fixed**); it re-emits a refined `GeometryArtifact` and re-derives the
  interpolated tracking-frame camera states from the refined keyframes. No
  geometry backend is re-run. New module `room4d/geometry/pose_refine.py`
  (`PoseRefiner`); pipeline method `Room4DPipeline.refine_poses`; stage wired
  into `STAGE_ORDER` / `build_stage_registry` (now 11 stages) with the six
  downstream stages rewired to depend on `pose_refine`. Why BA closes the
  native-geometry pose gap when photometric pose-opt could not: BA minimizes
  *reprojection* error of feature correspondences — a signal tied to the true
  scene independent of the current geometry — breaking the pose↔geometry gauge
  ambiguity. Validated in the step-12 spike (Path B): paper-windmill pairwise
  rotation **1.73° → 0.26°** and masked mPSNR **17.01 → 19.58** (+2.57, ~69% of
  the gap to the GT-pose oracle 20.75); spin **4.0° → 1.08°**. Robust BA rejects
  dynamic-foreground tracks on its own at typical dynamic fractions, so **no
  foreground mask is generated** (the spike ablation found masking is not
  load-bearing: none 1.078° ≈ image-mask 1.062° ≈ motion-mask 1.098°). Refining
  the principal point is harmful (drifts ~9px, wrecks render registration) and
  stays off. Opt-in and **off by default** (`--pose-refine-enable`); when
  disabled or when the refinement runtime is unavailable, the stage is an
  identity pass-through that re-emits the geometry unchanged so downstream has a
  stable dependency either way. Resumable via the existing stage window
  (`--start-stage pose_refine --prior-run-dir <cached geometry run>`) to re-tune
  BA without re-running geometry. New `pose_refine` extra in `pyproject.toml`
  (`pycolmap`, `hydra-core`, `omegaconf`, `scipy`) installed by default in
  `scripts/setup_room4d.sh` (`--no-pose-refine` to skip); `hydra-core` /
  `omegaconf` are required by the vendored VGGSfM tracker and were previously
  undeclared.

## [0.1.15.3] - 2026-06-19

### Fixed
- VGGT-Long loop closure (`--vggt-long-loop-enable`) was non-functional in the
  room4d setup — doubly broken: (1) `LoopModels/backbones/dinov2.py`
  `torch.hub.load`s `./LoopModels/dinov2` relative to the **process CWD**, so it
  fails whenever VGGT-Long runs from any directory but its own (always, under
  room4d); and (2) `faiss` (place-recognition retrieval) is stripped from
  VGGT-Long's pip requirements. `scripts/setup_room4d.sh` now installs the
  conda-forge **faiss-cpu** build (`ensure_faiss`, pinned `FAISS_VERSION=1.9.0`)
  and applies an idempotent patch anchoring the DINOv2 path to the module's own
  location via `__file__`. With both, loop closure runs end-to-end (verified on
  paper-windmill: 5 loop constraints, Sim3 loop optimization). Loop closure is
  opt-in and off by default; on a forward-sweep clip it left pose error unchanged
  (its lever is accumulated drift, not per-frame noise), but it's now available
  for clips that genuinely revisit places.

## [0.1.15.2] - 2026-06-18

### Fixed
- VGGT-Long camera poses were globally inconsistent. VGGT-Long estimates
  per-chunk Sim(3) transforms (overlap registration + optional loop closure)
  and applies them **only** when writing `camera_poses.txt`; the per-chunk
  `_tmp_results_aligned/*.npy` files keep **chunk-local** extrinsics (only
  `world_points` are globalized there). `VGGTLongInitializer` merged those
  chunk-local extrinsics, so an N-chunk sequence became N independently-anchored
  mini-trajectories — each chunk's poses were self-consistent but anchored at
  its own identity frame, with the camera trajectory jumping at every chunk
  boundary. The wrapper now reads VGGT-Long's authoritative globally-aligned
  `camera_poses.txt` (new `_load_global_camera_to_world`) for the camera poses,
  with a graceful fallback to the chunk-local extrinsics when the file is absent
  or its frame count mismatches; the already-global `world_points` and
  camera-frame `depth` are unchanged. On the paper-windmill 277-frame train clip
  this drops the stored-pose error from **pairwise rot 20.5° / sim3 134°** to
  **1.73° / 10.2°** — matching VGGT-Long's native trajectory exactly, vs the
  COLMAP ground truth. Plain VGGT (single forward pass, no chunking) was never
  affected. Found via a throwaway de-risk spike that scored the geometry stage
  end-to-end against the dataset GT before wiring VGGT-Long into the render path.

## [0.1.15.1] - 2026-06-18

### Changed
- `scripts/setup_room4d.sh` now installs the **ffmpeg CLI** into the conda env
  (conda-forge, pinned `FFMPEG_VERSION=8.1.1`, GPL build → includes `libx264`
  for `-crf 0` lossless H.264). Default-on with a `--no-ffmpeg` opt-out; skipped
  in `--use-venv` mode. ffmpeg is a **dataset-prep convenience** — assembling
  image-sequence datasets (DyCheck, DAVIS) into the mp4s the pipeline ingests;
  room4d's own decode path uses OpenCV and does not require it. A post-install
  smoke test asserts `libx264` is actually present (the non-GPL conda-forge
  variant omits it). This removes the prior need to borrow ffmpeg from another
  conda env by absolute path.

## [0.1.15.0] - 2026-06-18

### Added
- `run_single_video_room4d.py` gained a `--stop-stage` flag (inclusive), the
  complement to the existing `--start-stage`. The runner was resume-forward only
  — pick a start, run to `render`. Now any contiguous stage window `[start, stop]`
  runs: stages before `start` load from `--prior-run-dir`, stages in the window
  run, stages after `stop` are skipped. Default is the final stage, so existing
  invocations are unchanged. This lets the real pipeline run e.g.
  `ingest → sync → geometry` and stop, instead of needing a throwaway driver to
  exercise a single stage (e.g. the step-12 VGGT geometry de-risk). Window
  resolution + validation (unknown stage; stop-before-start) live in a unit-tested
  helper `room4d.api.resume.resolve_stage_window`.

### Fixed
- Partial runs no longer crash the end-of-run summary: the per-stage print loop
  indexed every stage in `STAGE_ORDER` unconditionally (`KeyError` once a run
  stops early); it now skips stages absent from `outputs`. The run manifest also
  records `stop_stage` alongside `start_stage` for reproducibility.

## [0.1.14.0] - 2026-06-17

### Added
- Render-path geometry-provider seam (step 12, PR 2) — the first step toward
  replacing the borrowed DyCheck geometry with room4d's own reconstruction. New
  `room4d/lifting/geometry_provider.py`: a `GeometryProvider` ABC
  (`cameras(split)` + `frame_stack(split)`) with `DyCheckGeometryProvider` — the
  borrowed dataset ground truth, now the permanent oracle/baseline a native
  backend (VGGT, future) is scored against in the same harness/eval.
  `build_native_init_inputs` / `NativeMotionInitializer` / `build_native_init`
  gained an optional `provider` (defaults to a `DyCheckGeometryProvider` built
  from `dataset_root`/`factor`/`depth_type`/`camera_type`), so the geometry source
  is swappable. **Behavior-preserving:** the DyCheck provider delegates to the
  existing loaders unchanged — verified **byte-identical** (277-frame train
  `FrameStack` + 419 val cameras) against the old direct path. The e2e through the
  provider scores masked mPSNR **20.68** (mSSIM 0.670, mLPIPS 0.158, 419 val
  frames), matching the native baseline (20.70–20.74) within GPU training
  run-to-run noise — the byte-identical inputs are the rigorous lossless proof.
- A throwaway de-risking spike (outside the repo) measured the VGGT-vs-GT geometry
  gap on paper-windmill before committing to the design: VGGT poses land within
  ~0.5° relative rotation of COLMAP (median pairwise rot 0.45°, sim3 abs rot 1.98°,
  one global up-to-scale factor s≈0.83); depth AbsRel ~0.02 with a consistent
  global scale. So step 12 is integration + a scale-alignment hook, not a
  quality-closing research effort.

### Changed
- `TODOS.md`: marked the calibration S1 kyotovision spike **DONE** (verdict
  CONTINUE-qualified — robustness YES, scale QUALIFIED YES, the D15 "person
  complements SfM" reframe) and reframed the post-spike entry from the superseded
  "M0-hardening + M1 batch" into the now-justified "M2/M3 joint-opt fusion" build.

### Tests
- `tests/test_geometry_provider.py` — the `GeometryProvider` ABC contract, the
  `DyCheckGeometryProvider` config/backend wiring, that it delegates to the
  existing loaders with identical args (lossless by construction), and that
  `build_native_init_inputs` accepts the `provider` override.

## [0.1.13.0] - 2026-06-17

### Added
- Native 2D point tracker — closes the **last borrowed input** in the motion-init
  path (only the dataset's RGB/depth/poses remain dataset-provided). A faithful
  port of Shape-of-Motion's `preproc/compute_tracks_torch.py`, wired as a real
  plug-and-play tracker backend alongside the CoTracker wrapper:
  - `room4d/tracking/tapnet/` vendors the DeepMind TAPIR torch model **byte-for-byte**
    (re-diffable against upstream; Apache-2.0 headers kept). `einshape` is added to
    the `vision` extra + `scripts/locks/room4d.lock` specifically to keep the port
    byte-faithful. `scripts/fetch_tapir_checkpoint.sh` fetches the bootstapir/tapir
    checkpoints (idempotent, size-guarded) to `data/tapnet/checkpoints/`.
  - `room4d/tracking/tapir_wrapper.py::TAPIRWrapper` is now a real `PointTracker`
    backend (model_type bootstapir|tapir): per-frame foreground-masked query grid
    (grid_size 4, SoM's value), 256×256 model input, all-pairs tracking to
    `(x, y, occ_logit, dist_logit)`.
  - `TrackArtifact` gains a first-class all-pairs payload (`room4d/tracking/all_pairs.py::AllPairsTracks`)
    with SoM `2d_tracks/<name_i>_<name_j>.npy` save/load and a `write_2d_tracks`
    helper. The extension is additive — the single-query CoTracker/scaffold path is
    unaffected.
  - `lift_tracks_3d` / `build_native_init_inputs` / `NativeMotionInitializer` /
    `build_native_init` gained a `tracks_dir` override; `generate_native_tracks`
    bridges the train-split frames+masks to the tracker. `run_render_proof.py dynamic
    --own-tracker [--own-tracks-dir DIR]` produces the init with no borrowed tracks,
    and `scripts/compute_tracks_native.py` dumps the tracks standalone.
- Validated on paper-windmill (factor 2): the native tracker reproduces the shipped
  tracks' foreground seeding **exactly** (1739/1739 diagonal query coords) and agrees
  with the shipped tracks to ~1–2px on nearby frames (looser over long temporal gaps,
  as expected for two independent tracker runs on a fast-moving scene). The bar that
  matters is end-to-end: masked mPSNR **20.70** (mSSIM 0.668, mLPIPS 0.161, 419 val
  frames) vs borrowed-init 20.68 / native-borrowed-tracks 20.74 — parity with no
  borrowed tracks.

### Tests
- `tests/test_tapir_wrapper.py` — `AllPairsTracks` shape validation / filename slicing
  / round-trip; `TrackArtifact` all-pairs save/load (+ empty path unaffected) and
  `write_2d_tracks` stream selection; wrapper checkpoint-path resolution, config
  defaults, and missing-checkpoint error; plus a GPU+checkpoint-gated model forward
  (exact FG-seeded diagonal coords + finite occ/dist logits).

## [0.1.12.0] - 2026-06-16

### Added
- Native motion initializer (PR-B; closes the "own motion init" follow-up) — the
  borrowed SoM init is now fully reproducible natively. `render/gaussian4d/
  procrustes_init.py` ports SoM's clustering + Procrustes + Gaussian placement
  (`solve_procrustes`, `get_weights_for_procrustes`, sklearn-KMeans on velocity
  directions replacing cuML, a numpy port of the cupy masked-track interpolation,
  `init_fg_from_tracks_3d` / `init_bg`, roma-free). `NativeMotionInitializer` in
  `render/gaussian4d/motion_init.py` chains PR-A's lift → cano_t → clustering/
  Procrustes → fg/bg placement into the same `(DynamicScene, {train/val:
  [BorrowedCamera]})` shape as `load_borrowed_init`. The only borrowed input left
  is the 2D point tracker output.
- `run_render_proof.py dynamic --native` builds the init natively (no SoM env, no
  cuML); `scripts/native_motion_init.py` dumps the native bundle to an `.npz` for
  parity/inspection (same layout as `borrow_som_motion_init.py`).
- Validated on paper-windmill at the full SoM schedule (iters=5000): native masked
  mPSNR **20.74** (mSSIM 0.670, mLPIPS 0.157) vs borrowed **20.68** — parity at the
  quality bar; both grow to ~955k Gaussians.

### Fixed
- Dynamic densification opacity-reset collapse. The schedule constants
  (`reset_opacity_every: 3000`, `stop_iter: 4000`) were SoM's, but the run used
  `iters: 3000`, so the opacity reset landed on the final iteration — every
  Gaussian clamped to ~1% opacity with no recovery training, leaving eval to render
  a near-black scene (masked mPSNR collapsed ~20.7 → 6.3 for *both* native and
  borrowed inits). Fixed by deriving a SoM-shaped schedule from the iteration
  budget: `densify.py::densify_schedule` sets `start/reset/stop` as fractions of
  `iters` (0.04 / 0.60 / 0.80) so `reset < stop < iters` always holds (reset always
  has recovery + a polish phase); `DensifyConfig.from_cfg(..., max_iters, num_frames)`
  derives any schedule field a config omits and errors if `stop_iter >= iters`; the
  opacity reset is now guarded to the densify window (as SoM's control loop does).
  The dynamic config's default `iters` is now 5000 (matches SoM's `max_steps`), and
  its schedule fields auto-derive to SoM's exact `200 / 100 / 3000 / 4000`.

### Tests
- `tests/test_procrustes_init.py` — `solve_procrustes` recovers a known SE(3) (+ 6D
  decode), kNN, masked interpolation, KMeans on synthetic 2-cluster velocity data,
  and `init_fg`/`init_bg` shape + log/logit/quaternion sanity.
- `tests/test_densify_schedule.py` — schedule reproduces SoM at 5000, the
  `reset < stop < iters` invariant across budgets, explicit-overrides-derived, the
  `stop_iter >= iters` guard, and backward-compatible defaults without `max_iters`.

## [0.1.11.0] - 2026-06-15

### Added
- Native 3D-track lifting (PR-A of the native-motion-init port) — room4d's own
  port of Shape-of-Motion's track lift, replacing the first borrowed step on the
  way to a native motion initializer. Only the 2D point tracks
  (`flow3d_preprocessed/2d_tracks/`) are still borrowed; the lift itself is
  room4d code. New `room4d/lifting/native_track_lift.py`: TAPIR track parsing,
  SoM scene normalization (gravity-up → +z, center, scale), foreground-track
  unprojection (`get_tracks_3d`), and background point sampling
  (`get_bkgd_points`) — all ported onto room4d's `DatasetGeometryInitializer`.
  Lifts from SoM's aligned inverse-depth source (`aligned_depth_anything_colmap`,
  with the per-stack depth clamp) and COLMAP-refined cameras
  (`camera_type="refined"`, via pycolmap; ARKit `"original"` poses optional), so
  the output lands in the same normalized frame as the borrowed bundle.
- `room4d/lifting/tracks_3d.py` — `Tracks3D` / `StaticPoints3D` containers (room4d
  analogs of SoM's `TrackObservations` / `StaticObservations`) consumed by the
  upcoming clustering + Procrustes + Gaussian-placement initializer (PR-B).
- `DatasetGeometryInitializer.load_fg_mask` — reads Track-Anything foreground
  masks (`flow3d_preprocessed/track_anything/{factor}x/`).
- `scripts/check_native_lift.py` — validates the native lift against SoM's cached
  `tracks_3d_40000.pth` + `scene_norm_dict.pth`. On paper-windmill: scene-norm
  rotation matches to 3.9e-8, scale 0.1093 vs 0.1089 (0.4%), 39 500 vs 39 525
  foreground tracks (0.1%) — a faithful reproduction of the borrowed frame.

### Tests
- `tests/test_native_track_lift.py` — dataset-free unit tests for the ported math
  (TAPIR parsing, Rodrigues/`rt_to_mat4`, `normalize_coords`, depth-normals,
  scene-norm up-alignment/centering) and the `Tracks3D`/`StaticPoints3D`
  `check_sizes`/`filter_valid` contracts.

## [0.1.10.0] - 2026-06-15

### Added
- Dynamic densification (closes the PR-4 follow-up) — standard 3DGS adaptive
  density control now runs on the dynamic foreground + background fields, the
  same recipe/thresholds already ported for the static trainer (no new
  algorithm). `DensificationController` gained `param_names` + `group_prefix`
  (so the fg field can grow its extra `motion_coefs` param and fg/bg share one
  optimizer), per-field telemetry slicing of the combined cloud, and a
  generalized clone/split (split children inherit every non-positional param,
  including `motion_coefs`, from the parent). `DynamicGaussianTrainer` drives one
  controller per field, each fed its slice of the combined-cloud `means2d`/`radii`
  telemetry. New `densify:` config section in the dynamic config.
- Effect on paper-windmill (Gaussians grow ~137k → ~970k over training):
  masked mLPIPS 0.178 → 0.164 and mSSIM 0.661 → 0.669 (perceptual/structural
  sharpness, where densification helps most), dynamic train-view PSNR ~25 → 27.2,
  and a visibly sharper flythrough. masked mPSNR is flat (~20.7) because that
  metric is dominated by the already-reconstructed static background. Still beats
  the SoM baseline on all three masked metrics.

### Tests
- `tests/test_dynamic.py`: a CPU test that the foreground controller grows
  `motion_coefs` in lockstep with the other params (and the Adam state) and that
  the optimizer still steps on the grown field.

## [0.1.9.0] - 2026-06-15

### Added
- CLI flythrough (Renderer Track 2, PR 5) — renders the trained dynamic scene
  along a hand-authored camera path while time advances and encodes
  `flythrough.mp4`. This completes the v1 renderer track (design-doc steps 8–11):
  borrowed dataset geometry → static trainer → masked-NVS benchmark vs SoM →
  dynamic SE(3) branch → free-viewpoint video. Validated: a 120-frame orbit
  flythrough of paper-windmill renders the scene (and the spinning windmill)
  coherently from novel viewpoints.
- `render/camera_paths.py::orbit_cameras` — a standalone near-trajectory orbit
  arc (center/radius/up derived from the input cameras so the path stays near
  the input trajectory, where monocular 4D is stable, rather than orbiting
  behind the scene). `replay` (the input cameras) is a config alternative.
- `render/export.py::write_video` — RGB-frames → mp4 via OpenCV (`mp4v`;
  imageio/imageio-ffmpeg aren't in the env).
- `run_render_proof.py dynamic` now emits `flythrough.mp4` after Checkpoint 3,
  driven by a `flythrough:` config section (path/num_frames/fps/arc_degrees).

### Tests
- New `tests/test_flythrough.py` (3 CPU tests): orbit-camera geometry
  (distance-from-center, target/up), single-frame edge case, and mp4 encoding.

## [0.1.8.0] - 2026-06-15

### Added
- Dynamic Gaussian branch (Renderer Track 2, PR 4) — the Shape-of-Motion SE(3)
  motion model, ported onto room4d's contracts. The renderer now handles motion;
  Checkpoint 3 passes. On paper-windmill the dynamic branch lifts masked NV
  mPSNR to **20.71** (from the static 18.51, +2.2 dB at the same factor 2) and
  beats the SoM baseline on all three masked metrics (mPSNR 20.71 vs 19.53,
  mSSIM 0.661 vs 0.559, mLPIPS 0.178 vs 0.198 — SoM scored at factor 1, a noted
  resolution caveat). Dynamic train-view PSNR is 25.4 dB (vs the static model's
  motion-capped ~22) — the moving windmill is reconstructed, not ghosted.
- Deformation model (our code, ported recipe): `render/gaussian4d/transforms.py`
  (6D-rotation + gradient-stable quaternion helpers, replacing `roma`),
  `motion.py` (`MotionBases` — K low-dim SE(3) bases over T frames),
  `dynamic_field.py` (`DynamicGaussianField` with softmax `motion_coefs` + warp,
  and `DynamicScene` that materializes the warped fg + static bg at a frame,
  plus the per-Gaussian `entity_id` channel: fg = 1, bg = 0).
- `dynamic_trainer.py` (`DynamicGaussianTrainer`): samples (camera, time, gt),
  renders the warped scene, optimizes with photometric L1 + D-SSIM + temporal
  motion-bases smoothness. `run_render_proof.py dynamic` mode +
  `configs/render/render_proof_windmill_dynamic.yaml`;
  `render/eval/novel_view.py` gains `score_masked_dynamic_views`.
- `scripts/borrow_som_motion_init.py` + `render/gaussian4d/motion_init.py` — the
  `MotionInitializer` seam. Today's only backend loads a bundle dumped by running
  SoM's track-based init (cuML clustering + Procrustes) in the reference env — a
  **stopgap**; a native room4d motion initializer is tracked follow-up work, as
  is dynamic densification (v1 trains the fixed-size borrowed init).

### Tests
- New `tests/test_dynamic.py` (7 CPU tests): 6D↔rotmat and rotmat→quat
  round-trips, quaternion product, identity/pure-translation warps, the combined
  scene + entity_id channel, and the borrowed-init bundle loader.

## [0.1.7.0] - 2026-06-15

### Added
- Co-visibility-masked novel-view-synthesis benchmark (Renderer Track 2, PR 3) —
  the DyCheck masked-metric protocol, so room4d's renderer is scored against the
  Shape-of-Motion baseline on identical sequences. First numbers (room4d static
  vs SoM dynamic on paper-windmill, all 419 val frames): masked mPSNR 18.51 vs
  19.53, mSSIM 0.448 vs 0.559, mLPIPS 0.234 vs 0.198 — a static model lands ~1 dB
  behind on mPSNR (the masked region is mostly static background) with the larger
  gaps on SSIM/LPIPS from un-reconstructable motion, the gap PR 4 closes.
- `room4d/eval/metrics/image.py` — shared, stage-agnostic masked image metrics
  `MaskedPSNR`/`MaskedSSIM`/`MaskedLPIPS`, ported verbatim from Shape-of-Motion's
  `flow3d/metrics.py` (which follows DyCheck) so the numbers are byte-comparable
  to the SoM/published baselines. Uses only torch + torchmetrics.
- `room4d/eval/benchmarks/covisibility_masked_novel_view_synthesis/` — the
  benchmark package: `schema.py` (the `reference_benchmark_v1` record contract —
  now the single source of truth), `compare.py`, and a `README.md` documenting
  the DyCheck protocol, datasets, record format, and caveats.
- `room4d/render/eval/novel_view.py` — render-stage evaluator: renders a trained
  Gaussian field at held-out cameras and accumulates the masked metrics.
- `run_render_proof.py train` now emits a `reference_benchmark_v1` record over the
  val split and can `--compare-to` a SoM baseline; `DatasetGeometryInitializer`
  gained `load_covisible(...)`. New `benchmark:` config section (both render-proof
  configs) exposes the eval split, frame cap, LPIPS net, and compare target.

### Changed
- Evaluation reorganized for scale: per-stage metrics moved from a flat
  `room4d/eval/*_metrics.py` into co-located `room4d/<stage>/eval/` packages
  (render/geometry/tracking/sync/graph), `room4d/eval/` now holds only the
  cross-cutting layer (shared `metrics/`, `benchmarks/`, and the
  `benchmark_runner` orchestrator). `git mv` preserves history; no behavior change.
- `scripts/run_reference_benchmark.py` imports the `reference_benchmark_v1` schema
  + `compare()` from the benchmark package instead of defining them inline
  (de-duplication — both emitters share one contract).

### Tests
- New `tests/test_nvs_benchmark.py` (7 CPU tests): masked PSNR (incl. that the
  mask excludes corrupted regions), masked SSIM, and the schema build/load/validate
  + compare-on-schema-mismatch behavior. All existing eval tests pass through the
  migrated import paths.

## [0.1.6.0] - 2026-06-15

### Added
- Static Gaussian splatting trainer (Renderer Track 2, PR 2). Turns the PR-1
  depth-backprojection seed into a trained static Gaussian field, scored on
  train and held-out views. The renderer now optimizes a scene rather than only
  materializing one.
- `Rasterizer` ABC extended with a differentiable training path: `rasterize()`
  returns grad-bearing rgb/alpha/depth plus densification telemetry
  (`means2d`/`radii`) as a `RasterOutput`; `render()` is now a no-grad numpy
  convenience built on it (one gsplat call site). The trainer/densification
  depend only on this contract — gsplat is never imported outside the backend.
- `GaussianField(nn.Module)` (`render/gaussian4d/gaussian_field.py`): the
  trainable scene in the standard 3DGS parameterization (log-scales,
  logit-opacities, sigmoid RGB color; SH is a future lever), seeded via
  `from_cloud`.
- `DensificationController` (`render/gaussian4d/densify.py`): our own adaptive
  density control — clone / split / prune / opacity-reset with Adam-state
  surgery — operating on the rasterizer's telemetry (so it is backend-agnostic),
  with thresholds ported from gsplat's validated `DefaultStrategy` and a
  `max_gaussians` safety cap. The NDC `means2d` gradient is rescaled to pixel
  units so the ported `grad2d_threshold` applies in the right space.
- `StaticGaussianTrainer` (`render/gaussian4d/gaussian_trainer.py`): Adam with
  per-group learning rates (means LR exponentially decayed), an L1 + D-SSIM
  photometric loss (self-contained torch SSIM, no new dependency), and the
  densification schedule. Plain PSNR/SSIM eval (masked-vs-SoM scoring is PR 3).
- `run_render_proof.py train` mode (Checkpoints 1 & 2) plus
  `configs/render/render_proof.yaml` and `configs/render/render_proof_windmill.yaml`
  exposing every trainer/densify/seed knob; the CLI overrides only per-run
  paths. Multi-frame seeding (`seed_times`) supports moving monocular cameras.
- Validated: trains end-to-end on the 5090. On DyCheck paper-windmill (a moving
  monocular camera — the proper static-training regime; Balloon1's train camera
  is a single static rig view) the static structure reconstructs sharply on
  both the train view and a held-out camera that contributed no gradient.
  Train/novel PSNR plateaus at ~22/~20 dB because a static model cannot fit the
  scene's motion — the dynamic branch (PR 4) is what lifts it.

### Changed
- `RasterOutput` carries render width/height so the controller can rescale the
  NDC means2d gradient.

### Tests
- New `tests/test_gaussian_trainer.py` (7 CPU tests): field activations
  round-trip; SSIM/PSNR sanity; and the densification Adam-state surgery
  end-to-end — clone grows the field+optimizer, split consumes parents,
  low-opacity prune, and the `max_gaussians` cap — each asserting the optimizer
  still steps on the resized parameters.

## [0.1.5.0] - 2026-06-15

### Added
- Renderer Track 2 step 8 + Checkpoint 0 (coordinate sanity). Stands up
  borrowed dataset geometry flowing through room4d's real contracts into a
  pluggable rasterizer, validated by rendering at real cameras.
- `Rasterizer` abstract base class with `GsplatRasterizer` as the first
  pluggable backend, plus a `build_rasterizer(cfg)` factory
  (`room4d/render/gaussian4d/rasterizer.py`). gsplat is no longer welded into
  the renderer — it sits behind an enforced contract, mirroring the
  `GeometryInitializer`/`PointTracker` backend pattern, so the splatting kernel
  is swappable. A new `RenderCamera` (with `from_look_at` and `from_world_cam`
  constructors) is the backend-agnostic camera the rasterizer renders from,
  carrying a real world-to-camera matrix + intrinsics (previously the
  rasterizer synthesized intrinsics and only accepted procedural look-at
  cameras).
- `DatasetGeometryInitializer` (`room4d/geometry/dataset_wrapper.py`): reads
  pre-computed DyCheck-format camera poses/intrinsics/depth from disk and emits
  a standard `GeometryArtifact`, bypassing on-the-fly SfM/VGGT. One backend
  handles both NVIDIA Dynamic Scenes (Balloon1) and DyCheck iPhone
  (paper-windmill). Camera convention matches the Shape-of-Motion loader
  (`w2c = [R | -R p]`; `T_world_cam = inv(w2c)`). Selectable via
  `--geometry-backend-name dataset`.
- `seed_cloud_from_geometry` (`room4d/render/gaussian4d/seed.py`): backprojects
  per-frame metric depth + RGB into a single colored `GaussianCloud` in the
  world frame — the untrained seed for the renderer.
- `scripts/run_render_proof.py`: the render-proof harness. Checkpoint 0 loads a
  sequence through the dataset backend, seeds a point-cloud splat from one
  view's depth, renders at real dataset cameras, and writes GT-vs-render
  side-by-side panels. On Balloon1 (seed cam 0): the same-view reconstruction
  and all three held-out cameras render recognizably and correctly placed —
  coordinate conventions validated before any training time.

### Changed
- `Gaussian4DTrainer` now constructs its rasterizer via `build_rasterizer` and
  renders through `RenderCamera`; flythrough behavior is unchanged.

### Tests
- New `tests/test_dataset_geometry.py`: dataset pose-convention (camera center
  == position; `T_world_cam = inv(w2c)`), factor-scaled intrinsics,
  `GeometryArtifact` construction, and the seed backprojection math (identity +
  translated pose), all CPU-only.
- `tests/test_gsplat_rasterizer.py`: migrated to the `RenderCamera` contract;
  added a `from_world_cam` coordinate-convention guard (known 3D point →
  principal point) and a `build_rasterizer` factory test.

## [0.1.4.0] - 2026-06-12

### Added
- Calibration GT scoring (calibration arc M0-minimal; design decisions
  D2–D12): `scripts/score_calibration_gt.py` collects dataset frames with
  their ground-truth poses, runs the existing hloc+COLMAP SfM, and scores
  the recovered extrinsics. PRIMARY metrics are gauge-free pairwise
  relative errors (per-pair rotation geodesic, dual-anchor baseline
  direction, normalized baseline ratio) — same-position pairs are reported
  separately as registration repeatability so they cannot flatter the
  headline numbers; SECONDARY is Sim(3)-APE on camera centers with
  conditioning guards (GT and estimate sides). Fragmentation is a result,
  not an error: the largest alignable component is scored with coverage
  reported. First real numbers: Balloon1 cross-camera pairwise rotation
  error mean 0.176° (free intrinsics) / 0.136° (known-intrinsics
  protocol); paper-windmill 1.89° across 3 cameras against ARKit odometry.
- Known-intrinsics protocol: `--fix-intrinsics` injects shared GT
  intrinsics into COLMAP (`camera_params` passthrough in the SfM runner)
  and disables refinement; refuses per-camera intrinsics (hloc applies
  image options globally — DyCheck's per-frame ARKit focal drift correctly
  triggers the refusal). The score artifact records recovered intrinsics,
  the effective camera mode, and verifies the injection actually held
  (`intrinsics_match_injected`) so a silent failure can't masquerade as a
  fixed-intrinsics result.
- GT manifest contract: the frame collector writes `gt_manifest.json`
  (image filename → GT pose) at collection time, next to — never inside —
  the image dir; the scorer joins blindly, so correspondence can't drift.
  Re-collections purge stale images (hloc sweeps every file in the dir).
- Pose math is oracle-proven: hand-rolled Umeyama Sim(3) + geodesic errors
  pinned against evo v1.36.5 fixtures, with 0°/180° clamps, reflection
  handling, and small-N degradation guards.
- hloc compatibility patches for pycolmap ≥ 3.10 in `setup_room4d.sh`
  (import_images options object + kwarg rename, verify_matches RANSAC
  options, incremental_mapping pipeline options; idempotent with loud
  anchor-miss failure), plus vendored-tree restore before ref checkout and
  a setup smoke test of the exact patched option fields. Collected frames
  are copied, never symlinked (pycolmap ≥ 3.13 resolves symlinks during
  import, desyncing database names from hloc feature keys).

### Tests
- 33 new unit tests in `tests/test_gt_scoring.py`: evo-oracle alignment
  fixtures, gauge-invariance and per-camera error localization, degenerate
  and estimate-collapse flagging, adapter schema guards against real
  Balloon1 data, manifest round-trip + purge, CLI end-to-end through a
  mock SfM model in both intrinsics modes, and the exact injection
  payload.

## [0.1.3.1] - 2026-06-12

### Fixed
- Resuming an interrupted reference-benchmark run now recalls the schedule
  recorded in the run's `cfg.yaml` and passes it back to Shape of Motion
  explicitly — SoM itself never re-reads it, so a plain resume previously
  retargeted any interrupted run toward the default 500-epoch schedule
  (a 5-epoch smoke run would silently become a 500-epoch run). A run whose
  `cfg.yaml` is missing now refuses to resume rather than guessing.
- A resume whose checkpoint is already at or past its recorded schedule
  aborts up front (the checkpoint's epoch is read via the reference env)
  instead of letting SoM train zero epochs, exit 0, and get stamped as a
  completed baseline carrying mid-training metrics.
- Benchmark invocations now hold an exclusive kernel lock (`fcntl.flock`)
  on their run dir for the duration of training + eval, so a forgotten
  concurrent run is refused instead of interleaving checkpoints and
  TensorBoard event files into plausible-looking corrupt metrics. The lock
  releases on any process death (`kill -9` included) and is inherited by
  the training child, so an orphaned trainer keeps the run dir protected
  until it actually exits.
- An explicit `--num-epochs 500` and a plain run now share one schedule
  label (`default`), so alternating between them can no longer trigger a
  spurious schedule-mismatch abort against the same run dir.
- The completion sentinel is written atomically (a crash mid-write can no
  longer leave an empty sentinel that reads as a completed default-schedule
  run), and resumed runs record `resumed`/`resumed_from_epoch` in the
  result JSON so a segment-only `train_minutes` can't masquerade as the
  full training cost.
- The checkpoint-epoch reader uses `weights_only=True` only (SoM
  checkpoints are plain tensor dicts; the pickle fallback was unnecessary
  code-execution surface) and times out rather than wedging the pipeline
  on a hung `conda run`.

### Tests
- 9 new unit tests (plus extended scenarios in 6 existing ones): schedule
  recall (explicit and default), zero-epoch abort boundary (`>=` equality),
  fail-closed missing schedule, flock exclusivity/release-on-death/
  contention-abort behavior, label canonicalization, resume provenance
  fields, fresh-run-dir passthrough, and checkpoint-reader degradation
  paths.

## [0.1.3.0] - 2026-06-11

### Added
- NVIDIA Dynamic Scenes dataset backend for the reference benchmark:
  `scripts/run_reference_benchmark.py --dataset {iphone,nvidia}` selects the
  layout validation, the Shape of Motion `data:nvidia` training subcommand,
  and the eval path. SoM ships no `evaluate_nvidia.py`, so nvidia metrics are
  harvested from the trainer's final-epoch `val/*` TensorBoard scalars
  (covisibility-masked novel-view metrics, mapped onto the same
  `masked_mpsnr`/`masked_mssim`/`masked_mlpips` schema keys, plus
  `fg_`/`bg_` splits). Proven end-to-end on real GPU with a 5-epoch smoke
  run on the Jumping scene (masked mPSNR 18.69, all nine metrics emitted).
- Deep download verification for nvidia sequences in both validators
  (`verify_reference_data.sh` and the wrapper): full layout
  (`rgb/{1x,2x}`, `camera/`, `depth/2x`, masks, tracks, `covisible/2x/val`),
  a non-empty val split (an empty one would train for hours and yield no
  metrics), and the complete N² pairwise track set (the dataloader loads
  random pairs per batch, so a partial download would crash mid-training).
- Result JSONs now record the evaluation protocol (`eval` block: source and
  resolution factor — nvidia trains/evaluates at the dataloader's hardcoded
  2x, iphone at 1x), and `--compare` warns when two results' protocols
  differ instead of silently declaring winners on incomparable numbers.

### Fixed
- Reusing a completed run dir with the wrong `--data-dir` now aborts instead
  of silently relabeling the old run's metrics as the new sequence (the run
  dir's `cfg.yaml` records what it was trained on).
- `--retrain` also clears TensorBoard event files, the stale `cfg.yaml`, and
  SoM's per-invocation `code/` backups — a stale event file would otherwise
  outrank the new run's scalars in the nvidia harvest, since a retrain
  restarts `global_step` at 0.
- The nvidia harvest rejects mixed-run event files (val scalars disagreeing
  on their final step) and skips non-finite scalar values rather than
  recording NaN/Inf as a baseline metric.
- Corrected the interrupted-run guard's false claim that SoM "continues its
  saved schedule" on resume — SoM never re-reads `cfg.yaml`; resume trains
  toward the CLI/default 500-epoch schedule (proper resume handling is
  tracked in TODOS.md).
- `--num-epochs` values below 1 are rejected up front (SoM would accept
  them, pay the full dataset-load cost, train zero epochs, and exit 0).

### Tests
- 24 new unit tests: nvidia layout/content validation, TensorBoard harvest
  mapping and error paths, provenance and mixed-run guards, compare-side
  protocol warnings and fg/bg metric directions, and mutation-style
  bash↔python validator parity (each required entry deleted in turn must
  fail both validators).

## [0.1.2.0] - 2026-06-10

### Added
- Native-gsplat render module (design-doc Track 2, step 7). `GaussianRasterizer`
  is now a real gsplat wrapper (eye/target/up → OpenCV viewmat, legacy framing
  intrinsics preserved so existing camera paths look the same), `Gaussian4DScene`
  materializes the scene contract as device-resident `GaussianCloud` tensors,
  and `Gaussian4DTrainer` orchestrates real GPU renders of the static, dynamic,
  and full Gaussian layers. The debug viewers (triptych frames, layer dumps,
  both HTML players) and the `RenderArtifact` contract are preserved unchanged.
- `tests/test_gsplat_rasterizer.py`: viewmat/projection/intrinsics sanity on
  CPU plus GPU rasterization tests — centered-blob coordinate sanity
  (checkpoint 0 of the renderer ladder), depth values, empty-cloud handling,
  cloud concat, and depth-ordered occlusion.

### Changed
- Removed every fake render path: the per-Gaussian NumPy splatting loop, the
  hand-rolled projection, the sprite-billboard compositor and observation-bank
  selection, the per-source static re-splatting, and the blur/sharpen
  "NeRF refinement". There is no CPU fallback — rendering hard-requires gsplat
  and CUDA with an error pointing at `scripts/setup_room4d.sh`. Dynamic
  entities render as their template Gaussians until trained Gaussians and
  motion bases land (steps 9–10).
- `OfflineHighQualityTrainer` and the `quality_mode` split are gone; the
  `offline_high_quality` and `gaussian_layered` backend names remain as
  registry aliases for the one gsplat renderer. The only surviving quality
  knob is `render_scale` supersampling. Render configs (`configs/base.yaml`,
  `configs/render/render.yaml`, single-video runner presets) drop the fake
  knobs (`offline_high_quality` block, sprite/`nerf_refinement` settings,
  `dynamic_point_budget`, `use_gpu_projection`, render `cpu_threads`).
- `scripts/setup_room4d.sh` builds gsplat by default (`--no-gsplat` to skip)
  now that the renderer requires it.

## [0.1.1.1] - 2026-06-10

### Added
- `--with-gsplat` option in `scripts/setup_room4d.sh` (Track 2 milestone-0):
  builds gsplat from source for sm_120 in the main `room4d` env, pinned to the
  same Blackwell-proven commit as the reference env, with SHA-verified
  idempotency (the rebuild is skipped only when the installed VCS commit
  matches the pin) and a post-install smoke test that rasterizes a trivial
  scene on the GPU. First verified build of that pin against CUDA 13.0
  (torch 2.11 +cu130); the reference env had only proven it on CUDA 12.8.
- Cleanup of gsplat's leaked top-level packages after install: the pinned
  wheel ships `tests` and `experimental` into site-packages (upstream
  packaging bug), and the leaked `tests` shadows the repo's own tests package
  and breaks pytest collection. The installer now removes every top-level
  package gsplat's RECORD claims outside its own namespace.
- `jaxtyping` and `ninja` are installed alongside gsplat: they are its only
  declared deps not already in the locked tree (gsplat itself is installed
  `--no-deps` so its torch/numpy declarations cannot fight the lock), and
  without them the env fails `pip check`.

### Changed
- CUDA-toolkit and compiler helper messages in `scripts/setup_room4d.sh` no
  longer claim to be VGGT-Long-specific; the helpers now serve any CUDA
  extension build (VGGT-Long, gsplat).

## [0.1.1.0] - 2026-06-10

### Added
- Reference comparison tool: benchmark room4d against Shape of Motion on the
  public DyCheck iPhone dataset. `scripts/setup_reference_tool.sh` builds an
  isolated, Blackwell-capable (sm_120) conda env — pinned Shape of Motion
  checkout with reproducible, sentinel-verified compatibility patches, gsplat
  built from source at a pinned commit, RAPIDS/cuML on a CUDA-12 release —
  with safety guards that refuse to touch the main `room4d` env.
- `scripts/run_reference_benchmark.py`: one command turns a processed dataset
  sequence into a normalized `reference_benchmark_v1` JSON (masked
  mPSNR/mSSIM/mLPIPS plus 2D/3D tracking metrics, environment versions, video
  paths). Completed runs are reused, interrupted runs resume, `--retrain`
  starts clean; `--compare a.json b.json` prints a side-by-side scoring table
  with per-metric better-direction.
- `scripts/verify_reference_data.sh`: validates a downloaded dataset sequence
  against the layout Shape of Motion actually reads (training and eval
  inputs), before any GPU time is spent.
- Full pip-layer lockfiles for both project conda envs (`scripts/locks/`,
  generated by `scripts/gen_env_locks.sh`); both setup scripts install from
  their lock by default so env rebuilds replay the known-good dependency tree
  (`--unlocked` to re-resolve).
- First reference baseline recorded on DyCheck `paper-windmill` (RTX 5090,
  500 epochs, ~69 min): masked mPSNR 19.53 / mSSIM 0.559 / mLPIPS 0.198 —
  consistent with the published Shape of Motion results.
- Unit tests for the benchmark tool (104 tests total): metric parsing,
  compare-mode edge cases (NaN, unknown metrics), run-completion semantics,
  validator parity between the bash and Python checkers, and installer safety
  guards.
- Release-tag automation: `tag-on-version-bump.yml` (annotated `v<version>`
  tag on every develop version bump) and `tag-on-release.yml`
  (`release-v<version>` tag + GitHub Release on develop→main merges). The
  canonical version now lives in a `VERSION` file, with `pyproject.toml`
  reading it at build time via setuptools dynamic metadata — one source of
  truth, standard workflows unmodified.

### Fixed
- Benchmark runs interrupted mid-training are no longer mistaken for
  completed baselines (completion is proven by a sentinel or final-epoch
  outputs), and `--retrain` now genuinely retrains instead of silently
  resuming a finished run.

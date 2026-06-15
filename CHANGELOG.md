# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(loose variant), and this project uses a 4-digit `MAJOR.MINOR.PATCH.MICRO`
version scheme (canonical version in the `VERSION` file, read by
`pyproject.toml` at build time).

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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(loose variant), and this project uses a 4-digit `MAJOR.MINOR.PATCH.MICRO`
version scheme (canonical version in the `VERSION` file, read by
`pyproject.toml` at build time).

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

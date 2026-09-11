# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(loose variant), and this project uses a 4-digit `MAJOR.MINOR.PATCH.MICRO`
version scheme (canonical version in the `VERSION` file, read by
`pyproject.toml` at build time).

## [0.2.1.0] - 2026-09-11

### Added
- **A resume loads only what this run would make, and says why** (`reifyscene/api/contracts.py`,
  `resume.py`, `requirements.py`). `--start-stage X --prior-run-dir P` used to compare each
  loaded stage's own config section and the input's absolute path, after copying it into the
  new run. An audit of all 15 stages found it wrong both ways: segment's settings live in
  `objects`, the solver reads `optimizer` while `optimization` was compared, tracking's lift
  reads `lifting` — so changed settings resumed silently — while the absolute path made every
  run from before the project rename unresumable and the tracking profile, picked from free GPU
  memory at launch, rejected a resume started while the GPU was busy. Now:
  - **Every stage declares what its output depends on**: the settings it reads, from any
    section, with defaults filled and debug / device keys left out; folders it names by what is
    in them, not where they are; a contract version (bumped, with a note, when its output
    changes meaning); and the stages it consumes. Each artifact records that fingerprint, the
    identity of the frames the decoder read (name, size, mtime — not the folder's path) and the
    digests of the upstream artifacts it was built from. `tests/test_stage_settings_coverage.py`
    runs the pipeline on a config that logs every key read and fails on one a stage does not
    fingerprint.
  - **The resume is checked read-only before anything is copied or run.** One report lists every
    problem and the latest stage the prior run supports; `--check-resume` prints it and exits.
    Resuming into the prior's own folder is refused. Settings picked from the machine are pinned
    to the prior run's values.
  - **Each stage lists what a loaded artifact must carry**, and what can be derived from what is
    loaded is: a moved input or dataset folder, photos from the run's own copies or the dataset,
    a crop box, re-lifted tracks. Cheap stages that run no model (sync, objects, graph,
    optimization) are rebuilt; anything else rejects the resume and names the stage to restart
    from. Loading copies a stage exactly and patches only its `artifact.json`.

  Runs from before these records get them derived from their run manifests. All three spin runs
  kept on disk — `spin_native_full`, and `spin_oracle_allframes` / `spin_vggtlong_capstone` from
  the pre-rename `project` paths — resume at `scene`; the static oracle below is one.
- **`scripts/export_checkpoint_views.py`: see a trained scene, not just its score.** The
  masked-NVS benchmark rendered every held-out view and kept only the average, and the
  pipeline's one saved render, the orbit flythrough, swings far off the captured path and
  smears. The exporter renders straight from a run's `render/trained_dynamic.npz` (nothing is
  retrained, so old runs work too) and writes:
  - **held-out views**, evenly spaced over the benchmark's held-out split: the render, the real
    photo, the covisibility mask, and that view's masked PSNR (`heldout/index.json`);
  - **a replay video**: the scene re-rendered from every training camera at its own time, i.e.
    the captured path. It shows reconstruction fidelity, not novel views;
  - **`score.json`**: the full held-out split's masked mPSNR / mSSIM / mLPIPS with how the
    cameras were placed, so an old run can be re-scored without touching its records.

  `--align auto` (default) reads the run's geometry backend from its manifest, as the pipeline
  does. On the spin oracle run: 8 held-out views at 17.3–21.1 dB (its full-split average is
  19.64) and a 426-frame replay, in 6 s on the GPU. The logic is in
  `reifyscene/render/eval/checkpoint_views.py`.
- **`configs/experiment/spin_native_full.yaml`**, the config for the headline native run
  (VGGT-Long + pose refinement on every frame of spin, scored on all 722 held-out views: masked
  mPSNR **14.96** with this release's photo fix on geometry whose bundle adjustment converged;
  12.37 before the fix — but see Findings: pose refinement can diverge). No config for it was
  in the repo before; its eval block lived only in one
  run's recorded config. `--config` replaces `base.yaml` rather than merging onto it, so the file
  is `base.yaml` plus `scene.kind: dynamic` and `evaluation.masked_nvs`.

### Fixed
- **VGGT's photos did not match its cameras.** VGGT and VGGT-Long see a resized,
  centre-cropped square of each frame (720×960 → 518×686 → rows 84–602), so their cameras, K
  and depth maps describe that square, but six places paired them with the whole frame: static
  training photos at full size with the square's K; seed colours read from the photo's top-left
  corner; dynamic training photos, and the dynamic init's frames and foreground masks, squashed
  to 518×518; and track positions squashed into the square's depth, those outside it clamped to
  its edge. `reifyscene/geometry/image_transform.py` records the exact ops per stream
  (`metadata.image_transforms`, written by every geometry backend; pixel-identical to both VGGT
  loaders), and every one of those places goes through it; nothing is squashed any more, and a
  photo that does not fit raises. Tracks outside the crop get no 3D point. Tracking and the
  optimizer's re-lift share one lift (`lifting/lift.py`: the geometry's depth and pose-refined
  camera, through the crop), so a track no longer has two positions — tracking used to read
  VGGT's point maps, which pose refinement does not refine. Spin, 722 held-out views, same
  geometry before and after: native dynamic **12.37 → 14.96** (mLPIPS 0.46), the gap to the GT
  oracle down from 7.3 dB to 4.7. The oracle itself barely moves (19.64 → 19.65 retrained: its
  frames already match its depth; only its foreground masks are now resized nearest, not bicubic).
- **Static scenes were scored with misplaced cameras.** The static scorer placed the dataset's
  held-out cameras with no transform, but a static scene lives in its geometry's
  ground-plane-canonicalized frame on every backend, the GT oracle's included, and its checkpoint
  has no scene-norm to undo that — the native static run's 11.46 was a render of the wrong
  place. Static scoring now uses the dynamic path's similarity fit (`align: sim3`, fitted on the
  scene's training views, same refusal rules), the default for every static scene. On the static
  oracle the fit is exact — scale 0.99999998, median rotation residual 0.0°, centre residual
  5·10⁻⁸ — and it scores **16.22**; the native static scene scores **14.56**.
  `export_checkpoint_views.py` exports and scores static runs too (no replay).
- **Resume bugs the audit found**:
  - `--run-dir` equal to `--prior-run-dir` deleted each prior stage before copying it;
  - the check ran after the copy, so a rejected resume had already copied earlier stages and
    deleted whatever was at the destination;
  - every resume dropped the object embeddings (filed under the track id, reloaded by the
    detection id);
  - artifact loaders skipped a listed mask or embedding file that was missing, and took their
    folder from the saved JSON rather than the folder they were loaded from;
  - ingest recorded its input by the path as typed, so a relative path resolved only from the
    working directory it was run in.
- **The VGGT input cache was shared by every run of a sequence.** It was keyed by the input
  folder's name (`vggt_long_inputs/1x`) and emptied and rewritten by each run, so an older run's
  photo paths could point at another run's images. It is now keyed by what decides its pixels
  (the frames' name, size and mtime, and the downscale), never wiped, and its paths are absolute.
- **A backend asked for as `real` could still fall back silently.** The wrappers raise when a
  `real` backend is unavailable, but each stage's backend ladder caught that and ran the next
  rung — for geometry the empty floor, so `--geometry-backend-mode real` (the run script's
  default) could hand every later stage an empty geometry, stamped and saved. Ladder rungs can
  now be strict: the geometry rung when `geometry.backend_mode` is `real` (also when its package
  is missing), segment's rungs when the object detector is, reid's ResNet floor when the embedder
  is, humans' Keypoint-RCNN rung when the pose model is. A strict rung that fails stops the run
  (`BackendRequired`) instead of falling back. Under `auto`, falling back is still allowed, but
  never quietly: every run ends with a table of each stage that ran (or was loaded) on a weaker
  backend than its best, and records them in the run manifest (`fallbacks`).
- **Tracking stopped at frame 95.** The machine profile (`--resource-profile auto`) capped the
  tracked frames at 24 / 48 / 96 by free GPU memory, so spin's 426-frame clip was tracked, lifted
  and put into the graph only up to frame 95 (3.2 s of 14.2). Every profile tracks in windows,
  which is what bounds memory, so none caps the frame count now; `--tracking-max-frames` still
  does. A value the user did not give counts as machine-picked for a resume even when the profile
  picks nothing, so a run tracked under the old cap still resumes with it.
- **Pose refinement was not repeatable, and a diverged bundle adjustment was used anyway.** Two
  runs of `spin_native_full.yaml` got identical VGGT-Long geometry and pixel-identical tracker
  images, yet one bundle adjustment converged (final reprojection 0.88 px) and the other diverged
  (6·10¹⁴⁹ px, one camera turned 160°) — and its poses went downstream: that run scored 12.67
  against 14.96 on the converged geometry. The tracker shuffles its query points with an
  unseeded `torch.randperm`; pose refinement now seeds it (`pose_refine.seed`, default 0). A
  bundle adjustment that ends above `pose_refine.max_final_reproj_px` (8) or turns a camera more
  than `max_rot_delta_deg` (90°) from its seed pose is refused: the stream keeps its seed poses
  and `metadata.pose_refine.rejected_streams` says why. A resume rejects a pose_refine artifact
  that kept a diverged stream (restart at pose_refine) and loads one that refused it. Seeding
  does not bump pose_refine's contract version: an unseeded refinement that converged is as good
  as a seeded one.
- **Our no-upscale VGGT-Long loader squashed portrait frames.** Below a 518 target it scaled by
  a floating-point ratio (720 at 252 floored to 238) and clamped *both* sides to the target, so a
  720×960 frame went in at 238×252 instead of being cropped like upstream. It now takes the width
  to the target in whole 14-pixel patches, keeps the height in proportion and crops the rows
  (252×336 → rows 42–294). Geometry records the loader as version 2. A resume refuses geometry
  the old loader made from frames it squashed (restart at geometry); for frames it did not
  squash, the old and new loaders agree and the geometry loads. None of the runs above use it.
- **VGGT-Long's scratch outputs were shared by every run of a sequence and never removed.** Its
  chunk predictions and `camera_poses.txt` went to `.reifyscene_cache/vggt_long_run/<input
  folder name>` (about 11 GB for spin's 426 frames), where two runs of one input would write
  the same files. Each geometry invocation now gets its own folder, removed once the geometry is
  built — nothing reads it after; `geometry.keep_backend_outputs: true` keeps it, and an
  explicit `output_cache_dir` is always kept. The geometry artifact records where it was
  (`metadata.vggt_long_outputs`). Neither setting is part of the geometry fingerprint.
- **Pose refinement wrote its COLMAP seed model into the geometry stage's folder**
  (`geometry/colmap_seed/`), so a geometry artifact collected another stage's files. It now
  writes into its own `pose_refine/` folder.
- **The optimization artifact called its stage `optimize`**; it now uses the stage registry's
  name, `optimization`, like every other stage.
- **Self-reconstructed scenes could not be scored: every native run's held-out renders were
  black.** The dynamic scorer placed the dataset's held-out cameras in the scene through the
  bundle's scene-norm, which is only right when the scene was built on the dataset's own
  geometry (the oracle). A VGGT / VGGT-Long reconstruction lives in its own frame (for today's
  run: scale 1.21 and 43° of rotation from the dataset's world), so the held-out cameras looked
  at empty space and every native run on disk scored 4.5–4.6 whatever its quality. The scorer
  now places them through a similarity fitted on the training cameras, whose poses are known in
  both frames (`align: sim3`, the pipeline's default for any non-`dataset` geometry backend) —
  the usual way to score a method that estimates its own cameras. The fit absorbs one global
  offset; per-camera drift still counts. Fits whose scale is not positive, or whose median
  rotation / centre residual exceeds 5° / 0.2 of the trajectory, are refused: the score
  records why and gives no number. Every score now carries an `alignment` record (method,
  scale, residuals). The oracle keeps its placement (`align: none`) and scores bit-identically
  (19.638906478881836).
- **Sparse-keyframe scenes rendered held-out views at the wrong time.** Scene time counts
  keyframes, but views were rendered at the dataset frame number, which the scene clamps to its
  last keyframe — every held-out view past it showed the scene's final state. Each view now
  renders at the nearest training keyframe's time. Full-frame scenes are unaffected.

### Correction
- **The native numbers in 0.1.31.3 (10.29 / 11.95) and 0.2.0.0 (12.56) were never scored by
  the repo.** Re-scoring the surviving checkpoints with the scorer of the day (0.1.31.3) and of
  today both gives ~4.6. They came from an aligned scoring step run outside the repo: its pose
  errors (0.88° / 0.615° / 2.3° / 0.71°) are exactly the median rotation residuals of the
  training-camera fit, and the aligned scorer now reproduces its scores within 0.03–0.36 dB
  (10.06 / 11.92 / 9.19 / 12.20). The headline native number is replaced by the reproducible
  **12.37** from a fresh full run of `spin_native_full.yaml` (0.60° residual), and with the
  photo fix above by **14.96** on that same run's geometry.

### Docs
- **`base.yaml` builds a static scene, and the README now says so.** The README's run command
  used `configs/base.yaml` and described the output as `render/trained_dynamic.npz`, but
  `base.yaml` has no `scene.kind`, so the command builds a *static* scene
  (`render/trained_gaussians.npz`); the dynamic 4D scene needs `scene: {kind: dynamic}`. Found
  when the first run of the new experiment config (which lacked it too) built a static scene.

### Changed
- **Held-out camera placement is shared, not duplicated.** `place_heldout_views` (with
  `load_heldout_view`, `eval_dataset` and `score_dynamic_bundle`) in `masked_nvs_benchmark.py`
  now does the placement that `score_render_artifact_masked_nvs_dynamic` did inline, and the
  exporter calls the same function, so an exported view is exactly a scored view. That placement
  has had real bugs before (0.1.31.2's missing ground-plane rotation; the missing alignment
  above); a second copy would be a second place for one to hide. The refactor alone changed no
  result: scoring the oracle checkpoint on all 722 views with the old and new scorer in one
  process gave identical mPSNR / mSSIM / mLPIPS.
- **The Sim(3) helpers moved to `reifyscene/geometry/similarity.py`** (`fit_similarity`,
  `apply_similarity_to_c2w`, `camera_center`, plus `similarity_residuals`), shared by
  `pose_refine`'s gauge re-anchor and the scorer's alignment; `pose_refine` keeps its old names.
- **`scene/train_views/*.png` are the photos in the geometry's pixels** (518×518 for VGGT), not
  copies of the input frames, so a scene's photos match its cameras by construction; the render
  stage's `_load_gt` refuses a photo that is not its camera's size instead of resizing it.
  `pose_refine` builds its tracker images from the recorded crop instead of re-running VGGT's
  loader. A frame the geometry lists without a photo on disk raises instead of being dropped.

### Removed
- **Evaluation code nothing ran, which would have carried the crop bug.** `BenchmarkRunner`
  (`reifyscene/eval/benchmark_runner.py`) was built by the pipeline but never run, and it alone
  used `RenderingMetrics` (train-view metrics that squashed ground truth onto the prediction and
  called a renderer method that does not exist), `TrackingMetrics` (projected with the crop's K
  onto full-frame track positions; its quality score read summaries nothing writes, so it was
  always 0.0) and `GeometryMetrics`. Also removed: the pipeline's human-centre projection
  (`_attach_human_debug_projections`, `_project_world_to_image`, called by nothing, and relying on
  an assembler method that no longer exists) and the human debug view's branch that drew those
  projections. The real scoring is `masked_nvs_benchmark.py`.

### Findings
- **On spin, the dynamic scene barely beats a static one on held-out views.** On the same
  native geometry the dynamic scene scores 14.96 masked mPSNR against the static scene's 14.56 —
  and the static one has the better mLPIPS (0.43 against 0.46) — though the dynamic scene fits
  its training views 4.3 dB better (21.3 against 17.0). Its motion modelling generalizes little
  of what it fits: a lead for the renderer, not changed here.
### Tests
- `tests/test_checkpoint_views.py`: placement from bundle-carried cameras (K scaled by the
  factor ratio; truncation) and from the dataset through the scene-norm; unplaceable bundles
  (the scorer returns `{}`, the exporter refuses); the default alignment per geometry backend;
  a known similarity recovered from synthetic training cameras and the held-out cameras placed
  through it; sparse keyframes rendered at the nearest keyframe; an unrelated camera set refused
  with no score; even view picking; the replay's image size from the principal point; the
  held-out export's files, index and per-view masked PSNR (checked against a closed form);
  replay order and size. CPU-only stand-ins for the rasterizer, scene and dataset reader.
- Static scoring: a known similarity recovered through the scene's training views, `align: none`,
  refusal, the render stage's static branch, the static export and the CLI's run-kind detection.
- `tests/test_image_transform.py`, `tests/test_crop_box_sites.py`: both VGGT loaders
  reproduced pixel for pixel; point mapping in and out of the crop (a crop-free transform maps
  exactly as before); size dispatch; each backend's record; the six sites on synthetic photos
  with distinct top and bottom bands; the shared lift recovering known points.
- `tests/test_resume_pipeline.py`, `tests/test_resume_requirements.py`,
  `tests/test_run_script_resume.py`: a real prior run from the CPU pipeline resumed, rejected for
  a changed setting in another section, pinned for a machine-picked one, re-rooted for a moved
  input, rebuilt or rejected for a changed upstream; runs converted to the pre-fingerprint format
  resumed through derived records (the digests equal a fresh run's), re-rooted by name for a gone
  dataset folder, rejected for a leftover stage; every requirement's derivation and rejection;
  the run script end to end (load, rebuild, reject before writing anything, `--check-resume`).
- `tests/test_stage_settings_coverage.py`: every config key a stage reads is in its fingerprint.
  Full suite: **483 passed, 1 skipped**.

## [0.2.0.2] - 2026-09-10

### Fixed
- **The lockfile did not lock the environment.** `TRANSFORMERS_SPEC` is an open-ended
  lower bound (`transformers>=4.44`) and transformers is not in the lockfile, so those
  installs run *after* the pinned tree and escape it — a fresh machine resolves whatever
  is newest that day. Today that is transformers 5.17.0, which requires
  `safetensors>=0.8.0` while the lock pins `safetensors==0.7.0`; transformers rejects
  that combination at import, breaking test collection outright. The long-lived env was
  unaffected only because it had been sitting on 5.12.1 since June. Now installed via a
  new `pip_install_constrained` that passes the lockfile as a pip constraints file, so
  the resolver honours the pins (resolves to 5.12.1 — the version that works).
- **`cmd | grep -q` is unsafe under `set -o pipefail`**: `grep -q` exits at the first
  match and closes the pipe, the producer dies of SIGPIPE (141), and pipefail reports the
  pipeline as failed *even though the match succeeded*. Two permanent (not flaky)
  consequences: the ffmpeg install guard never fired, so conda reinstalled ffmpeg on
  every run; and the ffmpeg smoke test could never pass, so the script always exited 1 at
  the final step after all real work had succeeded. Fixed by capturing output then
  matching, here and in `gen_env_locks.sh` / `setup_reference_tool.sh` (where the same
  pattern is racy rather than deterministic).
- **`setup_reifyscene.sh` could only ever be run once per environment.** conda's
  `activate.d` hooks are not `set -u` safe — the cuda-nvcc hook expands
  `${NVCC_PREPEND_FLAGS}` unconditionally — so under `set -euo pipefail` activation
  aborted with "unbound variable", but only once cuda-nvcc was installed. The first run
  activates a bare env and installs the hook; every later run died. `conda activate` is
  now wrapped in `set +u`/`set -u`. This also masked the pipefail bug above: you could not
  re-run the script to discover that re-running was broken.
- **detectron2 was installed editable against a `mktemp -d` clone**, leaving the module
  permanently resolving into `/tmp` — the environment broke as soon as that directory was
  reaped — and the temp tree leaked because nothing removed it. Now a non-editable install
  with cleanup. (The `third_party/` checkouts stay editable deliberately: the installer
  patches them and the patched tree must be the imported one. detectron2 is unmodified.)
- **`reifyscene.__version__` had been stale at `0.1.0` for 49 releases.** `VERSION` is the
  single source of truth — `pyproject.toml` reads it at build time and the release workflow
  reads it to name the tag — so a hardcoded `__version__` was a fourth copy nothing kept in
  sync. Now resolved: `VERSION` file first (never stale between a bump and the next
  reinstall, which is every version-bump PR), dist metadata as the fallback for a built
  wheel, which does not ship `VERSION`.

- **An unbounded `setuptools` upgrade escaped the lock the same way.** The bootstrap ran
  `pip install --upgrade pip setuptools wheel`, so a fresh env landed on setuptools 84
  while the pinned torch declares `setuptools<82`; the long-lived env sat on 81 only
  because that was current when it was built. Now bounded by `SETUPTOOLS_MAX_SPEC`
  (default `<82`), to be raised alongside a torch pin bump.
- **The regenerated lockfile was itself unusable, twice**, and nothing checked. `pip freeze`
  records whatever the env contains — including an internally inconsistent state — and emits
  unportable direct-URL lines for anything installed from a local path. Both shipped: a lock
  pinning `mypy==2.3.1` beside `pathspec==0.12.1` (mypy needs `pathspec>=1`) that pip could
  not resolve at all, and a `detectron2 @ file:///tmp/...` line pointing at an already-deleted
  build directory. `gen_env_locks.sh` now rejects `@ file://` lines and resolves the lock with
  `pip install --dry-run` before writing it, so an uninstallable lock cannot be committed.

### Added
- **`--create-env`** creates the conda environment when it does not exist. Previously the
  installer could only *activate* an env, so "run the setup script" silently assumed one
  someone had already built by hand — the repo was not reproducible from scratch.
  Deliberately opt-in: without the flag a missing env is a hard error naming it, so a typo
  in `--conda-env` fails loudly instead of silently building a multi-GB environment.
  `--conda-env` now defaults to `reifyscene`.
- The Python version for a created env is read from the lockfile header rather than
  hardcoded, so the script and the pins cannot drift. An existing env is checked against
  it: MAJOR.MINOR must match (wheels are tagged `cp311`, so a 3.11 tree cannot install on
  3.12), while a PATCH difference is reported and allowed — the patch level is not part of
  the ABI, and requiring an exact match would fail on a working env every time Python is
  updated. (A fresh `--create-env` produced 3.11.16 against a lockfile captured under
  3.11.15, exercising exactly this path.)
- `scripts/locks/reifyscene.lock` replaces `room4d.lock`, which is **removed** now that a
  from-scratch `reifyscene` env reaches the same suite result as the long-lived one
  (373 passed, 1 skipped). `gen_env_locks.sh` now maps `reifyscene -> reifyscene.lock`.
  `room4d_som_ref.lock` is unchanged — that reference environment keeps its name.

- **The working directory is renamed** `room4d` ->
  `reifyscene`, completing the rename. This breaks things and the
  repair is part of the change: seven **editable installs** hardcode the absolute repo path, so
  `vggt`, `cotracker`, `sam2`, `hloc` and `moge` all stopped importing until
  `setup_reifyscene.sh` was re-run to reinstall them against the new location. 39 repo (and 51
  `third_party`) `__pycache__` directories also held the old path baked into `.pyc` files, which
  made pytest report tests at `../../project/...`; cleared. The Shape-of-Motion env is
  unaffected — it has no editable installs bound to the repo path. `tests/test_gt_scoring.py`'s
  `data` default and one `TODOS.md` path are updated. Suite verified at 373 passed / 1 skipped,
  including the 33 real-data `data` tests that would silently skip if that path were wrong.
- **`setup_reference_tool.sh` gains `--create-env` too**, so both installers behave the same way:
  creation is opt-in, a missing env without the flag is a hard error naming it, and the Python
  version comes from the lockfile header rather than a hardcoded `PINNED_PYTHON` (now only a
  fallback). It previously created the env *automatically* — a typo in `--conda-env` would
  silently build ~21 GB including a gsplat source build. **Verified from scratch**: a clean
  `--create-env` build exits 0 with all three smoke tests green (torch GPU matmul on the 5090,
  cuML KMeans on GPU, gsplat rasterization) and reproduces the existing environment *exactly* —
  261 packages on both sides, zero differences. conda produced Python 3.10.21 against a lockfile
  captured under 3.10.20, exercising the patch-tolerance path a second time; an exact-match check
  would have failed a perfectly good build.
- **`setup_reference_tool.sh` had the same unbounded-`setuptools` escape** and is bounded the same
  way. Its pinned torch (2.11.0+cu128) also declares `setuptools<82`; that env sat on 70.2.0 only
  because that was current when it was last built, so a fresh build today would have installed 84
  and violated it. Verified by re-running the script: setuptools now lands on 81.0.0. That is the
  fourth package found escaping a lockfile this way. (The script was already correct about the
  other big one — `run_conda_relaxed()` guards `set -u` around conda calls, the very bug that made
  `setup_reifyscene.sh` runnable only once per env; the fix pattern existed here and had simply
  not been carried across.)
- **The Shape-of-Motion reference conda env is renamed `room4d_som_ref` -> `reifyscene_som_ref`**
  (renamed in place with `conda rename`, 21 GB, verified: torch 2.11.0+cu128, CUDA available),
  along with its lockfile and every reference — `gen_env_locks.sh`'s mapping,
  `run_reference_benchmark.py`'s `DEFAULT_CONDA_ENV`, and `setup_reference_tool.sh`'s default.
  Prose that pointed at "the `room4d` env" for day-to-day work now points at `reifyscene`, which
  is the development environment going forward. The only old-name references left in the repo
  are deliberate: the `.room4d_cache` gitignore line (so the not-yet-deleted cache stays
  ignored), the two paths naming the `project` working directory (renamed separately),
  and the README's link to the rename entry.
- **README** now leads with reconstruction from casually captured, uncalibrated video — *one clip
  or several* — rather than "single-video", matching the multi-stream architecture (`stream_id`
  runs through the core types and the `sync` stage); the works-today / designed-not-built split
  below it is unchanged and still says multi-stream fusion is not built. The inline rename note
  is replaced by a link to the `0.2.0.1` entry.
### Changed
- **detectron2 is no longer part of the standard install, and is pinned to a `main` commit**
  (`a2f4a877`, 2026-08-19) rather than the `v0.6` tag. v0.6 (Nov 2021) pins `black==21.4b2`,
  which requires `pathspec<1` and so cannot coexist with a modern mypy — that pair is what
  made the generated lock unresolvable. It had also been building on every full setup since
  the FP5a Sapiens rung landed, for a backend that has never existed: `detectron2` is imported
  by no Python file in the repo, in any commit, and `densepose_wrapper.py` reports
  `unavailable` unconditionally. The `--with-detectron2` flag is retained, off by default:
  DensePose is intended as the **permissively-licensed dense top rung** — the one to reach
  for when Sapiens is ruled out by its CC-BY-NC-4.0 licence, its gated weights, or its size
  — since the dense ladder currently drops straight from Sapiens' four dense modalities to
  17 sparse Keypoint-RCNN points. Wiring that rung (and resolving detectron2's `iopath<0.1.10` cap
  against SAM-2's `iopath>=0.1.10`) is follow-up work. The Sapiens-unavailable fallback is
  unaffected — `pose2d.py`'s Keypoint-RCNN rung is torchvision-backed, not detectron2.
- `pyproject.toml` description no longer reads "Prototype room-scale 4D reconstruction
  pipeline scaffold".

## [0.2.0.1] - 2026-09-10

### Changed
- **Renamed the project and the Python package `room4d` → `reifyscene`.** "room4d"
  ("room-sized 4D") was a working name; the scope has grown to an *interpretable,
  queryable, semantically editable* 4D reconstruction, and "reify" — to make something
  abstract concrete and manipulable — names that goal. Mechanical and literal, not a
  rebrand: no behavior changes, and the test suite is unchanged at 373 passed / 1 skipped
  before and after.
  - package directory `room4d/` → `reifyscene/` (226 files); all imports updated
  - `Room4DPipeline` → `ReifyScenePipeline` (`reifyscene/api/pipeline.py`)
  - distribution name and CLI entry point: `reifyscene = "reifyscene.cli.main:app"`,
    so the command is now `reifyscene` (was `room4d`)
  - scripts renamed: `setup_room4d.sh` → `setup_reifyscene.sh`,
    `run_single_video_room4d.py` → `run_single_video_reifyscene.py`,
    `run_calibration_room4d.py` → `run_calibration_reifyscene.py`
  - env vars `ROOM4D_*` → `REIFYSCENE_*` (`REIFYSCENE_LOCK_FILE`, `REIFYSCENE_RUN_VGGT`,
    `REIFYSCENE_DATA`)
  - benchmark `method_name` (and its Python fallbacks in `run_render_proof.py`):
    `room4d` → `reifyscene`, `room4d-dynamic` → `reifyscene-dynamic`. This is only the
    display label in `reference_benchmark.json`; records written before this release
    carry the old label but remain directly comparable.
  - default run name / output dir: `artifacts/runs/room4d_dev` → `artifacts/runs/reifyscene_dev`

**Everything above this entry is left untouched, deliberately.** Prior CHANGELOG entries
describe what shipped *as `room4d`* at the time; rewriting them would falsify the record.
Expect to see the old name throughout the history below.

**Deliberately NOT renamed:**
- **The internal cache path is now `.reifyscene_cache`**, so a fresh cache is built going
  forward. The existing `.room4d_cache/` is left on disk (and still gitignored) until it
  is confirmed disposable, then deleted. Nothing reads it any more.
- **The conda environments** `room4d` and `room4d_som_ref` keep their names, along with
  `scripts/locks/room4d.lock` / `room4d_som_ref.lock` and every reference to them
  (`gen_env_locks.sh`, `run_reference_benchmark.py`'s `DEFAULT_CONDA_ENV`,
  `setup_reference_tool.sh`). These name *external* machine state, not code; renaming
  them means recreating the environments, which is a separate decision.
- **The repository working directory** `the project folder`, which is why
  `tests/test_gt_scoring.py`'s `data` default and one `TODOS.md` path still say
  `project`. Both need updating when that directory is renamed.

## [0.2.0.0] - 2026-07-08

**First release to `main`** — ships the accumulated `develop` batch (perception-stack
redesign → end-to-end pipeline robustness → frame-directory ingest → single-video dynamic-4D
reconstruction → novel-view-quality fixes → `--keyframes-all` → the scene-assembly speedup).
No code change over `0.1.31.5`; the MINOR bump marks the milestone: **single-video (monocular)
dynamic-4D reconstruction now works end-to-end on real, self-reconstructed geometry**
*(internally: the G1-single gate)*.

The pipeline holds on *native* (VGGT-Long) geometry, is now practical (scene assembly
**8.3 h → 12.5 min, ~40×**), and its novel-view quality ceiling is characterized and liftable:
held-out novel-view quality is limited by the reconstruction's chunk-stitch pose drift, which
`pose_refine` (bundle adjustment) cuts 2.3°→0.71° (competitive with a single-pass VGGT
reconstruction). Honest full-frame native number: **VGGT-Long + pose_refine = 12.56
masked-mPSNR** (vs a GT-oracle upper bound ~19 and the Shape-of-Motion reference 17.42; the gap
is the cost of using self-reconstructed rather than ground-truth geometry). This clears the way
for the next phase: the queryable 4D scene graph — perceive a moving scene into a graph, query
it, render an answer, and diff the re-perceived result against the original *(internally:
phases P3–P7)*. Per-feature detail lives under the `0.1.x` entries below.

### Docs
- `README` refreshed to match this release: replaced the earlier v0.1.4 "skeleton / stubs"
  framing with the current 15-stage single-video dynamic-4D pipeline, an honest
  works-today / partial / not-yet-built status, the `run_single_video_room4d.py` invocation
  (superseding `run_single_video_vggt.py`), and the v0.2.0.0 novel-view results.

## [0.1.31.5] - 2026-07-07

### Fixed
- **4D scene assembly took ~8.3 h on the full-frame (426f) spin oracle — a super-linear (~O(T²))
  blowup in the in-assembler all-pairs point tracker (`TAPIRWrapper.compute_all_pairs`), not the
  motion init.** The tracker (a faithful port of Shape-of-Motion's *offline* preproc) seeds a dense
  foreground query grid in **every** frame and tracks each point to **every** frame, and the model
  recomputed the full-clip ResNet feature pyramid on **every** 128-point forward. At 426 frames
  that is ~10⁴ redundant feature passes over ~850k query points — ~95% of which the native lift
  immediately discards (it keeps only ~`num_fg/T` per frame, uniformly at random). Four fixes take
  wall-time **8.3 h → 12.5 min (~40×)** while *holding* masked-NVS (spin oracle, full-val n=722:
  **19.639 → 19.857 mPSNR**, mSSIM 0.585→0.595, mLPIPS 0.331→0.325; `dynamic_fg` 35,835→35,687):
  - **Hoist the feature pyramid.** `TAPIR.forward` gains an optional `feature_grids=` passthrough;
    `compute_all_pairs` computes the pyramid **once per clip** and reuses it across every query
    frame/chunk (bit-identical tracks; verified 0.0px under matched RNG).
  - **Query budget (the dominant lever, ~7×).** `compute_all_pairs` caps tracked queries to
    `max_total_queries` (spread evenly per frame), subsampling **before** tracking so we only track
    what the lift keeps. The dynamic scene path defaults it to `3 × dynamic_num_fg`.
  - **Coarser grid.** The dynamic tracker defaults to stride 8 (was 4).
  - **Determinism.** The tracker seeds the global torch RNG before the all-pairs loop —
    `estimate_trajectories`' unseeded `randperm` previously made tracks vary ~8px run-to-run.

### Changed
- **Dynamic scene-assembly tracker defaults** (`SingleVideoSceneAssembler` → `TAPIRWrapper`):
  `grid_size=8`, `max_total_queries=3*dynamic_num_fg`, `seed=0`. The **standalone** `TAPIRWrapper`
  defaults are unchanged (`grid_size=4`, no budget) to preserve SoM shipped-track parity. All
  overridable via `scene.tracker.*`.

### Tests
- `tests/test_tapir_wrapper.py`: new-knob config (`seed`, `max_total_queries` defaults + overrides)
  and a gated (GPU+checkpoint) behavioral test asserting the per-frame budget cap engages and
  tracks are bit-identical across runs (determinism). `tests/test_scene_render.py`: the dynamic
  assembler's coarsened/​budgeted tracker defaults, and that explicit `tracker` config still wins.
  Full suite: **373 passed, 1 skipped**.

## [0.1.31.4] - 2026-07-06

### Added
- **`--keyframes-all` runner flag** — use *every* frame as a keyframe, for dense reconstruction /
  benchmark-protocol parity (e.g. training the dynamic scene on a sequence's full train split to
  compare against SoM's per-scene numbers, rather than a sparse keyframe subset). It overrides
  `--keyframe-stride-sec` / `--max-keyframes` (previously the only way to approximate this was to
  "game" a tiny stride + a huge cap). Plumbs a `keyframes_all` config flag through
  `MultiVideoDataset.sample_keyframes(all_frames=...)` (stride 1, no cap) and the pipeline ingest
  call site.
  - **Resume-signature is backward-compatible:** `keyframes_all` is only recorded in the ingest
    stage signature when *enabled*, so the default (strided) signature stays byte-identical to
    pre-flag artifacts — they resume unchanged; only an actual all-frames run gets a distinct
    signature that correctly invalidates a strided cache.

### Tests
- `tests/test_keyframe_sampling.py`: `sample_keyframes(all_frames=True)` emits every frame and
  ignores stride/cap (single- and multi-stream, zero-frame guard); and the resume-signature
  backward-compat invariant (default == pre-flag; enabled is distinct). Full suite: **370 passed,
  1 skipped**.

## [0.1.31.3] - 2026-07-05

### Fixed
- **`pose_refine` inflated the scene by thousands× — a BA gauge-scale drift that made the
  dynamic reconstruction degenerate on real (VGGT) geometry.** Found during the honest VGGT
  G1-single pass (spin): with `pose_refine` enabled, held-out masked-NVS collapsed to **~4.6**
  and the background gaussian count exploded (~2.4M) even though train-view PSNR stayed high
  (~26). Root cause: pycolmap BA optimizes reprojection, which is invariant to a global
  similarity of `{poses, points}`; with an unpinned datum the overall **scale walks freely along
  that null-space** (empirically ~4,000–20,000× the input depth scale). The scene assembler
  places gaussians from the *untouched* VGGT depth at the input scale, so refined poses at a
  wildly different scale wreck it. Confirmed uniform (not outlier-driven): per-camera radius
  ratios were tight around the median.
  - `geometry/pose_refine.py::_refine_stream`: after BA, **re-anchor the refined poses back onto
    the input-geometry frame with a single orientation-based Sim(3)** (`_sim3_src_to_dst` /
    `_apply_sim3_to_c2w`, rotation from camera orientations — well-conditioned for a looking-around
    rig — scale + translation from centers). Because it is a global similarity it **preserves BA's
    relative pose gains**, and because reprojection is invariant to it **no re-BA is needed**. Depth
    and point maps are left as-is (already at input scale). The recovered scale is logged.
  - **Validated end-to-end (VGGT geometry, spin, 29 keyframes, dynamic path → render → masked-NVS):**
    held-out val mPSNR **10.29 (pose_refine off) → 11.95 (on + re-anchor)**, a clean **+1.66 dB**
    pose-refinement win, with per-camera rotation vs GT improved 0.88° → 0.615°. Before the fix,
    `on` was **4.6** (degenerate); the camera-center span went 12886 → 0.607 (== the input scale)
    and the bg gaussian count 2.4M → ~300k.
- **`refine_focal` is *not* the scale trigger, and BA's poses+focal must be kept together.** Two
  diagnostics fell out of the fix: (1) disabling focal refinement does *not* prevent the blow-up
  (still ~4,300×) and it makes the poses *worse* (fixed-focal error aliases into pose, 0.615° →
  1.25°); (2) pairing BA's refined poses with the depth-native focal (instead of BA's refined
  focal) *hurts* NVS (11.95 → 10.66) — BA optimizes poses and shared focal jointly, so the refined
  poses are only consistent with the refined K. `pose_refine` keeps BA's refined focal downstream
  (unchanged behavior; now documented in `pipeline.py::refine_poses`).

### Tests
- `tests/test_pose_refine.py::test_reanchor_sim3_recovers_gauge_drift`: applies a known
  global-similarity drift (5000× scale + rotation + translation) to a seed pose set and asserts the
  orientation-based re-anchor inverts it exactly (recovered inverse scale; poses land back on the
  seed). Full suite: **365 passed, 1 skipped**.

## [0.1.31.2] - 2026-07-05

### Fixed
- **Dynamic masked-NVS was garbage (mPSNR ~5) due to held-out-camera *placement*, not
  reconstruction quality.** Diagnosed: train-view reprojection is **24.3** (great) while held-out
  val is ~5 → a placement bug. The dynamic init computes `scene_norm` on the *already
  ground-plane-canonicalized* geometry, so the bundle's `scene_norm.transfm` maps only
  canonicalized→scene; but val cameras are placed at scoring time from **raw GT** poses, missing
  the ~45° Z-up ground-plane rotation → they render the wrong place.
  - `scene/assembler.py::_fold_ground_plane_into_norm`: fold the geometry's ground-plane transform
    (`metadata['world_frame']['transform_world_old_to_new']`) into `scene_norm.transfm` so the
    bundle carries the **full raw-world → scene** transform (`transfm @ T_gp`; convention from
    `world_frame.apply`: `w2c_new = w2c_old @ inv(T_gp)`). No-op when no ground plane ran.
  - **Validated end-to-end (GT-oracle geometry, spin, 29 dense keyframes):** held-out masked-NVS
    **mPSNR 4.99 → 15.50** (SSIM 0.36→0.49, LPIPS 0.85→0.43), confirmed both by an isolated
    orig-vs-fixed comparison on the same trained scene and by a full in-pipeline scene→render→score
    re-run. **Residual gap to the ~20.7 F0b target** (train 24.3 → val 15.5, ~9 dB) is a separate,
    smaller matter — likely genuine monocular novel-view difficulty (spin is harder than F0b's
    windmill) plus training-iteration headroom, not another dominant bug.

### Tests
- `tests/test_dynamic_scene.py`: `_fold_ground_plane_into_norm` composes the ground-plane transform
  (identity `scene_norm` ∘ `T_gp` == `T_gp`) and is a no-op when `world_frame` is absent.
  Full suite: **364 passed, 1 skipped**.

## [0.1.31.1] - 2026-07-05

### Fixed
- **Dynamic 4D reconstruction now runs via the runner** — the G1-single stress test found that
  *every* runner-driven render came out `kind:static`. Three bugs in the (never-run-in-pipeline)
  dynamic path:
  - `api/stages.py`: the **scene stage's registry `run_fn` never passed the segment
    `instance_artifact`** (defaulted to `None`), and `segment` wasn't a scene dependency — so the
    runner path always handed the dynamic assembler **zero foreground**. (`pipeline.run()` passed
    it; the registry the runner uses did not.) *The operative bug.* Now `segment` is a scene dep and
    the masks are passed through.
  - `scene/assembler.py`: `_fg_mask_by_frame` read `frame_index`, which is only built at save-time
    (`rebuild_indices`/`save`) — so an **in-memory (pre-save) segment artifact had it empty** → no
    foreground. Rebuild from `observations` when empty (mirrors `save()`).
  - `scene/assembler.py`: `_assemble_dynamic` **crashed at `quantile()` on empty foreground**; now
    guarded (`frames_with_fg<2` upfront + a try/except backstop) and falls back to a **static** scene
    (the documented "no foreground → effectively static" intent). Extracted `_assemble_static` as the
    shared fallback.
  - **Validated** (GT-oracle geometry, spin, 29 dense keyframes): scene `kind=dynamic` with **35,257
    fg + 100,000 bg gaussians + 10 SE(3) motion bases**; render → dynamic `flythrough.mp4`. **Known
    open (next branch):** in-pipeline masked-NVS quality is poor (mPSNR ~5 vs the ~20.7 target) —
    the held-out val-camera placement / scene-norm path needs its own investigation.

### Changed
- **Design doc: G1 gate split into G1-single + G1-multi** (§7). The back-half contribution
  (queryable-graph→query→compiler→round-trip-eval) is **single-stream-viable**, so BLOCK 2 (P3–P7)
  is re-gated on **G1-single** (single-stream, dynamic 4D). **G1-multi** (multi-stream: F1 sync + F2
  fusion + F4) becomes a parallel track, off the critical path to the contribution.

### Tests
- `tests/test_dynamic_scene.py` (3): scene stage depends on `segment`; `_fg_mask_by_frame` rebuilds
  from observations when `frame_index` is empty (the in-memory case); multi-instance-per-frame union.
  Full suite: **360 passed, 1 skipped**.

## [0.1.31.0] - 2026-07-02

### Added
- **Complete, safe GT-oracle geometry + shared depth helper.** The dataset backend now emits
  the *full* geometry contract (poses + depth + **world point maps** + RGB linkage), so a
  GT-perfect run flows all the way to a rendered novel view.
  - `common/depth.py` (new): one definition of **valid depth** (`valid_depth_mask` — finite &
    `> near`, optional `far`) + a dense `unproject_depth_to_world` (world-from-camera, OpenCV
    pinhole — the VGGT convention), returning **NaN at invalid depth** with a validity mask.
  - `geometry/dataset_wrapper.py`: back-projects GT depth → `pointmaps_by_frame` (NaN at holes,
    confidence = validity), so ground-plane canonicalization and every point-map consumer treat
    the oracle like any other backend. Also wires `metadata['debug_image_paths']` (the GT RGB
    frames) so scene/render seeding can find images — the oracle is now **renderable**.

### Fixed
- **Depth-map consumers are now zero/NaN safe.** Zero depth is *finite*, so a plain `isfinite`
  guard back-projected depth holes to a spurious point at the camera origin.
  - `lifting/backproject.py::Backprojector`: flags `~isfinite(z) | z<=near` → NaN (also fixes
    the optimizer's `_relift_tracks`, which routes through it).
  - `lifting/geometry_provider.py`: the render frame-stack's `valids` mask now comes from
    `valid_depth_mask(depth)` instead of unconditional ones — a zero/NaN-depth pixel is no
    longer fed to the render trainer as valid supervision.
  - (`render/seed.py` and the fused `compose`/`scale_reconcile` were already safe; verified.)

### Tests
- `tests/test_depth_safety.py` (4): `valid_depth_mask` excludes zero/NaN/negative + far bound;
  `unproject_depth_to_world` (identity pose + NaN holes); `Backprojector` zero/NaN → NaN (not
  origin); dataset oracle emits complete safe point maps + RGB linkage.
- **Validated end-to-end on spin (GT oracle from frames):** point maps produced (finite frac
  0.90, NaN holes), `ground_plane_applied: True`, and the full chain runs rc=0 through render →
  **1.5M seed gaussians + a `flythrough.mp4`** novel view. The whole pipeline consumes NaN-hole
  depth without a crash. Full suite: **360 passed, 1 skipped**.

## [0.1.30.0] - 2026-07-02

### Added
- **Lossless frame-directory ingest + intrinsics-prior wiring** — closes the two P1 gaps the
  end-to-end run surfaced (fused geometry not pipeline-robust; no dataset/frame-dir ingest).
  - `ingest/decoder.py`: `decode()` now accepts a **directory** and ingests it as a lossless
    frame sequence (no re-encode) — `stream_id` + `frame_idx` are **parsed from filenames**
    (`0_00060.png` → cam `"0"`, idx `60`) so dataset ground-truth (`{cam}_{idx}.ext`) aligns.
    `get_frame`/`iter_frames` load originals via `cv2.imread`. Single-camera-per-dir scope
    (multi-camera split deferred). Videos are unchanged.
  - `run_single_video_room4d.py`: `--frames-dir` (lossless, alternative to `--video`; exactly
    one required — you can now A/B the same data as frames vs a video) + `--geometry-camera-params
    "fx,fy,cx,cy"` (sets `geometry.camera_params` + `fix_intrinsics` for the COLMAP pose rung).
  - **Why it matters (validated on spin):** the earlier lossy re-encode was a *test-harness*
    artifact, not a pipeline defect. From **raw frames + the GT-focal prior**, COLMAP registers
    **71/71** and the FP6b fused geometry spans **71 frames / 40,189 obs** (residual median 0.996,
    healthy) — vs a degraded **3-frame** fused set on the lossy video. The **dataset GT backend**
    now resolves (`dataset_gt`, `stream_id="0"` aligns) instead of falling to empty. One change,
    both previously-broken geometry combos fixed. Findings doc updated (gaps #2/#3 → FIXED).

### Tests
- `tests/test_frame_dir_ingest.py` (9): filename parsing; frame-dir decode (camera + indices +
  size); **lossless** `get_frame` (exact match to the source, by parsed index not position);
  missing-index/empty-dir errors; `iter_frames` order; no-camera-prefix fallback; multi-camera
  single-camera scope; `preprocess` end-to-end on a frame dir (parsed `stream_id` flows through).
  Full suite: **356 passed, 1 skipped**.

## [0.1.29.1] - 2026-07-02

### Fixed
- **Runner crashed mid-run on a stale stage-label map.** `run_single_video_room4d.py` kept a
  hardcoded `stage_labels` dict (labeled "/11") that had drifted from `STAGE_ORDER` — missing
  `dyn_mask` (FP6b step 3) plus the earlier `segment`/`reid`/`semantics`, so a full run hit
  `KeyError: 'dyn_mask'`. Labels are now **derived from `STAGE_ORDER`** (single source of truth),
  so numbering/coverage can't drift as stages are added. Display-only; no stage behavior changes.

### Docs
- **`docs/design/2026-07-02-e2e-pipeline-run-findings.md`** — first full end-to-end pipeline run
  on real data (spin), across the geometry-backend matrix (vggt / fused colmap+moge / dataset).
  **Combo A (vggt) runs end-to-end → rendered novel-view flythrough.** Combo B (FP6b fused) runs
  but isn't pipeline-robust (COLMAP erratic on lossy video frames; MoGe reliable; composer degrades
  correctly). Combo C (dataset GT) is incompatible with video ingest (falls to graceful-empty).
  Includes a severity-ranked gap ledger + recommended FP6b-hardening next steps (lossless
  frame-sequence ingest + intrinsics-prior/density as first-class fused config). The ladder's
  graceful degradation held throughout — no crashes past the label fix.

## [0.1.29.0] - 2026-07-01

### Added
- **FP6b step 3 — `dyn_mask` robustness rung + static-superset reorder + reconstruction
  health-check.** Closes the two robustness gaps step 2 surfaced: protects the COLMAP pose
  rung from dynamic-content correspondences, and makes its intermittently-degenerate draws
  *observable*.
  - **`perception/dyn_mask/`** — a new coarse, **geometry-free** stage emitting per-frame
    dynamic masks (`DynMaskArtifact`, masks persisted as `.npz`). Ladder:
    **detector** (COCO Mask R-CNN movable-category masks, reuses `ObjectDetector`) →
    **all-static** floor (no-op → VGGT-on-top stays a no-op). Consumes only raw frames —
    never `segment`/geometry — which is what breaks the `segment→geometry` cycle (§4A.1).
    The general optical-flow motion-seg top rung is a deferred follow-up (no current clip
    needs it; native-swap discipline).
  - **`geometry/colmap_wrapper.py`** — the pose rung consumes `dyn_mask`, **size-gated**:
    blacks out dynamic pixels (`rgb[mask>0.5]=0`, the spike's mechanism) only when the
    median per-frame dynamic fraction clears `mask_size_gate` (default 0.10). Per the spike,
    masking is a *size-gated* help (tightens large-dynamic scenes, neutral on small ones),
    so windmill-type small-dynamic clips are correctly left unmasked. The gate decision is
    recorded in the artifact notes/metadata.
  - **`geometry/reconstruction_health.py`** — a GT-free `assess_pose_health` flags
    degenerate SfM draws via a **per-stream, self-calibrating** outlier test on consecutive
    pairwise-relative rotation (per-stream so multi-cam large angles don't false-positive;
    flagged against the stream's own median + k·MAD *and* an absolute physical cap). Stamped
    into `metadata['pose_health']`; the composer downgrades fused fidelity to
    `fused_metric_suspect` when suspect. Documented limitation: catches the outlier-jump
    degenerate type, not a globally-bent-but-locally-smooth one.
  - **Pipeline reorder (§4A.1 fourth invariant)** — `dyn_mask` slots into `STAGE_ORDER`
    between `sync` and `geometry`; geometry's deps become `(ingest, sync, dyn_mask)`. The
    ladder gates work, never reorders — VGGT-on-top runs `dyn_mask` as all-static and
    ignores it. `dyn_mask` is threaded only to the COLMAP pose rung (other backends'
    signatures untouched). Default `all-static` → strict superset, existing configs
    unchanged.
  - `scripts/fp6b_validate_dynmask.py` — quantifies the hypothesis: detector median dynamic
    fraction + size-gate decision, then the **degenerate-draw rate** (health-flagged) over N
    COLMAP runs, unmasked vs detector-masked, on spin (large-dynamic) vs windmill (gated off).
  - **Validated on spin + windmill** (60 frames; spin 8 runs/arm, windmill 4). Size-gate
    fires correctly — spin detector fraction **0.351 → MASK**, windmill **0.000 → unmasked**.
    On spin (large-dynamic), detector-masking **reduced the degenerate-draw rate 1/8 → 0/8**
    and **tightened the pairwise-max rotation tail 1.09° → 0.33°** (matching the oracle-mask
    spike), and the health-check **caught the unmasked degenerate draw**; windmill was neutral
    (0/4 both arms). Caveats: small N (direction clear, matches the spike, not statistically
    conclusive); windmill gates off via detector fraction 0.000 (COCO-blind to the pinwheel),
    so the gate's small-but-nonzero branch is exercised by the deferred flow-seg rung, not here.

### Tests
- `tests/test_dyn_mask.py` (11): `DynMaskArtifact` npz round-trip; all-static empty masks;
  detector `mask_frame` unions only movable categories (mocked); health-check ok/flagged/
  too-few-frames; COLMAP size-gate masks-when-large / skips-when-small (verifies pixels
  blacked out) + stamps `pose_health`; pipeline `dyn_mask` stage (all-static default +
  manifest); composer downgrades to `fused_metric_suspect` on suspect health. Updated three
  stage-order tests for the new stage. Full suite: **347 passed, 1 skipped**.

## [0.1.28.0] - 2026-07-01

### Added
- **FP6b step 2 — geometry pose/depth split + COLMAP↔MoGe scale reconciliation.** Fuses
  the two decomposed rungs (FP6a MoGe depth + FP6b-step1 COLMAP poses) into a single,
  *complete* **GT-free, VGGT-free metric** `GeometryArtifact` (design §4A.1). Monocular
  COLMAP is scale-free, MoGe is metric; a single global scale ties them together.
  - `geometry/scale_reconcile.py`: solves the COLMAP→metric scale from sparse-point
    reprojections vs MoGe depth (robust median of per-observation depth ratios). The solve
    is a **per-frame camera-space** comparison, so it is **gauge-free** and runs *before*
    ground-plane canonicalization (scale is the one gauge DOF the ground plane doesn't pin,
    §4A.1). Pure `solve_global_scale` (unit-tested, COLMAP-free) + thin, probe-gated
    `extract_colmap_observations` (reads the sparse model via `pycolmap`). Emits the
    **per-frame residual** (→ F2), never a crash: too few samples ⇒ identity fallback.
  - `geometry/compose.py` (`GeometryComposer`): merges pose channel + depth channel —
    poses/intrinsics from COLMAP, depth from MoGe — applies the reconciled scale to the
    camera centers, and **rigidly** transforms MoGe's camera-space points into world via
    the metric poses (MoGe's own per-frame intrinsics kept self-consistent; never
    re-projected through COLMAP's focal, §4A.1(a)). Resulting `pointmaps_by_frame` are
    world-space, matching VGGT's convention so ground-plane + downstream work unchanged.
    Graceful degradation: a missing channel falls back to `depth_only`/`poses_only`/`empty`.
  - `pipeline.initialize_geometry`: `geometry.pose_backend` + `geometry.depth_backend`
    select the **two fused ladders**; each channel is a 2-rung ladder (backend →
    graceful-empty) and the composite `CapabilityManifest` records both channels'
    rung statuses (prefixed `pose:`/`depth:`) + the resolved fidelity
    (`fused_metric`/`poses_only`/`depth_only`/`empty`). A lone `backend_name` still
    selects the single-backend path — **strict superset, existing configs unchanged.**
  - `geometry/colmap_wrapper.py`: exposes the persisted sparse model dir
    (`metadata['colmap_model_dir']`) so the reconciler can read its 3D points +
    reprojections.
  - `scripts/fp6b_validate_fused.py`: native-swap validation vs GT through the shipped
    rungs + reconciler. Reports (1) whether the per-scene offset collapses to one global
    scale (per-frame residual band — FP6a needed a per-frame oracle scale; this needs one
    global scale + a small F2 residual), (2) **metric SE(3)-APE** poses (scale fixed = 1,
    unlike step 1's Sim(3)) with the residual Sim(3) scale ≈ MoGe's per-scene bias (→ F2),
    and (3) depth vs GT for context.
  - **Validated on spin + windmill** (60 frames, healthy COLMAP draws, reproducible across
    2 runs each). Reconciliation collapses the per-scene offset to **one global scale** with
    a tight per-frame residual (spin [0.92, 1.08] med 1.00; windmill [0.95, 1.10] med 1.02).
    Trajectory **shape** is excellent — **Sim(3)-APE trans 0.3–0.7%** of extent, rot ~1° —
    so the *only* remaining metric error is MoGe's own per-scene absolute-scale bias (spin
    +14%, windmill −34%), which shows up as metric **SE(3)-APE trans (spin 3.4%, windmill
    12.9%)** and is exactly the F2 person-anchored-scale hand-off. **GT-free, VGGT-free metric
    geometry, as designed.**

### Known issues
- **Monocular COLMAP pose rung is intermittently unstable** on these low-parallax dynamic
  clips: incremental BA occasionally hits `Linear solver failure` (dense Cholesky) and lands
  a **degenerate** reconstruction (100% registered, low reproj error, but geometrically wrong
  — observed ~once each on spin and windmill across ~8 runs; healthy draws are the norm and
  reproduce tightly). **Not introduced by step 2** — a property of the step-1 rung; step 2's
  fusion is correct on healthy draws. Mitigation → FP6b step 3 (dyn_mask robustness rung +
  dense motion-preserving sampler) plus a GT-free reconstruction health-check (e.g. reject
  implausibly large consecutive pairwise-relative rotation) so degenerate draws are
  *observable* rather than silently fused.

### Tests
- `tests/test_fused_geometry.py` (11): pure scale solver (recovers a known scale, identity
  fallback below `min_obs`, skips out-of-bounds/non-finite); `GeometryComposer` full fusion
  (scale reconciled, poses scaled, world pointmaps, provenance) + all three degradations
  (`depth_only`/`poses_only`/`empty`); pipeline fused-mode wiring (composite manifest,
  `pose:`/`depth:` rungs), pose-channel failure ⇒ `depth_only` (no crash), and back-compat
  (a lone `backend_name` is not fused). Full suite: **336 passed, 1 skipped**.

## [0.1.27.0] - 2026-07-01

### Added
- **FP6b step 1 — COLMAP-SfM pose rung (native-swap).** The **pose** mirror of FP6a's
  MoGe depth rung (design §4A.1): monocular hloc+COLMAP camera egomotion, the first
  piece of the geometry pose/depth split. Gated behind a spike (below) that cleared the
  one regime the calibration arc never tested (monocular egomotion on dynamic handheld).
  - `geometry/colmap_wrapper.ColmapPoseInitializer`: runs the calibration arc's
    `HlocColmapSfMRunner` (SuperPoint + SuperGlue + NetVLAD) over **dense** frames with
    **intrinsics fixed from the focal prior** (`camera_mode=SINGLE`), reads the binary
    model via `pycolmap`, and fills `camera_states_by_stream` (poses) +
    `intrinsics_by_stream`. **Pose-only, honest:** `depthmaps`/`pointmaps` stay empty
    (depth comes from the MoGe rung). Shared `reconstruct()` core (mirrors MoGe's
    `infer_frame`) so the in-pipeline path and the validator share one code path.
  - `pipeline.initialize_geometry`: `backend_name='colmap'` force-selects the rung on the
    geometry ladder (fidelity `sfm_poses_no_depth`, probe `importable('pycolmap','hloc')`);
    the load-bearing vggt/dataset rungs stay probe-free.
  - `scripts/fp6b_validate_colmap.py`: native swap vs GT extrinsics through the shipped
    `reconstruct()`. **Result (60 frames, fixed intrinsics):** 100% registered, gauge-free
    pairwise-rel rotation **0.19°/0.09°** median (spin/windmill), Sim(3)-APE <1% translation.

### Changed
- **Design doc §7 FP6b + §4A.1.** FP6b marked **SPIKE CLEARED** (2026-07-01): monocular
  COLMAP recovers sub-degree egomotion on spin+windmill; frame **density is load-bearing**
  (dense sampler first), and `dyn_mask` is a **size-gated robustness rung**, not a hard
  gate (it tightens large-dynamic scenes, is neutral on small ones) — refining "masks
  likely needed to converge."

### Tests
- `tests/test_colmap_geometry.py` (5): empty-in→empty-out; poses-filled/no-depth artifact;
  partial registration; geometry ladder selects COLMAP + stamps the manifest; and degrades
  to graceful-empty (failed rung observable) when reconstruction raises. Full suite: 325
  passed, 1 skipped.

## [0.1.26.0] - 2026-06-30

### Added
- **FP6a — MoGe-2 metric-depth rung (native-swap) + geometry backend ladder.**
  Adds Microsoft's **MoGe-2** (MIT, ungated) single-image metric point map as an
  explicitly-selectable depth rung, and gives the `geometry` stage the
  observable-degradation machinery every other stage already had. The full
  pose/depth split + calibration-arc SfM pose rung + COLMAP↔MoGe scale
  reconciliation are scoped as **FP6b** (design §4A.1 / §7).
  - `geometry/moge_wrapper.MoGeInitializer`: per-frame MoGe-2 (`from moge.model.v2
    import MoGeModel`) → fills `depthmaps_by_frame` + camera-space
    `pointmaps_by_frame` + per-frame `intrinsics_by_stream` (normalized→pixel).
    **Depth-only, honest:** MoGe has no cross-frame poses, so
    `camera_states_by_stream` is left empty (poses come from a pose source — VGGT
    / GT / the FP6b COLMAP rung). Lazy + import-safe; shared `infer_frame()` core;
    releases VRAM after the frame loop.
  - `geometry/geometry_artifact.GeometryArtifact`: gains `attach_manifest()` /
    `capability` — the last geometry artifact that lacked a `CapabilityManifest`.
  - `pipeline.initialize_geometry`: now runs a `BackendLadder('geometry', …)` —
    the config-selected backend as the top rung + a **graceful-empty floor** (§4A
    invariant), stamping the manifest. `backend_name='moge'` force-selects the
    MoGe rung; only that rung is probe-gated (`importable('moge')`) so the
    load-bearing vggt/dataset rungs never false-skip. Auto-selecting MoGe below
    VGGT is inert (VGGT/GT already emit depth) — deferred to FP6b with the split.
  - `scripts/setup_room4d.sh --with-moge`: vendors `microsoft/MoGe` at a pinned
    commit (`--no-deps` + utils3d/opencv/scipy — self-contained), pre-caches
    **and smoke-infers** the `Ruicheng/moge-2-vitl-normal` weights (~1.3 GB, ungated).
  - `scripts/fp6_validate_moge.py`: the native-swap — scores MoGe depth vs the GT
    depth oracle (metric + scale-invariant). **Result:** depth *structure* is
    excellent (scale-aligned δ<1.25 ≈ 0.985 spin / 0.998 windmill), but absolute
    *metric* scale carries a per-scene offset (spin +13%, windmill −35%) —
    empirically confirming MoGe is a strong structural backstop, not a drop-in
    metric one (motivates the FP6b scale reconciliation).

### Changed
- **Design doc §4A.1 (new) + §4/§4A/§7.** After an adversarial review, geometry is
  reframed as **two fused ladders — pose + depth** (VGGT is a *joint* rung; MoGe /
  COLMAP are *decomposed*, so stepping down the ladder pulls in upstream inputs the
  top rung never needed). Adds the coarse geometry-free `dyn_mask` capability
  (breaks a real `segment`→`geometry` cycle), a fourth **static-superset input
  ordering** invariant, and resequences **FP6 → FP6a / FP6b**.

### Tests
- `tests/test_moge_geometry.py`: MoGe `infer_frame` shapes + normalized→pixel
  intrinsics; empty-in→empty-out; depth-filled/no-poses artifact; geometry ladder
  selects MoGe + stamps the manifest; and degrades to graceful-empty (with the
  failed rung observable in the manifest) when the rung raises. Full suite: 320
  passed, 1 skipped.

## [0.1.25.0] - 2026-06-30

### Added
- **FP5a — Sapiens dense human rung + F3 de-fabrication (ladder-ified `humans`).**
  Climbs the `humans` dense sub-ladder (design §4A / §7) with Meta's Sapiens, and
  executes the F3 subsumption by making the humans wrappers honest. FP5b
  (GVHMR world-SMPL) is deferred — its blockers (MPI-registered SMPL assets +
  gravity-aligned poses) and its payoff both live near the back-half graph.
  - `perception/humans/sapiens_wrapper.SapiensEstimator`: wraps Sapiens-lite
    **TorchScript** (pose / body-part seg / depth / normals) top-down per person
    crop — aspect-pad to 768:1024, ImageNet-normalize, decode heatmaps→keypoints
    (Goliath-308) + seg (28 parts) + depth + normals back to image space. Ungated
    HF checkpoints (`facebook/sapiens-{task}-0.6b-torchscript`), **CC-BY-NC-4.0**
    (non-commercial — fine for this no-ship research repo). Lazy + import-safe.
  - `perception/humans/dense_ladder`: the dense-ladder rung adapters
    (`SapiensDenseRung` / `KeypointRcnnRung` / `BboxOnlyRung`) — thin glue that
    does the decode-once-per-frame loop and calls each **tool wrapper's**
    `predict(frame, boxes)`. The tool wrappers stay separate + swappable.
  - `pipeline.compute_humans` (renamed from `perceive_humans`): runs
    `BackendLadder('humans', [sapiens → keypoint_rcnn → bbox_only])` gated by a
    Sapiens **checkpoint-presence** probe (`--with-sapiens`), degrading to the
    honest Keypoint-RCNN floor. `HumanArtifact` gains a `CapabilityManifest` (the
    last stage that lacked one); `HumanState`/`HumanArtifact` carry a `dense`
    payload (seg/depth/normals).
  - `scripts/setup_room4d.sh --with-sapiens`: pre-caches the 4 Sapiens 0.6B
    TorchScript checkpoints (~5 GB, ungated).
  - `scripts/fp5_validate_humans.py`: runs the Sapiens rung on real segmented
    `spin` people. **Result:** the real person track gets 308 keypoints (~250
    visible, **100% inside the box**), **8–9 body-part seg classes**, well-formed
    depth + unit normals; spurious detections correctly yield near-empty. (Humans
    output is dead data downstream until the P3 graph re-cut, so there is no
    render gate.)

### Changed
- **F3 subsumption — humans wrappers de-fabricated (kept as swappable seams).**
  `pose2d.HumanPose2DEstimator` no longer splices a fabricated skeleton for
  unmatched boxes and stamps it `'real'` — it returns `None` with an honest
  `keypoint_rcnn`/`unavailable` label. `densepose_wrapper` drops the fake
  bbox-grid colorizer (honest `unavailable`; Sapiens seg is the dense source now)
  and `smpl_fitter` drops the fabricated betas/pose (honest `skip`; GVHMR at
  FP5b). This brings the humans wrappers in line with the honest `backend_used`
  every other wrapper already reports. `detector.py` is left as a kept-but-unused
  seam (detection lives in `segment`).

### Tests
- `tests/test_humans_dense.py`: honest wrappers (densepose `unavailable`, smpl
  `skip`), the dense-ladder rungs (fake estimators, no weights), and the pure
  Sapiens decode math (heatmap→image mapping, pad removal, resize/unpad).
- `tests/test_human_perception.py`: updated for the `compute_humans` rename and
  the `humans.use_sapiens=false` floor path.

## [0.1.24.0] - 2026-06-30

### Added
- **FP4a — SigLIP2 open-vocab semantics rung (new `semantics` stage).** Names the
  class-agnostic instances `segment` produces: a `semantics` stage attaches an
  open-vocab **category** (+ a SigLIP image embedding) to each canonical
  (reid-merged) tracklet, replacing the empty/`""` category the SAM 2 rung leaves
  behind (which previously flowed straight to an empty graph-node label).
  Qwen2.5-VL free-form attributes/captions arrive at FP4b.
  - **New `semantics` stage** between `reid` and `objects` (`STAGE_ORDER`,
    `stages.py`; deps ingest+segment+reid). `pipeline.compute_semantics` runs a
    `BackendLadder('semantics', [siglip2 → coco80 → unknown])`: the SigLIP2 top
    rung (probe `importable('transformers')`) zero-shot matches masked tracklet
    crops against a candidate vocabulary; the floor carries the detector's COCO-80
    `concept`, degrading to `"unknown"`. `perceive_objects` now reads the semantics
    category (falling back to the COCO concept) into `ObjectState.category` → graph node.
  - `perception/semantics/semantics_artifact.SemanticsArtifact`: per-canonical-track
    `SemanticRecord` (category, score, top-k, attributes, caption) + SigLIP
    embeddings (npz), mirroring `ReidArtifact` persistence.
  - `perception/semantics/siglip_semantics`: `SiglipSemantics` (lazy transformers
    SigLIP2; per-track masked-crop image features, mean-pooled, zero-shot scored)
    + `FloorSemantics` (COCO concept → `"unknown"`) + the pure `score_zero_shot`
    helper. Default model `google/siglip2-so400m-patch14-384` (ungated/Apache); a
    `_pooled` shim handles the transformers-5.x `BaseModelOutputWithPooling` return.
  - `scripts/setup_room4d.sh --with-siglip2`: installs transformers + pre-caches
    the SigLIP2 weights (idempotent, ungated; mirrors `--with-dinov2`).
  - `scripts/fp4_validate_semantics.py`: two-gate validation driver (`semantics`
    does not feed render, so there is no mPSNR gate). **(1) COCO-agreement** on
    `spin` (a person scene) — SigLIP2 zero-shot vs Mask R-CNN over COCO-80:
    **top-1 62% / top-5 77%** raw (~85% on persistent objects; person 4/4, chair
    3/4 — every miss is a one-frame detector false positive on background clutter).
    **(2) windmill open-vocab** — SigLIP2 over an extended vocab on SAM 2 tracks:
    names the fragmented petals only loosely (small non-COCO part-crops + SAM 2
    over-segmentation are an honest limit; whole-object naming wants the
    reid-merged whole mask).

### Tests
- `tests/test_siglip_semantics.py`: the pure `score_zero_shot` (argmax / top-k /
  empty-vocab / min-score abstain), `FloorSemantics` (majority concept,
  reid-canonical grouping, unknown), and a fake-embed `SiglipSemantics.run`
  integration (decode / crop / pool / score, no weights).
- `tests/test_resume_pipeline.py`, `tests/test_stage_window.py`: updated for the
  new `semantics` stage in `STAGE_ORDER` + the `objects` dependency on it.

## [0.1.23.0] - 2026-06-30

### Added
- **FP3 — DINOv2 cross-occlusion re-ID top rung (climbs the `reid` ladder).**
  A frozen DINO ViT appearance rung above the ResNet18 floor that emits a
  `merge_map` re-acquiring identity *across occlusion gaps* — two tracklets the
  segmenter gave separate ids but that are the same entity (it left, came back)
  fold onto one canonical id. Probe-gated on `transformers`; degrades to the
  ResNet18 floor (empty merge_map) when the dep/weights are absent, so the
  pipeline never breaks. **DINOv3-L swaps in by `model_id` alone** once HF access
  lands — the same `AutoModel.from_pretrained` call (default
  `facebook/dinov2-large`, ungated/Apache; gated
  `facebook/dinov3-vitl16-pretrain-lvd1689m` later).
  - `perception/reid/dino_reid.DinoReid`: per-tracklet mean DINO features
    (decode-once-per-frame, masked crops, L2-normalized), with a lazy model load
    so constructing the rung is import-safe when `transformers` is absent.
  - `perception/reid/dino_reid.build_cross_occlusion_merge_map`: a **pure**
    (no-torch) merge builder. Candidates are same-stream, span-**disjoint** pairs
    (co-visible tracklets are a different problem and are never merged); a pair
    merges only if it is a **mutual nearest neighbour** (plus an optional
    Lowe-style ratio test, off by default), folding onto the earlier-appearing
    canonical. Mutual-best matching is a *matching*, so it merges pairs, not
    multi-gap chains (a deliberate v1 precision-over-completeness choice).
  - `api/pipeline.compute_reid`: prepends the `dinov2` rung above `resnet18` in
    the reid ladder; the reid runner contract is now `(embeddings, merge_map)`
    (mirrors segment's `(observations, mask_arrays)`), threaded into
    `ReidArtifact.merge_map` (previously hardcoded empty).
  - `scripts/setup_room4d.sh --with-dinov2`: installs HuggingFace transformers +
    pre-caches the DINOv2-L weights (idempotent; the gated DINOv3-L path needs
    `hf auth login` + granted access).
  - `scripts/fp3_validate_reid_merge.py`: the synthetic-occlusion validation
    driver — splits real SAM 2 windmill tracks into time-disjoint halves and
    measures merge precision/recall on real DINOv2 features. **Result: precision
    1.0 (0 false merges), recall 2/5** on the adversarial near-identical spinning
    petals — the rung abstains rather than mis-joining ambiguous identical
    instances (`reid` does not feed render, so there is no mPSNR gate;
    precision-over-recall is the correct bias for a graph identity merge, and
    cross-occlusion re-acquisition remains the design's open risk).

### Tests
- `tests/test_dino_reid.py`: the mutual-NN + ratio merge builder (mutual
  rejection, ratio rejection, co-visible / cross-stream exclusion,
  earliest-canonical, empty input) + a fake-embed `DinoReid.run` integration test
  (decode / pool / span / merge, no weights).
- `tests/test_reid_stage.py`: updated for the `(embeddings, merge_map)` floor
  contract (the floor leaves the merge_map empty).

## [0.1.22.0] - 2026-06-29

### Added
- **FP2c — motion-gated dynamic foreground (closes the perception→render loop).**
  The dynamic-init lift now keeps only *ever-moved* tracks as deformable
  foreground and reroutes static tracks to the background, so a broad
  class-agnostic segment foreground (SAM 2's "segment everything") is cleaned by
  **motion** rather than per-scene appearance heuristics. **Validated end to end
  on paper-windmill: SAM 2 fg + motion gate → 20.43 masked mPSNR vs the 20.72
  oracle gate** — the fully-native front half holds the oracle's novel-view
  quality within 0.3 dB (n=419 val), the SAM 2 foreground (66% frame coverage)
  gated down to the moving pinwheel.
  - `lifting/native_track_lift.gate_tracks_by_motion`: a track is foreground iff
    the **p95** of its per-frame world-space displacement from its median center
    exceeds a threshold — the robust "did it ever move" test. (A *median* reducer
    averages brief motion away — a static-then-moving object — and a raw *max*
    trips on single-frame lift glitches; p95 catches genuine motion and ignores
    one-frame outliers.) Static tracks become static background gaussians at their
    median position — their segment-fg-masked regions would otherwise be holes.
    Threaded through `NativeMotionInitializer` + the scene assembler,
    **opt-in** (`scene.dynamic_motion_gate`; default off → F0b gate + suite unchanged).
  - `render/gaussian4d/dynamic_field.DynamicScene.cloud_at`: clamp the frame index
    to the motion-bases range — an out-of-range time-id (e.g. a held-out camera
    beyond the trained frames) degrades to the nearest frame instead of an
    out-of-bounds CUDA assert.
  - `scripts/f0_validate_windmill.py`: `--fg-source {dataset,sam2}` (SAM 2 union
    masks injected via a wrapped oracle-geometry provider), `--motion-gate`,
    `--tracker-grid-size`, `--max-frames` — the native-swap ladder driver for the
    oracle-vs-SAM 2 foreground comparison.

### Tests
- Motion gate: ever-moved (p95) foreground vs static-rerouted-to-background; the
  static-then-moving case where a median reducer mislabels but p95 catches it;
  no-op when every track is dynamic. Full suite: **291 passed, 1 skipped**.

## [0.1.21.0] - 2026-06-29

### Added
- **FP2a — SAM 2 `segment` top rung (prompt-free, class-agnostic).** The first
  rung of the perception climb above the FP1 Mask R-CNN floor. SAM 2 automatic
  mask generation discovers object-like masks on a keyframe (no class list, no
  prompt), a tunable object-ness filter prunes them, and the SAM 2 video predictor
  propagates each across the stream with a **persistent native ID** — real
  temporal identity in place of the floor's greedy-IoU stitching. Naming stays
  downstream (FP4 semantics); this rung emits only masks + identity. SAM 3 is the
  eventual gated top rung; SAM 2 (Apache, ungated) is the stand-in.
  - `room4d/perception/instance/sam2_segment.py` — `SAM2InstanceSegmenter`
    (discover → object-ness filter → SAM 2 video propagation → person-tag →
    `InstanceArtifact`). The existing person detector tags `kind='human'` so the
    `humans` stage keeps working class-agnostically. sam2 is lazy-imported (loaded
    only on run), so the module is import-safe when sam2 is absent.
  - `room4d/api/pipeline.py` — the `segment` ladder gains a probe-gated `sam2`
    rung (`importable('sam2')`) above `maskrcnn_iou`, built inside the runner so
    the model loads only when selected and releases after the stage. Falls to the
    Mask R-CNN floor when sam2 is not installed — the degradability invariant holds.
  - `scripts/setup_room4d.sh` — `--with-sam2`: SAM 2 pinned into `third_party/sam2`
    + the `sam2.1_hiera_large` checkpoint (direct fbaipublicfiles URL, idempotent,
    size-guarded); `SAM2_BUILD_CUDA=0` keeps the install off the CUDA-toolchain path.

### Tests
- Object-ness filter (keep object / drop background, low-stability, tiny masks) +
  bbox/IoU helpers; a sam2-gated functional test (synthetic moving-square clip →
  one persistent track across frames). Full suite: **288 passed, 1 skipped**.

## [0.1.20.0] - 2026-06-29

### Added
- **F0b — in-pipeline dynamic Gaussian render path.** The render half of the
  scene→render seam now resolves a *moving* scene end to end: `scene.kind=dynamic`
  lifts an SE(3) `DynamicScene` from the pipeline's own geometry + foreground,
  serializes it into the `SceneArtifact`, and the render stage trains + scores it.
  This closes the loop so front-half perception backends become measurable by
  their downstream novel-view quality. **Gate: 20.72 masked mPSNR on DyCheck
  paper-windmill** (own-tracker, oracle geometry, 5000 iters, n=419 val frames) —
  reproducing the standalone `run_render_proof.py` reference (~20.7) through the
  new wiring.
  - **Serialization (`room4d/render/gaussian4d/motion_init.py`):** `dump_bundle` /
    `bundle_to_scene` round-trip a `DynamicScene` + cameras through the borrowed-init
    npz schema (torch↔numpy); `SceneArtifact.dynamic_bundle` persists it as a
    separate `dynamic/scene_bundle.npz` (not inlined JSON), mirroring the static
    seed. `NativeMotionInitializer.build(return_norm=True)` surfaces the scene-norm
    so held-out dataset cameras land in the scene frame at score time.
  - **Assembler dynamic branch (`room4d/scene/assembler.py`):** `scene.kind=dynamic`
    builds a `DynamicScene` from an `ArtifactGeometryProvider` (per-frame union of
    `segment` instance masks → foreground) + in-memory packed all-pairs tracks
    (no per-pair `.npy` files); the `segment` `InstanceArtifact` is threaded through
    `Room4DPipeline.assemble_scene`. Default `kind=static` keeps F0a unchanged.
  - **Render dispatch (`room4d/render/gaussian4d/render_backend.py`):** the
    `kind=="dynamic"` branch (formerly `NotImplementedError`) trains the bundle with
    `DynamicGaussianTrainer`, writes a `trained_dynamic.npz` checkpoint, and renders
    the trajectory per `time_id`.
  - **Dynamic masked-NVS scorer (`room4d/render/eval/masked_nvs_benchmark.py`):**
    `score_render_artifact_masked_nvs_dynamic` scores held-out val views against a
    scene re-materialized per `time_id` (DyCheck protocol), wired into `fit_renderer`.
  - **`scripts/f0_validate_windmill.py`** — reproducible gate driver (borrowed-init
    wiring smoke + own-tracker gate over oracle geometry).

### Tests
- Dynamic-bundle serialization round-trip, fg-mask union → provider keys,
  scene-norm ↔ lift convention, and dynamic render dispatch (CPU empty-bundle +
  gsplat end-to-end train/checkpoint). Full suite: **284 passed, 1 skipped**.

## [0.1.19.0] - 2026-06-27

### Added
- **FP1 — perception band reordered onto the new `segment`/`reid` stages
  (floor rungs).** The pipeline now runs
  `pose_refine → segment → reid → objects → humans → tracking → graph` end-to-end
  using today's backends as the ladder floors — no new model downloads; full
  suite green (279 passed).
  - `segment` stage (`Room4DPipeline.segment` + `room4d/perception/instance/segment.py`):
    floor rung = COCO Mask R-CNN + greedy-IoU association (`person`→human-kind,
    else object-kind) → `InstanceArtifact` (per-frame masks + persistent track
    ids). The shared mask + identity source. SAM 3 is the top rung at FP2.
  - `reid` stage (`Room4DPipeline.compute_reid` + `room4d/perception/reid/reid_stage.py`):
    floor rung = mean ResNet18 embeddings per tracklet, empty merge_map →
    `ReidArtifact`. DINOv3 + cross-occlusion merges at FP3.
  - Both stages run behind the `BackendLadder` (probe → run → capability manifest)
    with a graceful-empty floor; `stages.py` registry + `STAGE_ORDER` + `run()`
    updated; resume/stage-window inherited.

### Changed
- **`objects` / `humans` re-cut to consume `segment` + `reid`** instead of
  detecting from scratch: identity is the canonical (reid-merged) track id;
  `objects` carries the COCO category (floor semantics; open-vocab at FP4),
  `humans` runs pose/DensePose/SMPL over the segment masks. `tracking` is now
  motion-only and runs after perception (geometry-based static/dynamic partition
  unchanged; mask-seeding lands with F0b).
- Point-track support (`attributes['track_ids']`) is decoupled from identity
  (§4A) and empty at FP1 — the graph's `support_track_ids` node attr empties out
  accordingly (graph is the passive builder, re-cut at P3).

### Removed
- The hand-tuned associators `room4d/perception/objects/object_assoc.py` and
  `room4d/perception/humans/human_assoc.py` (identity now comes from
  `segment`/`reid`; the ~6-weight scoring soup is gone).

## [0.1.18.1] - 2026-06-27

### Changed
- **Design doc §4B (new): two-tier sync + per-stream-until-fusion** — refines the
  F1/F2 sync design for casually-captured, independently-clocked multi-stream
  input (decision-only doc change; no code).
  - Sync becomes **two-tier**: Tier-1 a **coarse temporal prior** (audio /
    motion-energy cross-correlation) at the early `sync` stage; Tier-2 a **fine
    temporal alignment** (time offset + frame-rate/drift) estimated **jointly
    with extrinsics + scale in the joint optimizer (F2)** via cross-view
    dynamic-content consistency.
  - Records the structural consequence: most of the front-half is **single-stream
    until a fusion point** (geometry/perception/motion run per-stream; sync +
    extrinsics + scale + cross-view identity are joint at F2/F4). The registry's
    `geometry depends on sync` edge is wrong for the casual case — geometry does
    not require sync.
  - Cross-view identity rides the same fusion (cross-stream `reid` on DINOv3
    features). Validation: rig ~0 recovery is necessary-not-sufficient; casual
    fine-sync needs an unsynced known-offset test.
  - Reconciled §3 (sync row), §4 (sync/geometry stage rows), §7 (F1 Tier-1
    coarse prior; F2 Tier-2 fine sync), §8 (new risk bullet).

## [0.1.18.0] - 2026-06-26

### Added
- **FP0 — perception redesign foundation: degradable backend ladder + new
  perception artifact types** (design §4A; purely additive — no existing
  behaviour changes, not yet wired into the pipeline).
  - `room4d/common/backend_ladder.py`: `BackendRung` / `BackendLadder` /
    `CapabilityManifest`. A ladder probes its rungs top-down (best → graceful-
    empty), runs the first available one, and records an observable manifest
    (which rung ran, at what fidelity, why higher rungs were skipped —
    `not-installed` / `no-access` / `probe-error`). A resolved rung that raises
    falls through to the next rung. Generalizes the ad-hoc per-backend
    `backend_used` idiom (detector `real→heuristic`, embedder `real→histogram`)
    into one reusable, observable abstraction. Includes an `importable()` probe
    helper.
  - `room4d/perception/instance/instance_artifact.py`: `InstanceArtifact`
    (+ `InstanceObservation`) — the shared mask + identity source of the
    reordered band: per-frame instance masks + persistent `track_id` + concept/
    kind routing hint, with npz-mask persistence + index (mirrors
    `ObjectArtifact`) and the capability manifest under `metadata['capability']`.
  - `room4d/perception/reid/reid_artifact.py`: `ReidArtifact` — per-tracklet
    appearance embeddings + a cross-occlusion `merge_map` with cycle-safe
    `canonical()` / `merged_groups()` and npz-embedding persistence.
  - Tests: `tests/test_backend_ladder.py`, `tests/test_perception_artifacts.py`
    (15 tests — ladder resolve/skip/fail-through/graceful-empty/no-rung, manifest
    round-trip, `importable` probe, artifact save/load incl. masks/embeddings/
    indices/manifest/merge-canonicalization).

### Changed
- Design doc §7: FP0 is purely additive (ladder base + new artifact types); the
  Object/Human/Track contract re-cut moves to FP1 where those stages are rewired.

## [0.1.17.1] - 2026-06-26

### Changed
- **Front-half perception redesign — design + build plan formalized** in the
  full-pipeline design doc (`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`).
  Decision-only doc change; no code yet.
  - **§4A (new section)** — replaces the hand-tuned 2D associator front-half with
    a modern foundation-model stack, verified by two adversarial deep-research
    passes: **SAM 3** (detect+segment+track / identity) → **DINOv3-L** (cross-gap
    re-ID) → **SigLIP 2 + Qwen2.5-VL** (open-vocab semantics) → **MoGe-2** (metric
    depth backstop) + **VGGT** → **Sapiens + GVHMR** (humans); **CoTracker3** kept
    as the orthogonal motion subsystem. Core principle: **decouple identity from
    motion** (the old design derived identity from point-track Jaccard). Every
    stage is an **artifact contract** filled by a **fallback ladder** (foundation
    model → lighter model → today's backend → graceful-empty) whose floor is the
    current pipeline, so a missing/ungated model degrades fidelity, never
    completeness. Runs **staged** (load→infer→release) within 32 GB.
  - **Stage map (§4)** — perception band reorders to
    `pose_refine → segment → reid → objects → humans → tracking → graph`; two
    net-new stages (`segment`, `reid`) and artifacts (`InstanceArtifact`,
    `ReidArtifact`); `objects`/`humans`/`tracking` re-cut.
  - **Build plan (§7)** — adds **FP0–FP6** (floor-first structure, then climb the
    ladder ungated-first), resumes **F0b** dynamic render on FP1 floor-rung masks,
    and subsumes the old **F3 pose2d de-stub** into FP5.
  - §0/§3/§8/§9 reconciled; `a private note` memory records
    the verified stack + licenses/gating.

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

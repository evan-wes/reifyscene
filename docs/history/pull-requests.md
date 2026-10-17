# Pull-request history

reifyscene's history holds the work of two earlier GitHub repositories, whose pull requests can no
longer be read there:

- **room4d**, pull requests 1–51 (2026-04-11 to 2026-07-09): the project before it was renamed reifyscene.
- **reifyscene, first repository**, pull requests 1–10 (2026-09-10 to 2026-10-17): this repository before its
  history was rewritten to go public, on 2026-10-17.

Their merge commits and release tags name them that way ("Merge room4d pull request 46 …", "PR:
reifyscene pull request 6, first repository"). This page keeps each one's title, branch, merge
date, size, merge commit and description. Pull requests in the current repository start again at
1 and live on GitHub.

Descriptions are as written at the time, except that paths on the author's machine and links to
private notes and sessions are removed, and headings are shifted to fit this page. Numbers such
as "#12" inside a description refer to that same repository's pull requests, and commit IDs quoted
in a description are from that repository's history at the time, so many no longer exist here.

## room4d

### <a id="room4d-pr-1"></a>room4d pull request 1: v0.1.0

Merged 2026-04-11 · `develop` → `main` · 34 commits, +15,216 −339 in 78 files · merge commit `e74948e`

Release v0.1.0

### Summary

This PR significantly expands `room4d` from an early prototype into a more structured, debuggable, and reusable 4D reconstruction pipeline.

The main improvements in this PR are:

- robust stage serialization and resume support across the full pipeline
- substantial VGGT / VGGT-Long geometry backend improvements for long-sequence execution
- richer rendering infrastructure, including layered Gaussian rendering and an offline high-quality mode
- world-frame / ground-plane canonicalization into a real Z-up coordinate system
- much stronger debug export plumbing across stages
- improved human grounding via SMPL-root anchoring, ground contact regularization, and temporal smoothing
- evaluation metrics and failure reporting for tracking, geometry, and rendering

---

### Major features added

#### 1. Full stage save/load + resume support

Added proper artifact serialization and resume support across all stages, including earlier stages that previously did not serialize cleanly.

##### Highlights
- added generic artifact save/load support using base artifact serialization
- added a stage registry with:
  - stage name
  - dependencies
  - load function
  - run function
- added `--prior-run-dir`
- added `--start-stage`
- copied reused stage outputs from prior runs into the new run directory so every run remains self-contained
- added compatibility checks to verify that reused stages match the current invocation
- made ingest, sync, graph, scene, and other stages properly serializable

##### Result
We can now:
- run expensive geometry once
- resume from tracking / humans / objects / optimization / scene / render
- keep every run directory reusable as a future baseline

---

#### 2. Geometry backend improvements (VGGT / VGGT-Long)

Added a number of practical improvements to make long-sequence geometry runs more reliable and easier to debug.

##### Highlights
- added `geometry.frame_source` semantics for:
  - `keyframes`
  - `tracking`
  - `all`
- added dense camera interpolation so tracking frames without direct geometry can still receive camera estimates
- added VGGT-Long auto-tuning:
  - tries aggressive chunk/overlap first
  - backs off chunk size and then resize on OOM
  - records attempt history
- added clearer auto-tune logging and failure summaries
- improved cleanup between attempts to reduce RAM / swap growth
- added VRAM-aware resize ladders
- patched weight loading to load checkpoints on CPU first, then move to GPU
- patched mixed-precision support while preserving FP32 where required for camera/extrinsic math
- fixed BF16 inverse failure in the camera branch
- prevented the third-party loader path from silently resizing smaller inputs back up in a way that defeated the external resize ladder

##### Result
The geometry path is now more robust, more inspectable, and significantly easier to run on constrained hardware.

---

#### 3. More intuitive geometry / tracking frame sampling controls

Added easier CLI control over which frames are used for geometry and tracking.

##### Highlights
- added more intuitive CLI arguments for:
  - geometry frame stride
  - geometry frame spacing in seconds
  - tracking frame spacing in seconds
- geometry can now be subsampled from its selected source timeline
- tracking can now be sampled by time spacing as well as stride

##### Result
It is now much easier to experiment with sparse vs dense processing without manually mapping everything onto raw frame indices.

---

#### 4. Render system refactor + layered Gaussian rendering

Refactored rendering so it is no longer a single hardcoded path and introduced a reusable render-scene contract.

##### Highlights
- added render backend registry
- added reusable render interfaces / base contracts
- added layered render contract stored in `SceneArtifact`
- static render layer is now seeded from:
  - geometry outputs
  - static lifted / classified tracks
- dynamic layer is now constructed from:
  - humans
  - objects
- added support for:
  - `gaussian_layered`
  - `gaussian4d`
  - `offline_high_quality`

##### Result
Rendering is now properly structured for future backends such as NeRF refinement / offline refinement, without rewriting scene assembly again.

---

#### 5. Offline high-quality render mode

Added an offline render mode that is substantially better than the earlier point/blob-style rendering.

##### Highlights
- moved render config under `configs/render/render.yaml`
- added offline high-quality render backend and config block
- added debug triptych output:
  - static only
  - dynamic only
  - full static + dynamic
- offline renderer now uses:
  - supersampled rendering
  - static multiview appearance reprojection from saved RGB + geometry sources
  - Gaussian fill for holes
  - textured dynamic sprite / crop rendering for humans and objects
  - offline refinement / NeRF-style image refinement hook

##### Result
Offline renders are much closer to a generated video than earlier simple Gaussian blobs and splats.

---

#### 6. World-frame canonicalization / ground-plane support

Added ground-plane estimation and canonical world-frame alignment so the scene can be transformed into a meaningful Z-up global coordinate system.

##### Highlights
- added ground-plane detection from geometry point maps
- built world transform so:
  - floor is near `z = 0`
  - `+Z` is vertical
- applied the transform consistently to:
  - point maps
  - camera poses
- added camera-up priors to reduce the risk of picking walls as the floor
- added diagnostics and inspectability:
  - saved plane-fit diagnostics
  - candidate plane scores
  - overlay images with estimated horizon and +Z direction
  - optional manual gravity override / lock
- fixed debug viewer basis handling so already-canonicalized geometry is not rotated a second time

##### Result
The pipeline now has a real global frame, which is essential for downstream reasoning, rendering, and future multi-camera calibration work.

---

#### 7. Human / object debug export plumbing

Fixed and expanded debug export behavior so the debug config actually controls HTML viewer generation across stages.

##### Highlights
- added proper plumbing of debug viewer flags through the runtime config
- ensured the end-to-end single-video runner performs deferred debug export generation
- updated debug config defaults to include sections for all stages, even if some stages do not yet emit custom visualizations
- enabled humans / objects debug viewer export through config, instead of hardcoded disabling

##### Result
Debug config is now much more reliable and stage-consistent.

---

#### 8. Human grounding with SMPL-root anchoring

Improved human world placement substantially by using SMPL root/pelvis as the primary anchor instead of relying only on unstable geometry-region centers.

##### Highlights
- human world placement now prefers:
  1. SMPL pelvis/root anchor lifted through geometry
  2. support-track geometry center
  3. region-based geometry fallback
- added local lifting near:
  - pelvis/root
  - ankle / foot anchors
- added ground-contact regularization using the detected ground plane
- added temporal smoothing of human 3D anchors across frames
- stored anchor provenance for debugging and downstream use

##### Result
Human 3D anchors are more stable, more semantically meaningful, and better aligned to the actual body than raw region-based lifting alone.

---

#### 9. Dynamic appearance + occlusion improvements

Improved dynamic rendering so it behaves more like an actual scene render and less like independent floating blobs.

##### Highlights
- added depth testing of dynamic elements against static background
- added dynamic-vs-dynamic ordering
- added soft alpha / boundary smoothing
- added temporal consistency in dynamic observation selection
- upgraded dynamic appearance bank:
  - multiple crops over time
  - view-aware selection
  - temporal smoothing / switching penalties
  - better mask-aware observation scoring

##### Result
Dynamic humans and objects are rendered with much better visual coherence and placement.

---

#### 10. Tracking / perception / render QA improvements

Added stronger debugging and evaluation signals to understand where the pipeline fails.

##### Highlights
- projected lifted 3D human centers back into image space for inspection
- added render-entity summary tying humans / objects to render entities
- added evaluation metrics and failure reporting for:
  - lifted-track reprojection error
  - human continuity / fragmentation
  - ground-plane consistency
  - render-vs-input comparison on held-out observed views
- added saved evaluation summary JSON and HTML report

##### Result
Pipeline quality can now be debugged more systematically instead of only by visual inspection.

---

### Config / CLI / repo structure changes

#### Config changes
- moved render config to:
  - `configs/render/render.yaml`
- added richer render config blocks for:
  - offline high-quality mode
  - dynamic appearance selection
  - occlusion controls
  - refinement settings
- added ground-plane / world-frame configuration
- added evaluation thresholds / settings
- expanded `configs/debug_exports.yaml` to cover all stages

#### CLI changes
- added:
  - `--prior-run-dir`
  - `--start-stage`
  - more intuitive geometry/tracking frame sampling args

---

### Testing / validation

A substantial number of focused regression tests were added or updated across:
- geometry
- scene/render
- resume/save-load
- debug export plumbing
- evaluation metrics
- tracking backend behavior

The pipeline was also exercised end-to-end through repeated real runs, including:
- geometry execution on all frames
- stage resume from prior runs
- render-stage debug output
- offline render path
- world-frame canonicalization
- human grounding improvements

---

### Notes / current limitations

This PR significantly improves the pipeline, but a few components are still prototype-level:
- offline high-quality mode is not yet a full trained NeRF backend
- VGGT-Long on lower-memory GPUs still depends heavily on tuning / runtime constraints
- world-frame estimation is much more inspectable now, but still benefits from scene-specific validation
- human 3D grounding is improved via SMPL-root anchoring, but further refinement is still desirable for highly articulated motion

---

### Why this PR matters

This PR moves `room4d` from a fragile experimental pipeline toward a reusable platform for:
- long-sequence geometry extraction
- resumable multi-stage reconstruction workflows
- richer debug and evaluation tooling
- improved dynamic scene rendering
- semantically grounded human placement
- future multi-camera calibration and high-quality offline refinement

### <a id="room4d-pr-2"></a>room4d pull request 2: Add SMPL modeling in tracking and rendering for humans

Merged 2026-04-11 · `develop` → `main` · 1 commit, +392 −0 in 4 files · merge commit `bdc889c`

Release v0.1.0 - add non-committed SMPL modeling

### <a id="room4d-pr-3"></a>room4d pull request 3: v0.1.1.0 feat: Shape of Motion reference comparison tool

Merged 2026-06-11 · `feature/reference-comparison-tool` → `develop` · 9 commits, +2,700 −1 in 16 files · merge commit `66b8392`

#### Summary

Adds the **reference comparison tool**: room4d can now be benchmarked against Shape of Motion (ICCV 2025) on the public DyCheck iPhone dataset, on this repo's Blackwell (RTX 5090) workstation.

**Reference environment (Blackwell port)**
- `scripts/setup_reference_tool.sh` builds the isolated `room4d_som_ref` conda env: torch 2.11.0+cu128 (sm_120), CUDA 12.8 toolkit, g++13 (CUDA caps host compiler <14), RAPIDS/cuML bumped 24.6→25.02 for sm_120, gsplat built from source at a pinned immutable SHA, Shape of Motion vendored at pinned commit `579753e` with reproducible, sentinel-verified compat patches (torch≥2.11 device strictness; gsplat≥1.4 radii shape). Safety guards refuse to target the `room4d`/`base` envs; smoke tests prove torch matmul, cuML KMeans, and a real gsplat rasterization on the GPU.

**Benchmark CLI + comparison contract**
- `scripts/run_reference_benchmark.py`: sequence in → train (completed runs reused via schedule-recording sentinel; interrupted runs resume; `--retrain` clears state) → eval → normalized `reference_benchmark_v1` JSON (masked mPSNR/mSSIM/mLPIPS + 2D/3D tracking metrics, epochs, env versions, `+dirty`-aware provenance). `--compare a.json b.json` scores any two results side-by-side with per-metric better-direction.
- `scripts/verify_reference_data.sh` validates a downloaded sequence against the layout SoM actually reads (training AND eval inputs) before GPU time is spent; parity with the Python validator is enforced by a shared-fixture test.

**Reference baseline recorded** (DyCheck `paper-windmill`, 500 epochs, 69 min): masked **mPSNR 19.53 / mSSIM 0.559 / mLPIPS 0.198**, 3D-track PCK@10cm 0.996 — in the published SoM regime (iPhone-avg 16.72/0.63/0.45), validating the port as faithful.

**Env reproducibility**
- Full pip-layer lockfiles for BOTH project envs (`scripts/locks/`, generator `scripts/gen_env_locks.sh`); both setup scripts install from their lock by default (`--unlocked` to re-resolve). Locks exclude editables and gsplat (owned by the setup scripts' commit pins).

**Review hardening** (two fix rounds from the in-flow review army + adversarial pass; details below)

#### Test Coverage

Coverage audit (subagent): **26/26 (100%) of unit-testable paths covered**; 26/46 (~57%) counting GPU/conda/network integration paths, which are annotated `[→E2E]` (validated manually via the real paper-windmill train+eval on the 5090). Coverage gate: overridden at 57% with that context — the residue is integration scope by nature, not skipped unit tests.

Tests: 76 → 104 (+28 new) — metric parsing (incl. scientific notation), compare direction/tie/NaN/Infinity/unknown-key handling, schedule-aware completion/resume/retrain semantics, contract-JSON happy path, validator parity (bash vs Python on one fixture), installer guard exits, `+dirty` provenance, sentinel-only-on-success.

#### Pre-Landing Review

In-flow review army (4 specialists + red team + always-on Claude adversarial; Codex unavailable): **25 findings total, all addressed across two fix commits** (`9fbd5da`, `e963d06`). Highlights:
- [CRITICAL][red-team] checkpoint-reuse trusted interrupted runs (SoM writes `last.ckpt` at epoch 0) → completion now proven by schedule-recording sentinel or final-epoch outputs
- [CRITICAL][red-team] `--retrain` silently resumed (trained ~0 epochs) → clears checkpoints/results/videos first
- [HIGH][adversarial] `pip freeze` emits VCS installs as `gsplat @ git+...` so the lock generator's `^gsplat==` exclusion leaked gsplat into the lock at `main@4e52698` — the SHA the env actually runs (same `__version__` as v1.5.3, so a version-equality guard could never correct it). Exclusion fixed for both freeze forms, pin corrected to the proven SHA, reinstall guard now compares the `direct_url.json` VCS SHA.
- [MULTI-SPECIALIST] sed patches could misfire silently on a pin bump → hard errors + per-edit sentinel counts; checkout restores tracked files so stale patches never survive
- Validator checked a layout SoM doesn't require while missing files it reads unconditionally (`extra.json`, `splits/val.json`, eval's `scale.npy`/`covisible/`/`keypoint/`) → aligned with the pinned source, both validators parity-tested

Scope Check: CLEAN — intent (reference comparison tool) matches delivery; no unrelated files touched.

#### Design Review

No frontend files changed — design review skipped.

#### Eval Results

No prompt-related files changed — evals skipped.

#### Plan Completion

Track 1 (Reference Comparison Tool) of the design doc: **10/13 implemented (9 done + 1 changed), 1 deferred by user decision, 2 deferred as P1 TODOs.**
- CHANGED: "align with `eval/rendering_metrics`" inverted to schema-as-contract — room4d's renderer eval adopts `reference_benchmark_v1` when the real renderer lands (P2 TODO)
- DEFERRED (user call): MoSca as second reference (`--with-mosca` intentionally errors; backlog)
- DEFERRED → P1 TODOs: NVIDIA Dynamic Scenes download + multi-view wrapper support; calibration-vs-GT extrinsics scoring

#### Verification Results

Skipped — plan has no web-verification section and no dev server (CLI tooling; the pipeline was verified by the real 500-epoch train+eval run).

#### TODOS

TODOS.md created. 3 items added (2× P1 deferred-from-plan, 1× P2 emitter contract); none completed by this PR.

#### Release-tag automation (added after initial review)

Standard workflows now ride this PR: `tag-on-version-bump.yml` (annotated `v<version>` from develop, merge-subject PR resolution, idempotent, `workflow_dispatch` backfill) and `tag-on-release.yml` (`release-v<version>` + GitHub Release from main). Canonical version moved to a **`VERSION` file** with `pyproject.toml` reading it via setuptools dynamic metadata (verified: `importlib.metadata` reports 0.1.1.0; gstack version CLI reads it natively; full suite re-passed 103/104). **On merge, the develop push carries both the workflow and the bump, so `v0.1.1.0` tags itself.**

#### Test plan
- [x] Full pytest suite passes fresh after final fix batch: **103 passed, 1 skipped** (re-verified after workflow/VERSION wiring)
- [x] Bash/python validators accept the real `paper-windmill` download; converged real run recognized as complete (no spurious retrain)
- [x] gsplat reinstall guard verified against the live env (installed SHA == pin)
- [x] End-to-end pipeline validated on hardware: setup → verify → 500-epoch train → eval → `reference_benchmark.json` (mPSNR 19.53)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-4"></a>room4d pull request 4: v0.1.1.1 feat: gsplat source build in the main room4d env (Track 2 milestone-0)

Merged 2026-06-11 · `feature/gsplat-milestone-0` → `develop` · 1 commit, +150 −6 in 3 files · merge commit `769c7b2`

#### Summary

Track 2 milestone-0 from the design doc: prove gsplat builds and runs in the main `room4d` env on CUDA 13, ahead of replacing the placeholder `GaussianRasterizer` (step 7, separate PR).

- **`--with-gsplat`** in `scripts/setup_room4d.sh` (opt-in, like other heavy source builds): builds gsplat from source for sm_120 at the same Blackwell-proven pin as the reference env (`4e52698e`), with the SHA-verified idempotency guard ported from `setup_reference_tool.sh` and a post-install smoke test that rasterizes a trivial scene on the GPU (an import-only check would pass a broken build).
- **Milestone finding №1:** the pin builds cleanly against CUDA 13.0 (torch 2.11+cu130, nvcc 13.0) — the one open variable, since the reference env had only proven it on CUDA 12.8. No patches needed; wheel `gsplat-1.5.3` built and the sm_120 kernels execute.
- **Milestone finding №2:** the gsplat wheel leaks top-level `tests` and `experimental` packages into site-packages (upstream packaging bug). The leaked `tests` shadowed the repo's own `tests` package and broke pytest collection. The installer now removes every top-level package gsplat's RECORD claims outside its own namespace, on both the build and skip paths.
- `jaxtyping` + `ninja` installed explicitly (gsplat's only declared deps not in the locked tree); gsplat itself stays `--no-deps` so its torch/numpy declarations cannot fight the lock. Env passes `pip check`.
- CUDA-toolkit/compiler helper messages genericized (they were written as VGGT-Long-only; they now serve any CUDA extension build).
- Version `0.1.1.0` → `0.1.1.1` (MICRO: setup-script plumbing, no new room4d module).

#### Test plan

- [x] Full setup run with `--with-gsplat` in the `room4d` env: build succeeds, all smoke tests pass including GPU rasterization (output `(1, 128, 128, 3)`)
- [x] Idempotent re-run: SHA guard skips the rebuild, leak cleanup removes `tests`/`experimental`
- [x] `pip check` clean
- [x] Repo test suite: 103 passed, 1 skipped (collection was broken by the leaked `tests` package before the cleanup fix)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-5"></a>room4d pull request 5: v0.1.2.0 feat: native-gsplat render module, no fake gaussians anywhere (Track 2 step 7)

Merged 2026-06-11 · `feature/gsplat-rasterizer` → `develop` · 1 commit, +569 −733 in 17 files · merge commit `7cd88f1`

#### Summary

Full replacement of the render module per design-doc Track 2 step 7: the renderer is natively gsplat, with **no fake NumPy gaussians anywhere** — while preserving the debug viewers and the artifact contracts.

**Deleted (733 lines of fakes):**
- The per-Gaussian NumPy splatting loop and hand-rolled projection (`_splat_gaussians`, `_project_points*`)
- The sprite-billboard compositor, observation-bank selection, and all `dynamic_sprite_*` knobs
- Per-source static re-splatting (`_render_static_appearance`)
- The blur/sharpen filter masquerading as `nerf_refinement`
- `OfflineHighQualityTrainer` and the `quality_mode` split (no fake quality left to toggle)

**New core:**
- `GaussianRasterizer`: real `gsplat.rasterization()` wrapper — eye/target/up → OpenCV viewmat, with the legacy framing intrinsics (fx = 0.75·max(w,h), centered principal point) preserved so existing camera paths keep their composition
- `Gaussian4DScene` + `GaussianCloud`: the scene contract materialized as device-resident torch tensors
- `Gaussian4DTrainer`: slim orchestrator (924 → ~400 lines) rendering real static-only / dynamic-only / full Gaussian layers per frame

**Preserved exactly:**
- `RenderArtifact` fields, JSON save/load, `summary()`
- `novel_views/` + `novel_views_debug/` layout, triptych frames, both HTML players, entity labels
- `_render_frame_bundle(scene_artifact, camera)` signature (consumed by `eval/rendering_metrics`)
- Registry names: `gaussian_layered` and `offline_high_quality` stay as aliases of the one real renderer
- The assembler and its `layered_scene_v2` contract are untouched

**Hard requirements:** rendering needs gsplat + CUDA — no silent CPU fallback; the error points at `scripts/setup_room4d.sh`, where gsplat now builds **by default** (`--no-gsplat` to opt out).

**Known interim regression (by design):** dynamic entities render as their template Gaussians (colored blobs) instead of photographic sprite billboards, until trained Gaussians + motion bases land in steps 9–10.

#### Test plan

- [x] New `tests/test_gsplat_rasterizer.py`: viewmat/projection/intrinsics sanity (CPU), centered-blob coordinate check — **renderer checkpoint 0** — plus depth values, empty-cloud handling, concat, and depth-ordered occlusion (GPU)
- [x] `test_scene_render.py` reworked: gsplat/CUDA skip guards, legacy-alias coverage, dynamic-entities-as-template-gaussians test asserting the dynamic debug layer is non-empty
- [x] Full suite on the 5090: **111 passed, 1 skipped** — all render tests executed on GPU (no skips triggered)
- [x] No dangling references to deleted symbols (remaining `offline_high_quality` strings are contract metadata the assembler emits and the intentional registry alias)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-6"></a>room4d pull request 6: v0.1.3.0 feat: NVIDIA Dynamic Scenes dataset backend for the reference benchmark

Merged 2026-06-12 · `feature/dataset-backends` → `develop` · 3 commits, +1,032 −82 in 7 files · merge commit `67b9454`

#### Summary

**NVIDIA Dynamic Scenes dataset backend** (`b36004d`)
- `scripts/run_reference_benchmark.py --dataset {iphone,nvidia}`: per-dataset layout validation, the SoM `data:nvidia` training subcommand, and a new eval path — SoM ships no `evaluate_nvidia.py`, so nvidia metrics are harvested from the trainer's final-epoch `val/*` TensorBoard scalars. They are covisibility-masked novel-view metrics computed with the same `flow3d.metrics` classes as the iphone eval, so they map onto the existing `masked_mpsnr`/`masked_mssim`/`masked_mlpips` schema keys, with `fg_`/`bg_` splits as extras.
- Result JSONs record the eval protocol (`eval` block — nvidia trains/evaluates at the dataloader's hardcoded 2x, iphone at 1x) and `--compare` warns when two results' protocols differ.
- **Proven on GPU:** 5-epoch smoke run on Jumping trains via `data:nvidia`, renders 288 held-out views at final-epoch validation, and emits the full contract JSON (masked mPSNR 18.69, all nine metrics). Completed-run reuse re-harvests in seconds without retraining.

**Review hardening** (same commit, from this ship's specialist/red-team/adversarial passes)
- Reuse provenance guard: pointing a completed run dir at the wrong `--data-dir` aborts instead of silently relabeling old metrics as the new sequence (verified live against the real smoke-run dir).
- `--retrain` clears TensorBoard event files, stale `cfg.yaml`, and SoM's per-invocation `code/` backups — a stale event file would otherwise outrank the retrained run's scalars (global_step restarts at 0).
- Harvest rejects mixed-run event files (val scalars disagreeing on final step) and skips non-finite scalars rather than recording NaN/Inf baselines.
- Deep download verification in BOTH validators: non-empty val split (an empty one trains for hours, then yields no metrics) and the complete N² pairwise track set (the dataloader loads random pairs per batch; a partial download crashes mid-training). Verified against the real Balloon1 (36,864 pairs) and Jumping (9,216 pairs) downloads.
- Corrected the interrupted-run guard's false claim that SoM resumes a "saved schedule" — verified that SoM never re-reads `cfg.yaml`; resume trains toward the CLI/default 500-epoch schedule. Proper resume handling filed as P1 TODO.
- `--num-epochs < 1` rejected up front.

#### Test Coverage

Coverage audit (subagent, mid-ship): 23/26 paths tested (88%) with 3 named gaps — all 3 closed during the ship (fg/bg compare directions, summary-print assertions, `--dataset` through real argv). The adversarial round added 14 more tests. `tests/test_reference_benchmark.py`: 22 → 50 tests. Full suite: **133 passed, 1 skipped**.

<details><summary>Coverage diagram (audit snapshot, pre-gap-closure)</summary>

```
scripts/run_reference_benchmark.py
├── validate_sequence (iphone/nvidia, accept + reject listing)   ★★★ tested
├── DATASET_LAYOUTS / NVIDIA_REQUIRED_{DIRS,FILES}               ★★★ tested (bash parity)
├── NVIDIA_TB_METRICS tag→key mapping (9 tags)                   ★★★ tested
├── clear_run_state events.out.tfevents.* removal                ★★★ tested
├── run_training data:{iphone,nvidia} construction               ★★★ tested
├── harvest_nvidia_metrics happy/warn/error paths                ★★★ tested
├── benchmark() dispatch (nvidia→harvest, iphone→evaluate)       ★★★ tested
└── argparse --dataset                                           [GAP→closed]
scripts/verify_reference_data.sh nvidia branch                   ★★★ tested
CLI flows (reuse, --retrain, resume guards, --compare)           ✓ tested
COVERAGE at audit: 23/26 (88%) → gaps closed in-ship
```
</details>

#### Pre-Landing Review

17 main-report findings (1 critical, 16 informational) across checklist pass + testing/maintainability specialists + red team + post-fix adversarial pass. 15 fixed in-branch, 2 deferred by user decision as TODOs (P1 proper resume handling, P2 run-dir lockfile — the harvest's mixed-run step detector covers the detection half). One cosmetic nit skipped (terminal-escape passthrough in `--compare` output of a local tool). PR Quality Score: 9.0/10 post-fix.

#### Design Review

No frontend files changed — design review skipped.

#### Eval Results

No prompt-related files changed — evals skipped.

#### Scope Drift

Scope Check: CLEAN — intent (TODOS.md P1: NVIDIA download + `data:nvidia` wrapper support) matches delivery; review hardening stayed within the new code's blast radius.

#### Plan Completion

No plan-mode file (scope agreed in-session). Conversation-scoped plan: download ✓ (manual, both scenes verified), dataset flag + validators + harvest ✓, verify-script update ✓, tests ✓, GPU proof ✓ — all DONE.

#### TODOS

- Completed: "Download NVIDIA Dynamic Scenes dataset + extend benchmark wrapper" → v0.1.3.0
- Added: P1 proper resume handling in the wrapper (SoM never re-reads cfg.yaml); P2 run-dir lockfile (+ data-dir cache race noted)
- Unblocked: "Score hloc+COLMAP calibration against dataset GT extrinsics" (NVIDIA dependency satisfied)

#### Documentation

- **README.md (Data):** added the nvidia validator example and documented the deep download verification — full directory layout, non-empty val split, and the complete N² pairwise track set (a partial download would otherwise crash mid-training).
- **README.md (Benchmark):** documented `--dataset {iphone,nvidia}` with an NVIDIA example; the per-dataset metric source (`evaluate_iphone.py` for iphone vs final-epoch TensorBoard `val/*` harvest with `fg_`/`bg_` splits for nvidia, since SoM ships no `evaluate_nvidia.py`); the result JSON's `eval` protocol block (nvidia at the dataloader's hardcoded 2x, iphone at 1x); the run-dir provenance guard (wrong `--data-dir` aborts); corrected resume semantics (resume targets SoM's default 500-epoch schedule); and `--retrain` clearing all run state including TensorBoard event files.
- **README.md (Compare):** noted that `--compare` warns on sequence or eval-protocol mismatch instead of silently declaring winners on incomparable numbers.
- **CHANGELOG.md:** reviewed only — v0.1.3.0 entries preserved verbatim.

##### Documentation Debt
- ⚠️ Benchmark workflow — reference + how-to coverage exists in README but no tutorial-quadrant walkthrough; `/document-generate` can fill it.
- ⚠️ CHANGELOG.md — not linked from README.md (discoverability, advisory).

#### Test plan
- [x] Full pytest suite passes (133 passed, 1 skipped) — run after every fix round
- [x] Real-GPU smoke: 5-epoch `--dataset nvidia` run on Jumping → contract JSON with all 9 metrics; reuse path re-harvests identically
- [x] Both validators pass against the real Balloon1/Jumping downloads (incl. N² pairwise check); content corruptions (empty val split, partial pairwise set) correctly rejected
- [x] TensorBoard harvest verified against a real event file written by the reference env's SummaryWriter (final-epoch max-step selection confirmed)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-7"></a>room4d pull request 7: v0.1.3.1 fix: benchmark resume semantics + run-dir flock

Merged 2026-06-12 · `fix/benchmark-resume-and-lock` → `develop` · 4 commits, +599 −95 in 6 files · merge commit `dc3e14a`

#### Summary

Knocks out both P-items filed by PR #6's ship review, in one PR (`1976af0` + docs `59a16fd`/`28440e4`).

**Resume semantics (was P1).** SoM writes `cfg.yaml` but never re-reads it — resume trains `range(ckpt_epoch, CLI num_epochs)` with the CLI default of 500. The wrapper now:
- recalls the recorded schedule from `cfg.yaml` and passes it back explicitly (a plain resume of an interrupted 1000-epoch run resumes toward 1000, not 500); fails **closed** when no schedule is recorded;
- reads the checkpoint's epoch via the reference env (`weights_only=True` only — SoM checkpoints are plain tensor dicts, so the pickle fallback was dropped as code-exec surface; 120s timeout) and aborts a resume with zero epochs left instead of stamping mid-training metrics as a completed baseline;
- canonicalizes `--num-epochs 500` and plain runs to one `default` label (no more spurious schedule-mismatch aborts), writes the completion sentinel atomically, and records `resumed`/`resumed_from_epoch` in the result JSON so a segment-only `train_minutes` can't masquerade as full training cost.

**Run-dir locking (was P2).** Each invocation holds an exclusive `fcntl.flock` on `.room4d_benchmark.lock` for its lifetime. Kernel-owned: releases on any death (`kill -9` included), no PID-liveness or reclaim protocol to race, and the training child inherits the fd via `pass_fds` so an orphaned trainer keeps the run dir locked until it exits. The lock file persists on disk; contents are diagnostics only. The first-cut O_EXCL+PID-reclaim design was **rejected during this PR's own adversarial review** — two independent passes found TOCTOU reclaim races (two waiters could both acquire) and the orphaned-trainer hole — and replaced with flock.

**GPU-verified:** real resume (sentinel stripped, cfg bumped 2→4) trained the remaining epochs and stamped `4`; a held lock refuses a second invocation with PID diagnostics and proceeds once released; `weights_only=True` reads real SoM checkpoints (epoch 0 on a short run — checkpoints only refresh every 200 global steps, an empirical find).

#### Test Coverage

Audit at 74% (23/31 paths) with 8 named gaps — all but one closed in-ship (the remaining one, lock read/unlink race paths, became moot when the reclaim protocol was deleted). `tests/test_reference_benchmark.py`: 50 → 59 tests. Full suite: **142 passed, 1 skipped**.

#### Pre-Landing Review

Specialists (testing, maintainability) + coverage audit + Red Team + Claude adversarial. The adversarial round drove a real design change: 4 CRITICAL findings against the hand-rolled lock (unlink-reclaim TOCTOU, empty-content steal window, wrapper-vs-trainer PID tracking, PID-reuse false-held) — all eliminated structurally by the flock redesign rather than patched. Also fixed from review: label split-brain, fail-open schedule assumption, non-atomic sentinel, pickle fallback, missing subprocess timeout, resume provenance, stale docstrings/comments, misleading reclaim message. Codex unavailable (not installed).

#### Scope Drift

Scope Check: CLEAN — exactly the two TODO items + review hardening of the code they introduce.

#### Plan Completion

No plan-mode file (scope agreed in-session). Both TODO items: DONE, GPU-verified.

#### TODOS

- Completed: "Proper resume handling" (P1) and "Lockfile guard" (P2) → v0.1.3.1
- Added: residual same-data-dir cache race split into its own P3 entry (the run-dir lock doesn't cover SoM's `flow3d_preprocessed/cache/` writes into the *data* dir)

#### Documentation

- **README.md** — current; the "Run handling" paragraph was verified claim-by-claim against the final implementation (schedule recall, zero-epoch abort, fail-closed, flock semantics, child fd inheritance, persistent lock file).
- **CHANGELOG.md** — v0.1.3.1 entry; stale header claim fixed ("versioned in pyproject.toml" → canonical `VERSION` file); doc-review caught my overcount of new tests (15 → 9, corrected).
- **TODOS.md** — groomed as above.

#### Test plan
- [x] Full pytest suite passes (142 passed, 1 skipped), re-run after every fix round
- [x] Real-GPU resume: cfg-bump 2→4 resumed and trained the remaining epochs, sentinel "4"
- [x] Live lock contention: held flock refuses a second invocation; released flock proceeds to reuse+harvest
- [x] `weights_only=True` reads a real SoM checkpoint after dropping the pickle fallback

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-8"></a>room4d pull request 8: v0.1.4.0 feat: calibration GT scoring (calibration arc M0-minimal)

Merged 2026-06-12 · `feature/calibration-gt-scoring` → `develop` · 3 commits, +1,894 −17 in 10 files · merge commit `72f1b04`

#### Summary

Calibration arc **M0-minimal**: the GT-scoring instrument, per the eng-reviewed design (D2–D12, design doc in a private folder; scope committed by decision D10 — the S1 kyotovision spike and all hardening are gated work, deliberately NOT in this PR). Commits `58842b4` (feature) + `9422e97` (README).

**The instrument** (`room4d/calibration/gt_scoring.py` + `scripts/score_calibration_gt.py`):
- GT adapters for DyCheck/NVIDIA camera jsons with schema guards (incl. rejecting non-rotations and reflections); frame collector emits a `gt_manifest.json` correspondence contract at collection time — the scorer joins blindly, correspondence can't drift (D3). Re-collections purge stale images, since hloc sweeps every file in the image dir.
- **Pairwise relative errors are the PRIMARY metric** (D11): per-pair rotation geodesic, dual-anchor baseline direction, median-normalized baseline ratio — all gauge-free, no alignment step to absorb error at small N. Same-position pairs (static camera at two timesteps) are split out as registration *repeatability* so they cannot flatter the headline numbers (ship-review catch: they diluted the original Balloon1 mean by ~6%, and would dominate on sparse configs). Estimate-collapsed pairs are flagged, never silently dropped.
- Sim(3)-APE on centers is SECONDARY, with conditioning guards on **both** GT and estimated centers (D6 math is oracle-pinned against evo v1.36.5, with 0°/180° clamps and reflection handling).
- Fragmentation is a result, not an error (D5): largest alignable component scored, coverage + per-component sizes reported, console summary shows scored scope.
- `--fix-intrinsics` = the known-intrinsics protocol (D4): shared GT intrinsics injected via the runner's new `camera_params` passthrough, refinement disabled; refuses per-camera intrinsics (D12 — hloc applies image options globally; DyCheck's per-frame ARKit focal drift correctly triggers the refusal in a real-data test). **The artifact proves the protocol held:** recovered intrinsics + effective camera mode recorded, `intrinsics_match_injected` verified.

**Compatibility collateral** (forced by the first real runs; ruled justified in-scope by the plan audit): three idempotent hloc patches in `setup_room4d.sh` for pycolmap ≥ 3.10 API drift, vendored-tree restore before ref checkout (next `HLOC_TAG` bump would otherwise abort on the dirtied tree), a setup smoke test of the exact patched option fields, and copy-not-symlink collection (pycolmap ≥ 3.13 resolves symlinks on import, desyncing names from hloc feature keys).

**First real numbers for the calibration stage, ever:**
| run | cross-camera pairwise rot (mean/max) | sim3 trans mean | registration |
|---|---|---|---|
| Balloon1, free | 0.176° / 0.392° | 0.0021 GT-u | 4/4 cams, 20/20 imgs, 1 component |
| Balloon1, fixed-gt (verified) | 0.136° / 0.376° | 0.0018 GT-u | 4/4, 20/20, 1 component |
| paper-windmill vs ARKit, free | 1.89° / 3.27° | 0.0100 GT-u | 3/3 cams, 24/24, 1 component |

Known-intrinsics improves everything, as the protocol predicts — and the Sim(3) rotation numbers running ~10× worse than pairwise at N=4 empirically validates the D11 pairwise-primary decision on the very first run.

#### Test Coverage
Audit: 74% (50/68 paths) at review time; the gap wave closed every non-integration finding — verified crash fixed (`center_conditioning` small-N IndexError), exact injection-payload assertion via the extracted `build_sfm_cfg`, schema-guard/manifest/clamp/CLI-exit tests, and a Shepperd rewrite of the test quaternion helper after the new 180° case exposed a threshold bug in it. `tests/test_gt_scoring.py`: **33 tests**. Full suite: **175 passed, 1 skipped**.

#### Pre-Landing Review
Coverage audit + plan-completion audit + testing/maintainability specialists + combined red-team/adversarial pass. Substantive review catches now in the code: headline-metric dilution by same-camera pairs (metric integrity for the future S1 kill gate), explicit-and-recorded `SINGLE` camera mode (was a silent PER_FOLDER default), artifact-level injection verification, stale-image purge, median-based degeneracy thresholds, dual-anchor direction errors, est-side conditioning. Deferred with rationale: GT-convention sanity check (lands with the Panoptic adapter), images.txt parser shape guard (post-spike hardening), model-dir ranking (recorded in artifact instead).

#### Plan Completion
19/22 DONE, 2 CHANGED (extraneous-images as result field; windmill proof initially missing — **then run for real**, so effectively closed), 1 correctly deferred (D8 hardening, gated on S1). Scope drift: CLEAN — setup-script patches ruled justified in-scope collateral.

#### TODOS
M0-minimal + S1 entry already groomed on this branch (built / spike open); post-spike hardening entry gated on the S1 verdict; stale cross-reference fixed.

#### Documentation
- **README.md**: new "Calibration GT scoring" section — CLI usage both modes, env/setup notes, artifact contract (`calibration_gt_score_v1`), metric design, measured numbers.
- **CHANGELOG.md**: v0.1.4.0 entry (reviewed 3/3 on the sell test). **TODOS.md/VERSION**: groomed/bumped.
- Debt flagged: library-API docs are docstring-only (CLI is the supported surface); CHANGELOG not linked from README.

#### Test plan
- [x] Full suite 175 passed / 1 skipped, re-run after every fix round
- [x] Three real hloc+COLMAP scored runs (Balloon1 free + fixed-gt with injection verified, paper-windmill) with the final metrics
- [x] evo-oracle alignment fixtures; real-data adapter tests against Balloon1/paper-windmill
- [x] Setup-script syntax checked; hloc patches idempotent with loud anchor-miss failure

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-9"></a>room4d pull request 9: Renderer Track 2 PR 1: dataset geometry backend + Rasterizer ABC + Checkpoint 0

Merged 2026-06-15 · `feature/render-dataset-geometry` → `develop` · 2 commits, +859 −28 in 13 files · merge commit `36273f8`

#### Renderer Track 2 — PR 1: dataset geometry backend + Rasterizer ABC + Checkpoint 0

First increment of carrying the renderer to completion (design-doc Track 2, step 8 + Checkpoint 0). Stands up borrowed dataset geometry flowing through room4d's real contracts into a **pluggable** rasterizer, validated by rendering at real cameras. No training yet — that's PR 2.

##### What's here
- **`Rasterizer` ABC + `GsplatRasterizer` (first backend) + `build_rasterizer` factory.** gsplat is no longer welded into the renderer — it sits behind an enforced contract, mirroring the `GeometryInitializer`/`PointTracker` backend pattern, so the splatting kernel is swappable. A new `RenderCamera` (`from_look_at` / `from_world_cam`) is the backend-agnostic camera, carrying a real world-to-camera matrix + intrinsics (the old rasterizer synthesized intrinsics and only took procedural look-at cameras).
- **`DatasetGeometryInitializer`** — reads DyCheck-format poses/intrinsics/depth from disk → standard `GeometryArtifact`, bypassing on-the-fly SfM/VGGT. One backend handles NVIDIA Balloon1 **and** DyCheck paper-windmill. Convention matches Shape-of-Motion (`w2c = [R | -R p]`, `T_world_cam = inv(w2c)`). Selectable via `--geometry-backend-name dataset`.
- **`seed_cloud_from_geometry`** — dense depth backprojection → colored `GaussianCloud` (the untrained seed).
- **`scripts/run_render_proof.py`** — Checkpoint 0 harness: dataset → seed → render at real cameras → GT-vs-render panels.

##### Checkpoint 0 — coordinate sanity: PASS
On Balloon1, seeding **only** from camera 0's depth:
- **Reconstruction** (cam0@t0): render ≈ GT (intrinsics + depth + backprojection correct).
- **Held-out novel views** (cam1/2/3@t0): recognizable and correctly placed from poses that contributed no points — the extrinsic / `w2c→T_world_cam` convention is right (a wrong convention would mirror/misalign/empty these). Coverage ~1.0; PSNR 18–19 dB (expected for an *untrained* point-cloud splat; PR 2's trainer lifts this).

##### Validation
- Full suite: **182 passed, 1 skipped** (incl. 4 GPU rasterizer tests now exercising the `RenderCamera` path).
- New CPU tests: dataset pose-convention, factor-scaled intrinsics, `GeometryArtifact` construction, seed backprojection math; `from_world_cam` convention guard; `build_rasterizer` factory.

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-10"></a>room4d pull request 10: Renderer Track 2 PR 2: static Gaussian trainer + Checkpoints 1 & 2

Merged 2026-06-15 · `feature/static-gaussian-trainer` → `develop` · 2 commits, +1,175 −108 in 12 files · merge commit `b9eb444`

#### Renderer Track 2 — PR 2: static Gaussian trainer + Checkpoints 1 & 2

Turns the PR-1 depth-backprojection seed into a **trained** static Gaussian field. The renderer now *optimizes* a scene, not just materializes one. Builds on the same plug-and-play discipline: the trainer and densification depend only on the `Rasterizer` contract — gsplat is never imported outside the backend.

##### What's here
- **`Rasterizer` ABC extended** — `rasterize()` (differentiable) returns grad-bearing rgb/alpha/depth + densification telemetry (`means2d`/`radii`) as a `RasterOutput`; `render()` is now a no-grad numpy wrapper over it (one gsplat call site).
- **`GaussianField(nn.Module)`** — trainable scene, standard 3DGS parameterization (log-scales, logit-opacities, sigmoid RGB; SH later), seeded via `from_cloud`.
- **`DensificationController`** — *our* clone/split/prune/opacity-reset with Adam-state surgery, operating on the contract telemetry (backend-agnostic). Thresholds ported from gsplat's validated `DefaultStrategy`; `max_gaussians` safety cap. The NDC `means2d` gradient is rescaled to pixel units so the ported `grad2d_threshold` applies correctly.
- **`StaticGaussianTrainer`** — Adam (per-group LRs, means-LR exp decay), L1 + D-SSIM loss (self-contained torch SSIM, no new dep), densification schedule.
- **Config-driven** — `configs/render/render_proof.yaml` + `..._windmill.yaml` expose **every** knob; `run_render_proof.py` gains a `train` mode (the CLI overrides only per-run paths).

##### Results
Validated on the 5090. The static checkpoints run on **paper-windmill** (a moving monocular camera = the proper static-training regime; Balloon1's train camera is a single static rig view, kept for Checkpoint 0).
- **Checkpoint 1 (train-view):** static structure (wall/pot/plant) reconstructs **sharply**; the spinning windmill is the dominant diff error.
- **Checkpoint 2 (held-out cam 1):** static scene renders **coherently from a camera that contributed no training gradient**.
- Train/novel PSNR plateaus at ~22 / ~20 dB because a *static* model cannot fit the scene's motion — the dynamic branch (PR 4) lifts it; masked-mPSNR-vs-SoM scoring is PR 3.

##### Validation
Full suite **189 passed, 1 skipped** (+7 new CPU tests: field activations, SSIM/PSNR, and the densification Adam-state surgery — clone/split/prune/cap — each asserting the optimizer still steps on the resized params). Checkpoint 0 unaffected.

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-11"></a>room4d pull request 11: Renderer Track 2 PR 3: covisibility-masked NVS benchmark + eval reorganization

Merged 2026-06-15 · `feature/render-eval-and-nvs-benchmark` → `develop` · 2 commits, +740 −101 in 28 files · merge commit `bbc81b4`

#### Renderer Track 2 — PR 3: covisibility-masked NVS benchmark + eval reorganization

Turns Checkpoint 2 into a real **vs-SoM** comparison: room4d's renderer now emits the DyCheck co-visibility-masked novel-view-synthesis metrics and `--compare`s against the Shape-of-Motion baseline on identical sequences. Also reorganizes `eval/` into a structure that scales as the pipeline's stub stages get fleshed out.

##### Eval reorganization (all stages)
- **Per-stage evaluators co-located** → `room4d/<stage>/eval/` (render, geometry, tracking, sync, graph). `git mv` preserved history; no behavior change.
- **Shared metric primitives** → `room4d/eval/metrics/image.py`: `MaskedPSNR`/`MaskedSSIM`/`MaskedLPIPS`, **ported verbatim from SoM's `flow3d/metrics.py`** (the DyCheck protocol) so numbers are byte-comparable. torch + torchmetrics only.
- **Benchmarks isolated** → `room4d/eval/benchmarks/covisibility_masked_novel_view_synthesis/`: `schema.py` (the `reference_benchmark_v1` contract — now the single source of truth), `compare.py`, and a full **`README.md`**. `scripts/run_reference_benchmark.py` now **imports** the contract instead of defining it.
- `room4d/eval/` keeps only the cross-cutting layer (shared `metrics/`, `benchmarks/`, the `benchmark_runner` orchestrator).

##### Feature
- `room4d/render/eval/novel_view.py` — renders a trained Gaussian field at held-out cameras, accumulates the masked metrics.
- `run_render_proof.py train` emits a `reference_benchmark_v1` record over the val split and `--compare-to` a SoM baseline; `DatasetGeometryInitializer.load_covisible(...)` added. New `benchmark:` config section.

##### Result (room4d static vs SoM dynamic, paper-windmill, all 419 val frames)
| metric | SoM | room4d | better |
|---|---|---|---|
| masked_mpsnr ↑ | 19.53 | 18.51 | SoM |
| masked_mssim ↑ | 0.559 | 0.448 | SoM |
| masked_mlpips ↓ | 0.198 | 0.234 | SoM |

A static model lands ~1 dB behind on masked mPSNR (the masked region is mostly static background); the SSIM/LPIPS gaps come from un-reconstructable motion — the gap PR 4 (dynamic branch) closes.

##### Validation
Full suite **196 passed, 1 skipped** (+7 CPU tests: masked metrics, schema build/load/validate, compare). `--compare` verified on the existing SoM JSONs.

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-12"></a>room4d pull request 12: Renderer Track 2 PR 4: dynamic Gaussian branch + Checkpoint 3

Merged 2026-06-16 · `feature/dynamic-gaussian-branch` → `develop` · 2 commits, +896 −2 in 14 files · merge commit `f838b95`

#### Renderer Track 2 — PR 4: dynamic Gaussian branch (Checkpoint 3)

The renderer now handles **motion**. Ports the Shape-of-Motion SE(3) motion model onto room4d's contracts: foreground Gaussians carry softmax motion coefficients over K low-dimensional SE(3) bases and are warped per frame, then composited with the static background and rasterized through the existing `Rasterizer` contract.

##### What's here
- **Deformation core (our code, ported recipe):** `transforms.py` (6D-rotation + **gradient-stable** quaternion helpers, replacing `roma`), `motion.py` (`MotionBases`), `dynamic_field.py` (`DynamicGaussianField` warp + `DynamicScene.cloud_at` + the per-Gaussian **`entity_id`** channel: fg = 1, bg = 0).
- **Dynamic trainer:** `dynamic_trainer.py` — samples `(camera, time, gt)`, renders the warped scene, optimizes photometric L1 + D-SSIM + temporal motion-smoothness. `run_render_proof.py dynamic` mode + a dynamic config; `score_masked_dynamic_views` in the render eval.
- **Borrowed init (stopgap seam):** `scripts/borrow_som_motion_init.py` dumps SoM's track-based init (cuML clustering + Procrustes, 37.5k fg + 100k bg + 10 bases × 277 frames); `motion_init.py` loads it into a `DynamicScene`. SoM's param storage matched ours 1:1.

##### Checkpoint 3 — PASS
On paper-windmill (all 419 val frames, covisibility-masked):

| metric | SoM | room4d-dynamic | static (PR3) |
|---|---|---|---|
| masked_mpsnr ↑ | 19.53 | **20.71** | 18.51 |
| masked_mssim ↑ | 0.559 | **0.661** | — |
| masked_mlpips ↓ | 0.198 | **0.178** | — |

The dynamic branch lifts masked mPSNR **+2.2 dB over the static baseline at the same factor 2** (apples-to-apples), and beats SoM on all three masked metrics (SoM scored at factor 1 — a noted resolution caveat). Dynamic train-view PSNR is **25.4 dB** (vs the static model's motion-capped ~22) — the moving windmill is reconstructed, not ghosted.

##### Caveats → tracked follow-ups (in TODOS)
- The motion **init is borrowed** from SoM (stopgap); a native room4d `MotionInitializer` is follow-up work behind the existing seam.
- **No dynamic densification** yet (v1 trains the fixed-size borrowed init).

##### Validation
Full suite **203 passed, 1 skipped** (+7 CPU dynamic tests: transform round-trips, warps, scene + entity_id, bundle loader).

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-13"></a>room4d pull request 13: Renderer Track 2 PR 5: CLI flythrough.mp4 (completes the track)

Merged 2026-06-16 · `feature/cli-flythrough` → `develop` · 2 commits, +190 −3 in 8 files · merge commit `3ac17e6`

#### Renderer Track 2 — PR 5: CLI flythrough (completes the track)

The final piece. Renders the trained dynamic scene along a hand-authored camera path while time advances, and encodes `flythrough.mp4` — the free-viewpoint video the whole track was building toward.

##### What's here
- **`render/camera_paths.py::orbit_cameras`** — a standalone near-trajectory orbit arc (center/radius/up derived from the input cameras, so the path stays *near the input trajectory* per the design-doc stability criterion, not orbiting behind the scene where monocular 4D collapses). **`replay`** (the input cameras) is a config alternative.
- **`render/export.py::write_video`** — RGB frames → mp4 via OpenCV (`mp4v`; imageio isn't in the env).
- **`run_render_proof.py dynamic`** emits `flythrough.mp4` after Checkpoint 3, with time advancing across the path so the windmill spins as the camera arcs. New `flythrough:` config section.

##### Validated
Produced a real **120-frame, 360×480, 24fps** orbit flythrough of paper-windmill — the scene (wall, pot, plant, spinning windmill) renders coherently from novel viewpoints. Full suite **206 passed, 1 skipped** (+3 CPU tests: orbit geometry, single-frame, mp4 encoding).

##### Track status
This finishes design-doc **steps 8–11**:
`dataset geometry backend → static Gaussian trainer → covisibility-masked NVS benchmark (vs SoM) → dynamic SE(3) branch → free-viewpoint flythrough`.

Step 12 (swap in room4d's own VGGT/CoTracker, one backend at a time) is the next phase, plus the two tracked follow-ups (own motion init, dynamic densification).

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-14"></a>room4d pull request 14: Dynamic densification (PR 6) — standard 3DGS density control on fg + bg

Merged 2026-06-16 · `feature/dynamic-densification` → `develop` · 2 commits, +187 −37 in 9 files · merge commit `f52d913`

#### Dynamic densification (PR 6)

Closes the deferred PR-4 follow-up: standard 3DGS adaptive density control (clone / split / prune / opacity-reset) now runs on the dynamic **foreground + background** fields — the same recipe and thresholds already ported for the static trainer in PR 2, **no new algorithm**.

##### What changed
- **`DensificationController` generalized** — `param_names` + `group_prefix` (so the fg field grows its extra `motion_coefs` param, and fg/bg share one optimizer with `fg.*`/`bg.*` groups), per-field **telemetry slicing** of the combined cloud's `means2d`/`radii`, and a generalized clone/split where split children inherit every non-positional param (incl. `motion_coefs`) from the parent — exactly how SoM / deformable-3DGS densify.
- **`DynamicGaussianTrainer`** runs one controller per field, each fed its slice of the combined-cloud telemetry after `backward()`.
- New `densify:` section in the dynamic config.

##### Effect (paper-windmill, Gaussians grow ~137k → ~970k)
| metric | SoM | dynamic (no densify) | dynamic + densify |
|---|---|---|---|
| masked_mpsnr ↑ | 19.53 | 20.71 | 20.66 |
| masked_mssim ↑ | 0.559 | 0.661 | **0.669** |
| masked_mlpips ↓ | 0.198 | 0.178 | **0.164** |
| train-view PSNR | — | ~25 | **27.2** |

The gain is in perceptual sharpness (mLPIPS) and structure (mSSIM) plus a visibly sharper flythrough — where densification helps most. Masked mPSNR is flat because that metric is dominated by the already-reconstructed static background. Still beats SoM on all three.

##### Validation
Full suite **207 passed, 1 skipped** (+1 CPU test: the fg controller grows `motion_coefs` + Adam state in lockstep and the optimizer still steps). The static densification path is unchanged (default `param_names`/`group_prefix`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-15"></a>room4d pull request 15: Native 3D-track lifting (PR-A of native motion init) — v0.1.11.0

Merged 2026-06-16 · `feature/native-track-lift` → `develop` · 1 commit, +885 −4 in 8 files · merge commit `dd6fd22`

#### What

PR-A of replacing the borrowed SoM motion init with room4d's own. This lands the **3D-track lifting bridge**: room4d's native port of Shape-of-Motion's track lift, so the only thing still borrowed on this path is the 2D point tracker output (`flow3d_preprocessed/2d_tracks/`).

`room4d/lifting/native_track_lift.py` ports, onto room4d's `DatasetGeometryInitializer`:
- TAPIR track parsing + `normalize_coords`
- SoM scene normalization (gravity-up → +z, center on fg tracks, scale to ~unit)
- foreground-track unprojection (`get_tracks_3d`) and background point sampling (`get_bkgd_points`)
- depth→normal helper, Rodrigues/`rt_to_mat4` (no roma)
- a `build_native_init_inputs` orchestrator returning `(Tracks3D, StaticPoints3D, cameras, scene_norm)` in the normalized frame

Two SoM-specific sources were matched so the output lands in the **same frame** as the borrowed bundle:
- **aligned inverse-depth** (`aligned_depth_anything_colmap` + per-stack clamp), not the sparse sensor `depth/`
- **COLMAP-refined cameras** (`camera_type="refined"` via pycolmap; ARKit `"original"` poses kept as an option)

Also: `room4d/lifting/tracks_3d.py` (`Tracks3D` / `StaticPoints3D` containers), `DatasetGeometryInitializer.load_fg_mask`, and `scripts/check_native_lift.py`.

#### Validation (paper-windmill, vs SoM's cached `tracks_3d_40000.pth` / `scene_norm_dict.pth`)

| metric | native | SoM cache | diff |
|---|---|---|---|
| scene-norm rotation `max|Δ|` | — | — | **3.9e-8** |
| scene scale | 0.1093 | 0.1089 | 0.4% |
| foreground tracks | 39 500 | 39 525 | 0.1% |

All checks PASS — a faithful reproduction of the borrowed frame. Full suite: **215 passed, 1 skipped** (incl. 8 new dataset-free unit tests for the ported math).

#### Version
VERSION → `0.1.11.0` (PATCH: new module), CHANGELOG + TODOS updated.

#### Next
PR-B — clustering + Procrustes + `NativeMotionInitializer` consuming these `Tracks3D` / `StaticPoints3D`, validated against the borrowed-init quality bar (masked mPSNR ≈ 20.7).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-16"></a>room4d pull request 16: Native motion initializer + densify schedule fix — v0.1.12.0

Merged 2026-06-17 · `feature/native-motion-init` → `develop` · 1 commit, +838 −32 in 11 files · merge commit `40d88d9`

#### What

**PR-B of the native motion init** — room4d now builds the dynamic init natively, end to end. The only borrowed input left is the 2D point tracker output.

- **`room4d/render/gaussian4d/procrustes_init.py`** (new) — ports SoM's `flow3d/init_utils.py`: `solve_procrustes` (SVD weighted SE(3), 6D), `get_weights_for_procrustes`, kNN, a numpy port of the cupy masked-track interpolation, `sample_initial_bases_centers` (**sklearn KMeans** on velocity directions, replacing cuML), `init_motion_params_with_procrustes`, `init_fg_from_tracks_3d`, `init_bg` (roma-free).
- **`NativeMotionInitializer`** in `motion_init.py` — chains PR-A's lift → `cano_t` → clustering/Procrustes → fg/bg placement into the same `(DynamicScene, {train/val: [BorrowedCamera]})` shape as `load_borrowed_init`.
- **`run_render_proof.py dynamic --native`** builds the init natively (no SoM env, no cuML); **`scripts/native_motion_init.py`** dumps the native bundle to `.npz` (parity with the borrow).

#### Also: fixes a pre-existing densification collapse

While validating, the full dynamic run collapsed to **masked mPSNR ~6.3 for *both* native and borrowed inits**. Root cause: the schedule constants are SoM's (`reset_opacity_every: 3000`, `stop_iter: 4000`) but the config ran **`iters: 3000`**, so the opacity reset landed on the *final* iteration — every Gaussian clamped to ~1% opacity with no recovery training, so eval rendered a near-black scene. SoM never hits this because it runs `max_steps: 5000` (reset at 3000 + ~2000 recovery + polish).

Fix — derive a SoM-shaped schedule from the iteration budget so the reset always recovers:
- `densify.py::densify_schedule(max_iters, num_frames)` sets `start/reset/stop` as fractions of `iters` (0.04 / 0.60 / 0.80) → `reset < stop < iters` always holds. Reproduces SoM's exact `200/100/3000/4000` at `iters=5000`.
- `DensifyConfig.from_cfg(..., max_iters, num_frames)` fills any schedule field a config omits, and **raises if `stop_iter >= iters`**.
- The opacity reset is guarded to the densify window (as SoM's control loop does).
- Dynamic config default is now `iters: 5000` (SoM's `max_steps`); its schedule fields auto-derive. Static path untouched (sets schedule explicitly; no bug there).

#### Validation (paper-windmill, iters=5000, full SoM schedule)

| init | masked mPSNR | mSSIM | mLPIPS | final N |
|---|---|---|---|---|
| **native** | **20.74** | 0.670 | 0.157 | 955k |
| **borrowed** | **20.68** | 0.668 | 0.160 | 954k |

Native matches borrowed at the quality bar; the opacity reset fires at 3000 and recovers; N ≈ 955k matches the v0.1.10.0 CHANGELOG's "~970k" (confirming the original 20.7 was measured at the full 5000-iter schedule). Full suite: **232 passed, 1 skipped**.

#### Tests
- `tests/test_procrustes_init.py` — `solve_procrustes` recovers a known SE(3) (+6D decode), kNN, masked interpolation, KMeans on synthetic clusters, `init_fg`/`init_bg` shape/log/logit/quat sanity.
- `tests/test_densify_schedule.py` — reproduces SoM at 5000, the `reset < stop < iters` invariant across budgets, explicit-overrides-derived, the `stop_iter >= iters` guard, backward-compat without `max_iters`.

#### Version
VERSION → `0.1.12.0` (PATCH). CHANGELOG + TODOS updated (native-init follow-up marked DONE; own-2D-tracker filed as the last borrowed input).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-17"></a>room4d pull request 17: Add native 2D point tracker (closes last borrowed init input) — v0.1.13.0

Merged 2026-06-18 · `feature/native-2d-tracker` → `develop` · 1 commit, +2,531 −20 in 19 files · merge commit `74bf339`

#### What

Replaces the borrowed `flow3d_preprocessed/2d_tracks` with room4d's **own** point tracker — the last borrowed input in the native motion-init path. Wired as a plug-and-play `PointTracker` backend alongside the CoTracker wrapper. A faithful port of Shape-of-Motion's `preproc/compute_tracks_torch.py`.

After this, the only dataset-provided inputs left in the native init are RGB/depth/poses — i.e. dataset ground truth, not borrowed reconstruction.

#### Changes

- **`room4d/tracking/tapnet/`** — DeepMind TAPIR torch model vendored **byte-for-byte** (sha-verified, re-diffable against upstream, Apache-2.0 headers kept, ruff-excluded). `einshape` added to the `vision` extra + `room4d.lock` to keep the port byte-faithful. `scripts/fetch_tapir_checkpoint.sh` fetches the bootstapir/tapir checkpoints (idempotent, size-guarded) to `data/tapnet/checkpoints/`.
- **`TAPIRWrapper`** (`room4d/tracking/tapir_wrapper.py`) — a real `PointTracker` backend (`model_type` bootstapir|tapir): per-frame foreground-masked query grid (`grid_size` 4, SoM's value), 256×256 model input, all-pairs tracking to `(x, y, occ_logit, dist_logit)`.
- **Contract extension (additive)** — `TrackArtifact` gains a first-class all-pairs payload (`room4d/tracking/all_pairs.py::AllPairsTracks`) with SoM `2d_tracks/{i}_{j}.npy` save/load + `write_2d_tracks`. The single-query CoTracker/scaffold path is unaffected.
- **Wiring** — `lift_tracks_3d` / `build_native_init_inputs` / `NativeMotionInitializer` / `build_native_init` gain a `tracks_dir` override; `generate_native_tracks` bridges train frames+masks to the tracker. `run_render_proof.py dynamic --own-tracker [--own-tracks-dir DIR]` and `scripts/compute_tracks_native.py` run the init with no borrowed tracks.

#### Validation (paper-windmill, factor 2)

| Init | masked mPSNR | mSSIM | mLPIPS |
|---|---|---|---|
| Borrowed init | 20.68 | — | — |
| Native init (borrowed tracks, v0.1.12.0) | 20.74 | 0.670 | 0.157 |
| **Native init + own tracker (this PR)** | **20.70** | **0.668** | **0.161** |

- FG seeding reproduces the shipped tracks **exactly** (1739/1739 diagonal query coords).
- Full test suite: **243 passed, 1 skipped**. New: `tests/test_tapir_wrapper.py` (all-pairs format/round-trip, contract extension, checkpoint resolution, GPU-gated forward).

#### Caveats (carried honestly)

- Per-track agreement with the shipped tracks is ~1–2px on nearby frames but loosens over long temporal gaps (~30px at 100-frame separation) — expected for two independent tracker runs on a fast-rotating scene (also JAX-vs-torch bootstapir + cv2-vs-mediapy resize). The end-to-end mPSNR is the bar that proves track quality.
- One intentional deviation from the reference: frame resize uses cv2 `INTER_AREA` instead of mediapy Lanczos (documented in the wrapper).
- The lock regen also captured pre-existing gsplat dep drift (`jaxtyping`/`ninja`/`wadler_lindig`) — makes the lock more faithful, but it's unrelated churn.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-18"></a>room4d pull request 18: Geometry-provider seam (step 12, PR 2) — v0.1.14.0

Merged 2026-06-18 · `feature/step12-geometry-provider` → `develop` · 1 commit, +309 −54 in 7 files · merge commit `4ce189e`

#### What

First step of **step 12** (replace borrowed DyCheck geometry with room4d's own VGGT reconstruction). Introduces a swappable geometry source behind the native motion-init path, with the borrowed dataset GT as the permanent oracle/baseline. **Lossless refactor** — no behavior change yet; this is the seam that PR 3 (VGGT) plugs into.

#### Changes
- **`room4d/lifting/geometry_provider.py`** — `GeometryProvider` ABC (`cameras(split)` + `frame_stack(split)`) + `DyCheckGeometryProvider`, which delegates to the existing `_load_cameras` / `_load_frame_stack` unchanged.
- **`build_native_init_inputs` / `NativeMotionInitializer` / `build_native_init`** gain an optional `provider` (defaults to a `DyCheckGeometryProvider` built from `dataset_root`/`factor`/`depth_type`/`camera_type`). Geometry source is now swappable; tracks stay separate from geometry.

#### Validation
- **Lossless (the key proof):** the DyCheck provider is **byte-identical** to the old direct path on paper-windmill — 277-frame train `FrameStack` (depths/rgbs/fg/valid) + 419 val cameras all equal.
- **E2e through the provider:** masked mPSNR **20.68** / mSSIM 0.670 / mLPIPS 0.158 (419 val frames) — matches the native baseline (20.70–20.74) within GPU training run-to-run noise.
- Full suite **247 passed, 1 skipped** (`tests/test_geometry_provider.py` added: ABC contract, config/backend wiring, lossless delegation, `provider` override).

#### De-risking spike (throwaway, outside the repo)
Before committing to the design, measured the VGGT-vs-GT geometry gap on paper-windmill: VGGT poses land within **~0.5° relative rotation** of COLMAP (median pairwise rot 0.45°, sim3 abs rot 1.98°, one global up-to-scale `s≈0.83`); depth AbsRel **~0.02** with a consistent global scale. → **step 12 is integration + a scale-alignment hook, not a quality-closing research effort.** (Caveat: one textured clip; depth GT is itself a monocular prior, so the pose evidence — vs real COLMAP SfM — is the stronger signal.)

#### Next (step 12)
- PR 3: `VGGTGeometryProvider` + global scale-alignment hook; swap **poses** first, score vs the oracle.
- PR 4: swap **depth**. Masks stay borrowed; native masks + `Room4DPipeline` convergence are deferred.

#### Note on the diff
`TODOS.md` is large because it **rides two things**: the calibration **S1-spike-DONE** update (verdict CONTINUE-qualified, D15 "person complements SfM" reframe; post-spike entry reframed to M2/M3 joint-opt fusion) requested earlier, **plus** the new step-12 progress entry. Per the repo's "TODOS edits ride with a PR" convention.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-19"></a>room4d pull request 19: Add --stop-stage flag to single-video runner (stage windows) — v0.1.15.0

Merged 2026-06-18 · `feature/stage-stop-flag` → `develop` · 1 commit, +99 −5 in 5 files · merge commit `13d5261`

#### Summary

`run_single_video_room4d.py` was **resume-forward only**: `--start-stage` picks where to begin, then the loop runs straight through to `render`. This adds the missing half — a stop point — so the runner can execute any contiguous **stage window** `[start, stop]`.

- New **`--stop-stage`** (inclusive), `choices=STAGE_ORDER`, default `render` → existing invocations unchanged.
- Window semantics: stages before `start` load from `--prior-run-dir` (unchanged); stages in `[start, stop]` run; stages after `stop` are skipped (`break`).
- Window resolution + validation extracted to a unit-tested helper **`room4d.api.resume.resolve_stage_window`** (rejects unknown stage names and stop-before-start with a clear error).

##### Why
Lets the **real pipeline** run e.g. `ingest → sync → geometry` and stop — no throwaway driver — which is exactly what the step-12 VGGT native-geometry de-risk needs (run VGGT-Long geometry on the full clip, skip tracking/humans/objects/optimization/render).

#### Fixed (latent bug the feature surfaced)
- The end-of-run summary loop did `outputs[name]` for **every** stage in `STAGE_ORDER` → `KeyError` once a run stops early. Now skips stages absent from `outputs`. (The frame-summary table already used defensive `.get()`.)
- Run manifest now records `stop_stage` alongside `start_stage`.

#### Tests
- New `tests/test_stage_window.py`: full-pipeline default, `ingest→geometry` window, single-stage window, mid-window, stop-before-start raises, unknown start/stop raise.
- Full suite: **254 passed, 1 skipped** (pre-existing).

#### Versioning
`### Added` (new CLI capability) → PATCH bump **0.1.14.0 → 0.1.15.0**. CHANGELOG updated. No `TODOS.md` change — this is runner infra, not a roadmap item.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-20"></a>room4d pull request 20: Install ffmpeg CLI in setup_room4d.sh for dataset prep — v0.1.15.1

Merged 2026-06-18 · `feature/setup-ffmpeg` → `develop` · 1 commit, +81 −1 in 3 files · merge commit `f1673b9`

#### Summary

Adds the **ffmpeg CLI** to `scripts/setup_room4d.sh` so it lives in the `room4d` conda env instead of being borrowed from another env by absolute path.

- conda-forge install, pinned **`FFMPEG_VERSION=8.1.1`**, **GPL build** (build string `gpl_*`) → includes **`libx264`**, required for `-crf 0` lossless H.264.
- **Default-on** with a **`--no-ffmpeg`** opt-out; skipped under `--use-venv` (with a note to install manually).
- Idempotent: skips if the pinned ffmpeg with libx264 is already present.
- Installed in the conda layer (alongside compilers/cuda-toolkit) via the existing `run_conda_install_relaxed` helper — **not** in the pip lockfile, since it's a system/conda package.
- **Post-install smoke test** asserts `libx264` is actually in `-encoders` (the non-GPL conda-forge variant omits it — a silent failure mode otherwise).

##### Why
ffmpeg is a **dataset-prep convenience**: turning image-sequence datasets (DyCheck, DAVIS) into the mp4s the pipeline ingests — needed by the step-12 VGGT geometry de-risk and similar experiments. room4d's own decode path uses OpenCV (`cv2.VideoCapture`) and does **not** depend on the ffmpeg binary, so this is purely workflow tooling.

#### Validation
- `bash -n` clean; `--help` shows `--no-ffmpeg`.
- Ran the new step into the live `room4d` env: installed `ffmpeg-8.1.1-gpl_h6d6c1bd_904`, smoke test confirms `libx264` + `libx264rgb` encoders present at `$CONDA_PREFIX/bin/ffmpeg`.

#### Versioning
Installer refinement = `### Changed` → **MICRO** bump **0.1.15.0 → 0.1.15.1**. CHANGELOG updated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-21"></a>room4d pull request 21: Fix VGGT-Long global pose alignment in geometry artifact — v0.1.15.2

Merged 2026-06-18 · `fix/vggt-long-global-pose-alignment` → `develop` · 1 commit, +185 −3 in 4 files · merge commit `f439912`

#### Summary

`VGGTLongInitializer` produced **globally inconsistent camera poses**. VGGT-Long estimates per-chunk Sim(3) transforms (overlap point-map registration + optional loop closure), accumulates them, and applies the result **only** when writing `camera_poses.txt`. The per-chunk `_tmp_results_aligned/*.npy` files keep **chunk-local** extrinsics (only `world_points` are globalized there). The wrapper merged those chunk-local extrinsics by first-occurrence, so an N-chunk sequence became **N independently-anchored mini-trajectories** — each chunk self-consistent but anchored at its own identity frame, with the trajectory jumping at every chunk boundary.

The wrapper now reads VGGT-Long's authoritative globally-aligned **`camera_poses.txt`** for the camera poses, with a graceful fallback to chunk-local extrinsics when the file is absent or its frame count mismatches. `world_points` (already globalized by VGGT-Long) and `depth` (camera-frame) are unchanged.

- `_load_global_camera_to_world(runner, n)` — parse `camera_poses.txt` (global camera-to-world) from the runner's output dir.
- `_combined_predictions_from_chunks(..., global_c2w=)` — override `extrinsic` with `inv(global_c2w)` (camera-from-world), so the shared `_initialize_from_predictions` / `_world_from_cam` path recovers a correct global `T_world_cam`. Matches plain VGGT's convention.
- Both the auto-tune and direct chunk paths load + pass the global poses; notes record global-alignment vs chunk-local fallback.

Plain VGGT (single forward pass, no chunking) was never affected.

#### Verification (paper-windmill, 277-frame train clip, vs COLMAP GT)

Stored `GeometryArtifact` camera poses, scored with `gt_scoring`:

| | pairwise rot | sim3 abs rot | baseline-dir |
|---|---|---|---|
| **Before (bug)** | 20.5° | 134° | ~98° |
| **After (this PR)** | **1.73°** | **10.2°** | **7.2°** |
| VGGT-Long native `camera_poses.txt` | 1.73° | 10.2° | 7.2° |

The fixed artifact poses now match VGGT-Long's native trajectory **exactly**. Confirmed by re-running the geometry stage end-to-end with the fixed code (`run_single_video_room4d.py --stop-stage geometry --geometry-backend-name vggt-long`).

#### Tests
- New `tests/test_vggt_long_global_alignment.py`: `camera_poses.txt` parse/validate (+ mismatch/missing → `None`), the global-pose override round-trip (`T_world_cam` recovers global c2w), and the legacy chunk-local fallback.
- Full suite: **257 passed, 1 skipped**.

#### How it was found
A throwaway de-risk spike scored the VGGT-Long geometry stage against the dataset ground truth at full 277-frame scale before wiring it into the render path. Depth was excellent (AbsRel 0.034) but global poses were 20° off; isolating native-vs-artifact poses surfaced this latent integration bug. (An earlier hypothesis — a c2w/w2c convention double-inversion — was tested and **rejected** by re-running on the real path; this PR is the verified root cause.)

#### Versioning
Correctness `### Fixed` → MICRO bump **0.1.15.1 → 0.1.15.2**. CHANGELOG updated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-22"></a>room4d pull request 22: Enable VGGT-Long loop closure in setup (faiss + dinov2 path patch) — v0.1.15.3

Merged 2026-06-19 · `feature/vggt-long-loop-closure-deps` → `develop` · 1 commit, +81 −2 in 3 files · merge commit `4eb1382`

#### Summary

VGGT-Long loop closure (`--vggt-long-loop-enable`) was **non-functional** in the room4d setup — doubly broken:

1. `LoopModels/backbones/dinov2.py` does `torch.hub.load('./LoopModels/dinov2', …)` — a path resolved against the **process CWD**, so it `FileNotFoundError`s on `./LoopModels/dinov2/hubconf.py` whenever VGGT-Long runs from any directory but its own (always, under room4d).
2. `faiss` (the loop detector's place-recognition retrieval) is **stripped** from VGGT-Long's pip requirements (`faiss-gpu` needs a CUDA build).

`scripts/setup_room4d.sh` now fixes both, in the `--with-vggt-long` block:

- **`ensure_faiss`** — installs the conda-forge **faiss-cpu** build (pinned `FAISS_VERSION=1.9.0`; `--use-venv`-guarded; skips if already importable). CPU is plenty for retrieval over a clip's frames and avoids a CUDA-version match.
- **DINOv2 path patch** — an idempotent patch anchoring the path to the module's own location via `__file__`, mirroring the existing hloc patch (unique anchor find/replace, skip-if-applied, `sys.exit` loudly if the upstream anchor drifts).

#### Why / verification

Found while testing whether loop closure could shrink VGGT-Long's ~1.3° pose gap (step-12 native-geometry ladder). With both fixes, loop closure runs **end-to-end** — verified on paper-windmill: **5 loop constraints found + Sim3 loop optimization**, geometry stage completes, `camera_poses.txt` written with the loop-optimized trajectory.

It's **opt-in / off by default**. On a forward-sweep clip it left the pose error unchanged (loop closure's lever is accumulated drift, not VGGT's per-frame pose noise), so it doesn't help *that* case — but it's now available for clips that genuinely revisit places.

#### Validation
- `bash -n` clean; `--help` clean.
- dinov2 patch verified **idempotent** against the already-patched tree ("already patched").
- faiss-cpu 1.9.0 installed + imports in the live env; full loop-closure run succeeded.

#### Versioning
Setup correctness fix → `### Fixed` → MICRO **0.1.15.2 → 0.1.15.3**. CHANGELOG updated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-23"></a>room4d pull request 23: Add pose_refine BA stage (Path B productionization) — v0.1.16.0

Merged 2026-06-20 · `feature/pose-refinement-stage` → `develop` · 1 commit, +897 −33 in 12 files · merge commit `5fad6d6`

#### Summary

Productionizes the validated **Path B** pose fix from the step-12 spike as a new, backend-agnostic **`pose_refine`** pipeline stage inserted between `geometry` and `tracking`, so every downstream stage consumes BA-refined camera poses.

It consumes the `GeometryArtifact` contract (poses + intrinsics + per-pixel world points) plus the source frames, runs the vendored VGGSfM tracker, seeds a `pycolmap` reconstruction with the backend's poses/K/world points and the visible tracks, and runs COLMAP's iterative robust BA (Cauchy loss + reprojection-outlier filtering, **refine shared focal, principal point fixed**). It re-emits a refined `GeometryArtifact` and re-derives the interpolated tracking-frame camera states from the refined keyframes. **No geometry backend is re-run.**

**Why BA closes the native-geometry pose gap** when photometric pose-opt could not: BA minimizes *reprojection* error of feature correspondences — a signal tied to the true scene independent of the current geometry — breaking the pose↔geometry gauge ambiguity.

#### Validated results (step-12 spike, Path B)
- paper-windmill pairwise rotation **1.73° → 0.26°**, masked mPSNR **17.01 → 19.58** (+2.57, ~69% of the gap to the GT-pose oracle 20.75)
- spin **4.0° → 1.08°**
- Robust BA rejects dynamic-foreground tracks on its own → **no foreground mask generated** (ablation: masking not load-bearing). Refining the principal point is harmful and stays off.

#### Behavior
- **Off by default** (`--pose-refine-enable`). When disabled or when the refinement runtime is unavailable, the stage is an **identity pass-through** that re-emits the geometry unchanged, so downstream has a stable dependency either way.
- Resumable via the existing stage window (`--start-stage pose_refine --prior-run-dir <cached geometry run>`) to re-tune BA without re-running geometry.
- Dependency resolution is by name, so the stage re-emits a plain `GeometryArtifact` (no subclass) and the six downstream stages are rewired to depend on `pose_refine`.

#### Changes
- **New** `room4d/geometry/pose_refine.py` — `PoseRefiner` (ports the spike machinery; generalized from the hard-coded 518² crop to the artifact's per-stream (H, W); heavy deps imported lazily). Includes a crop-consistency guard that degrades to pass-through rather than seeding BA with mismatched coordinates.
- `Room4DPipeline.refine_poses` + `run()` wiring
- `STAGE_ORDER` / `build_stage_registry`: `pose_refine` inserted (now **11 stages**), downstream rewired; `resume.py` volatile-keys
- Runner: 11-stage labels, release map, `--pose-refine-*` flags, debug print
- New `pose_refine` extra (`pycolmap`, `hydra-core`, `omegaconf`, `scipy`) installed by default in `setup_room4d.sh` (`--no-pose-refine` to skip) + importability smoke tests
- `VERSION` → **0.1.16.0**, CHANGELOG `### Added`

#### Testing
- New `tests/test_pose_refine.py` (config parsing, pass-through-when-disabled, fallback-when-unavailable, applies-refined-poses) + updated stage/resume tests → **15 passing**
- Vertical-slice smoke (full heuristic `run()` incl. the new pass-through) → **passing**
- All refiner heavy deps import in the `room4d` env
- The actual BA path (GPU + VGGSfM + pycolmap) is exercised by the spike, not unit tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-24"></a>room4d pull request 24: F0a: renderable core — self-contained renderable SceneArtifact + trainer-driven render backend (v0.1.17.0)

Merged 2026-06-26 · `feature/f0-renderable-core` → `develop` · 5 commits, +1,430 −2,371 in 16 files · merge commit `0acd067`

First phase of the full-pipeline architecture (design doc
`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`).
The render stage now **trains a real 3DGS scene and emits real novel views
in-pipeline**, replacing the no-train placeholder — with the design's clean
separation of concerns: the **scene stage** turns the reconstruction into a
**self-contained renderable scene**; the **render stage consumes only that
artifact**.

#### What changed
- **`SceneArtifact`** re-cut to a self-contained renderable contract: dense
  `render_seed` (npz) + per-frame `train_views` with the **GT frame persisted
  into the artifact** + a novel-view `render_trajectory` + `kind` + a
  `dynamic_bundle` placeholder (F0b). The toy `layered_scene_v2` /
  `static_scene` / `dynamic_entities` schema is gone.
- **`assembler.py`** re-cut (1046 → ~210 lines): dense seed
  (`seed_arrays_from_geometry`, factored numpy-only out of
  `seed_cloud_from_geometry`) + train views + orbit; sheds the toy
  sprite/template/entity machinery (entity semantics → P3 graph stage).
- **`GaussianRenderBackend`** (new): seed → `GaussianField` →
  `StaticGaussianTrainer.fit` → trajectory render + trained checkpoint. Degenerate
  scenes (no depth/GT) emit an empty render instead of crashing.
- **Masked-NVS metric** (new `render/eval/masked_nvs_benchmark.py`): scores the
  trained checkpoint vs a dataset's held-out val + covisibility (DyCheck
  protocol), reusing the proof scorer. Dataset-coupled; `{}` on arbitrary video.
- **Deleted the toy** (`scene_model.py` + `trainer.py`); rewired
  `registry.py` / `render/__init__.py` / `fit_renderer`.

#### Validation
In-pipeline masked mPSNR **18.54** on DyCheck paper-windmill (static; 277 train
frames, 7000 iters, 419 val views) **reproduces the standalone
`run_render_proof.py` static 18.51** — F0a static parity proven. Full test suite:
**260 passed, 1 skipped**.

#### Deferred to F0b
Static/dynamic *selection*, training-config home, and the `DynamicGaussianTrainer`
wiring (the +2.2 dB toward the 20.7 dynamic number). Resume is stage-level for now.

#### Reviews
Design eng-reviewed (`/plan-eng-review`) + independent outside voice; both folded
into the design doc.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-25"></a>room4d pull request 25: docs: front-half perception redesign (§4A) + FP0–FP6 build plan

Merged 2026-06-26 · `feature/perception-redesign` → `develop` · 1 commit, +230 −30 in 3 files · merge commit `e55b94c`

Formalizes the **front-half perception redesign** into the system-of-record design doc. **Design-doc only — no code yet.** VERSION `0.1.17.0` → `0.1.17.1`.

#### What & why
Replaces the hand-tuned 2D associator front-half (which derived **identity from CoTracker point-track Jaccard** — a motion cue doing an identity job) with a modern foundation-model stack, **verified by two adversarial deep-research passes**. Core principle: **decouple identity from motion**.

**Verified stack:** SAM 3 (detect+segment+track / identity) → DINOv3-L (cross-gap re-ID) → SigLIP 2 + Qwen2.5-VL (open-vocab semantics) → MoGe-2 (metric depth backstop) + VGGT → Sapiens + GVHMR (humans); CoTracker3 kept as the orthogonal motion subsystem. Template: 4D-LangSplat.

#### Degradability (the design's spine)
Every stage = a fixed **artifact contract** filled by a **fallback ladder** (foundation model → lighter model → today's backend → graceful-empty). The ladder **floor is today's pipeline**, so a missing/ungated model degrades *fidelity*, never *completeness* — guaranteed for the full graph/scene/render artifact set (§4A worked traces). Runs **staged** (load→infer→release) within 32 GB.

#### Doc changes
- **§4A (new):** verified stack, staged-execution constraint, reordered band, contract+ladder model, back-half guarantee, worked traces.
- **§4 stage map:** reorders to `pose_refine → segment → reid → objects → humans → tracking → graph`; net-new `segment`/`reid` stages + `InstanceArtifact`/`ReidArtifact`; `objects`/`humans`/`tracking` re-cut.
- **§7 build plan:** adds **FP0–FP6** (floor-first structure, then climb ungated-first); **F0b** dynamic render resumes on FP1 floor-rung masks; **F3** pose2d de-stub subsumed by FP5.
- §0/§3/§8/§9 reconciled; `perception-redesign-stack-verified` memory records the verified stack + licenses/gating.

#### Next
Scope + build **FP0** (backend-ladder base + new artifact schemas). SAM 3 / DINOv3 HF access requested in parallel (gated, manual approval); ungated tools cover FP0–FP1 + early climb.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-26"></a>room4d pull request 26: FP0: degradable backend ladder + InstanceArtifact/ReidArtifact

Merged 2026-06-27 · `feature/perception-redesign-fp0` → `develop` · 1 commit, +773 −5 in 10 files · merge commit `222208b`

**FP0** of the perception redesign (design §4A) — the **degradable-ladder foundation**. Purely **additive**: new files only, nothing existing imports them yet, so the suite is green by construction. VERSION `0.1.17.1` → `0.1.18.0`.

#### What
- **`room4d/common/backend_ladder.py`** — `BackendRung` / `BackendLadder` / `CapabilityManifest`. A ladder probes its rungs top-down (best → graceful-empty), runs the first available one, and records an **observable manifest**: which rung ran, at what fidelity, and why higher rungs were skipped (`not-installed` / `no-access` / `probe-error`). A resolved rung that raises **falls through** to the next. Generalizes the ad-hoc per-backend `backend_used` idiom (detector `real→heuristic`, embedder `real→histogram`) into one reusable abstraction. Includes an `importable()` probe helper.
- **`room4d/perception/instance/instance_artifact.py`** — `InstanceArtifact` (+ `InstanceObservation`): the shared **mask + identity** source of the reordered band — per-frame instance masks + persistent `track_id` + concept/kind routing hint. npz-mask persistence + index (mirrors `ObjectArtifact`); manifest under `metadata['capability']`.
- **`room4d/perception/reid/reid_artifact.py`** — `ReidArtifact`: per-tracklet appearance **embeddings** + a cross-occlusion **`merge_map`** with cycle-safe `canonical()` / `merged_groups()`; npz-embedding persistence.

#### Tests — 275 passed, 1 skipped (prior 260 + 15 new), zero regressions
`tests/test_backend_ladder.py` + `tests/test_perception_artifacts.py` cover ladder resolve/skip/**fail-through**/**graceful-empty**/no-rung, manifest round-trip, `importable` probe, and artifact save/load incl. masks/embeddings/indices/manifest/merge-canonicalization. Two tests pin the §4A degradability invariant directly: the floor rung always produces a result even when every model above is skipped, and an installed-but-crashing backend still degrades.

#### Doc
§7 updated: FP0 is additive; the Object/Human/Track contract re-cut moves to **FP1** (where those stages are rewired).

#### Next
**FP1** — add the `segment`/`reid` stages, reorder `STAGE_ORDER` to `pose_refine → segment → reid → objects → humans → tracking → graph`, wire the floor rungs (today's backends), re-cut Object/Human/Track to consume `InstanceArtifact`; pipeline runs end-to-end in the new shape, suite green. Then **F0b** resumes on floor-rung masks.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-27"></a>room4d pull request 27: docs: two-tier sync + per-stream-until-fusion (§4B)

Merged 2026-06-28 · `feature/sync-per-stream-design` → `develop` · 1 commit, +95 −14 in 3 files · merge commit `8bcfe58`

Folds the **sync / cross-stream-fusion** discussion into the design doc as new **§4B**. Decision-only — no code. VERSION `0.1.18.0` → `0.1.18.1`.

#### Why
The linear `ingest → sync → geometry → …` order is **rig-convenient**: on a hardware-synced rig sync is ~0 and there's nothing to estimate. For the north-star input — **several casually-captured, independently-clocked streams of the same dynamic volume** — robust temporal alignment depends on *reconstructed per-stream content*, so sync can't be a cheap stage-2 op on raw frames.

#### §4B captures
- **Two-tier sync.** Tier-1 = a **coarse temporal prior** (audio / motion-energy cross-correlation) at the early `sync` stage (this is what `SyncArtifact` carries; trivially ~0 on a rig). Tier-2 = **fine alignment** — time offset + frame-rate/drift estimated **jointly with extrinsics + scale in the F2 joint optimizer** via cross-view dynamic-content consistency (can't be written until each stream has geometry+motion+entities).
- **Per-stream-until-fusion.** Most of the front-half (`geometry`/`pose_refine`/`segment`/`reid`/`objects`/`humans`/`tracking`) is **single-stream**; cross-stream alignment (sync + extrinsics + scale + cross-view identity) is the **fusion** at F2/F4. The registry's `geometry depends on sync` edge is wrong for the casual case — geometry doesn't require sync.
- **Cross-view identity rides the same fusion** (cross-stream `reid` on DINOv3 features).
- **Validation:** rig ~0 recovery is necessary-not-sufficient; casual fine-sync needs an unsynced known-offset test.

#### Reconciled
§3 (sync inventory row), §4 (sync/geometry stage rows), §7 (F1 = Tier-1 coarse prior; F2 = + Tier-2 fine sync), §8 (new risk bullet). No renumbering — §4B slots between §4A and §5.

#### Note
Independent of the FP work — FP1 (structural reorder) continues on its own branch off `develop`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-28"></a>room4d pull request 28: FP1: reorder perception onto segment/reid stages (floor rungs)

Merged 2026-06-28 · `feature/perception-redesign-fp1` → `develop` · 1 commit, +695 −1,402 in 14 files · merge commit `bed7df6`

**FP1** of the perception redesign (design §4A) — the **structural reorder on floor rungs**. The pipeline now runs the new shape end-to-end using today's backends; **no new model downloads**, **full suite green (279 passed, 1 skipped)**. VERSION `0.1.18.1` → `0.1.19.0`.

#### New order
`pose_refine → segment → reid → objects → humans → tracking → graph` (identity now precedes motion).

#### New stages (floor rungs behind the `BackendLadder`)
- **`segment`** (`pipeline.segment` + `perception/instance/segment.py`) — floor = **COCO Mask R-CNN + greedy-IoU association** (`person`→human-kind, else object-kind) → `InstanceArtifact` (per-frame masks + persistent track ids). The shared mask + identity source. **SAM 3 is the top rung at FP2.**
- **`reid`** (`pipeline.compute_reid` + `perception/reid/reid_stage.py`) — floor = **mean ResNet18** embeddings per tracklet, empty `merge_map` → `ReidArtifact`. **DINOv3 + cross-occlusion merges at FP3.**

#### Re-cut consumers
- **`objects` / `humans`** consume `segment`+`reid` instead of detecting from scratch — identity is the canonical (reid-merged) track id; `objects` carries the COCO category (open-vocab at FP4), `humans` runs pose/DensePose/SMPL over the segment masks.
- **`tracking`** is now motion-only and runs after perception (geometry-based static/dynamic partition unchanged; mask-seeding lands with F0b).
- **Deleted** the hand-tuned associators (`object_assoc.py`, `human_assoc.py`).

#### Degradability verified
- Real-component graceful-empty path exercised end-to-end (`segment`=`maskrcnn_iou`, `reid`=`resnet18`, artifacts reload with capability manifests).
- Floor-rung ladders bottom out at today's backends, so this is a strict superset of the prior pipeline.

#### Known, intended degradation
Point-track support (`attributes['track_ids']`) is decoupled from identity (§4A) and **empty at FP1** → the graph's `support_track_ids` node attr empties out. Acceptable — `graph` is the passive builder, re-cut at P3. `scene`/`render` unaffected.

#### Tests
New `test_segment_stage.py` / `test_reid_stage.py`; `test_object_perception.py` / `test_human_perception.py` rewritten to consume-from-instance (associator tests dropped, artifact-roundtrip kept); resume/stage-window order assertions updated.

#### Next
**F0b** — resume dynamic render on FP1 floor-rung masks (prove the full perception→…→dynamic-render pipeline before any new tools), then climb FP2–FP6 ungated-first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-29"></a>room4d pull request 29: F0b: in-pipeline dynamic Gaussian render path (gate 20.72 mPSNR)

Merged 2026-06-29 · `feature/perception-redesign-f0b` → `develop` · 3 commits, +1,145 −79 in 16 files · merge commit `84825a6`

#### What

Closes F0b — the **render half** of the scene→render seam now resolves a *moving* scene end to end. `scene.kind=dynamic` lifts an SE(3) `DynamicScene` from the pipeline's own geometry + foreground, serializes it into the `SceneArtifact`, and the render stage trains + scores it. This closes the loop so **front-half perception backends become measurable by their downstream novel-view quality** — the back-half resolution guarantee from design §4A.

#### Gate (validation)

```
F0b GATE  source=own-tracker  iters=5000
  masked mPSNR=20.7219  mSSIM=0.6699  mLPIPS=0.1578  (n=419)
```

`20.72` on DyCheck paper-windmill reproduces the standalone `run_render_proof.py cmd_dynamic` reference (~20.7 / R0 20.75) **through the entire new wiring**: oracle `DyCheckGeometryProvider` geometry → in-memory packed all-pairs (no O(N²) disk files) → `NativeMotionInitializer` → `dump_bundle` → `SceneArtifact` **saved to disk + reloaded** → dynamic render backend (densified to 932k Gaussians, train PSNR 28.1) → dynamic masked-NVS scorer.

#### Pieces

- **(a) Serialization** — `motion_init.dump_bundle`/`bundle_to_scene` (torch↔numpy) + `SceneArtifact` dynamic-bundle npz (separate file, not inlined JSON); `NativeMotionInitializer.build(return_norm=True)` surfaces the scene-norm so held-out cameras land in the scene frame.
- **(b) Assembler dynamic branch** — `scene.kind=dynamic` builds from `ArtifactGeometryProvider` (segment-mask union → foreground) + in-memory all-pairs; `instance_artifact` threaded through `assemble_scene`. Default `kind=static` → F0a unchanged.
- **(c) Render dispatch** — `kind=="dynamic"` (was `NotImplementedError`) → `DynamicGaussianTrainer` + `trained_dynamic.npz`.
- **(d) Dynamic scorer** — `score_render_artifact_masked_nvs_dynamic`, wired into `fit_renderer`.
- **Driver** — `scripts/f0_validate_windmill.py` (reproducible gate).

#### Scope / follow-ons

- The gate validates the **render half** (pieces a/c/d + the in-memory all-pairs + native init) end-to-end on GPU. The **assembler dynamic branch (b)** is covered by its unit test + structural mirroring in the driver, but the driver does not call the pipeline/assembler directly (it injects oracle geometry via `DyCheckGeometryProvider`, not `ArtifactGeometryProvider`).
- **(B) fully-native baseline** — running the *full* pipeline (ingest→geometry→segment→…→dynamic scene) on paper-windmill is a deferred native-swap *ladder experiment*, not part of landing F0b.

#### Tests

Suite **284 passed, 1 skipped**. New: dynamic-bundle round-trip, fg-mask union → provider keys, scene-norm ↔ lift convention, dynamic dispatch (CPU empty-bundle + gsplat end-to-end train/checkpoint).

VERSION `0.1.19.0` → `0.1.20.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-30"></a>room4d pull request 30: FP2a: SAM 2 segment top rung (prompt-free, class-agnostic)

Merged 2026-06-29 · `feature/perception-redesign-fp2a` → `develop` · 1 commit, +433 −2 in 6 files · merge commit `9eddbe7`

#### What

First rung of the perception climb above the FP1 Mask R-CNN floor (design §4A). **SAM 2 automatic mask generation** discovers object-like masks on a keyframe — **no class list, no prompt** — a tunable object-ness filter prunes them, and the **SAM 2 video predictor** propagates each across the stream with a **persistent native ID**. That's real temporal identity in place of the floor's greedy-IoU stitching.

`segment` stays **class-agnostic** by design: it emits only masks + identity; naming is a downstream concern (FP4 semantics caption each tracked mask). SAM 3 is the eventual gated top rung — SAM 2 (Apache, ungated) is the buildable-now stand-in.

#### Pieces

- **`room4d/perception/instance/sam2_segment.py`** — `SAM2InstanceSegmenter`: discover → object-ness filter → SAM 2 video propagation → person-tag → `InstanceArtifact`. The existing person detector tags `kind='human'` so the `humans` stage keeps working class-agnostically. sam2 is lazy-imported (loaded only on run) → the module is import-safe when sam2 is absent.
- **`room4d/api/pipeline.py`** — `segment` ladder gains a probe-gated `sam2` rung (`importable('sam2')`) above `maskrcnn_iou`, built inside the runner so the model loads only when selected and releases after the stage. **Falls to the Mask R-CNN floor when sam2 is absent** — degradability invariant intact (CI path unchanged).
- **`scripts/setup_room4d.sh`** — `--with-sam2`: SAM 2 pinned (`2b90b9f`) into `third_party/sam2` + the `sam2.1_hiera_large` checkpoint (direct fbaipublicfiles URL, idempotent, size-guarded). `SAM2_BUILD_CUDA=0` keeps the install off the CUDA-toolchain path.

#### Verified

- **Real install ran here:** SAM-2 1.0 + 857 MB checkpoint; auto-mask-gen + video predictor load on GPU.
- **Functional test (real SAM 2):** synthetic moving-square clip → **one persistent track across 4 frames**, mask area a constant 3600 (60×60), `kind=object`.
- Pure tests: object-ness filter (keep object / drop background, low-stability, tiny) + bbox/IoU helpers.
- **Full suite: 288 passed, 1 skipped.**

#### Notes / scope

- **Dropped the SAM 3 placeholder rung** — a rung with an unimplemented runner is a latent bug; a comment marks where SAM 3 slots in instead.
- The sam2 install **downgrades setuptools 82→81** (benign; suite green).
- **v1 limits (follow-on):** single-keyframe seed + forward propagation (objects appearing later are missed → periodic re-discovery is a refinement). Full-pipeline run on real data is **FP2b**; re-validating F0b on real SAM 2 masks is **FP2c**.

VERSION `0.1.20.0` → `0.1.21.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-31"></a>room4d pull request 31: FP2c: motion-gated dynamic foreground (closes perception→render loop, 20.43 mPSNR)

Merged 2026-06-30 · `feature/perception-redesign-fp2c` → `develop` · 1 commit, +278 −16 in 8 files · merge commit `4803b35`

#### What

Closes the **perception→render payoff loop**. The dynamic-init lift now keeps only **ever-moved** tracks as deformable foreground and reroutes static tracks to the background — so a broad, class-agnostic segment foreground (SAM 2's "segment everything", incl. static deck/pot) is cleaned by **motion**, not a per-scene appearance heuristic.

#### Validated end to end (paper-windmill)

| Run | fg source | masked mPSNR | mSSIM | mLPIPS |
|---|---|---|---|---|
| F0b oracle gate | DyCheck dataset fg | 20.72 | 0.670 | 0.158 |
| **FP2c** | **SAM 2 fg + motion gate** | **20.43** | 0.660 | 0.192 |

The fully-native front half (real SAM 2 masks, motion-cleaned) **holds the oracle's novel-view quality within 0.3 dB** (n=419 val). The SAM 2 foreground covered **66%** of the frame; the gate cut it to the **moving pinwheel** (16k → after densify 88k fg Gaussians) and rerouted the static background to bg. This demonstrates the F0b thesis: front-half backends are now **measurable by their downstream NVS impact**.

*Caveat (on record):* FP2c used a coarser **grid-12** tracker (the broad SAM 2 fg OOMs at grid-4/277-frames), vs the oracle gate's grid-4, so part of the 0.3 dB may be grid, not fg-source. The loop holds either way; the grid control was deliberately skipped.

#### Pieces

- **`lifting/native_track_lift.gate_tracks_by_motion`** — foreground iff **p95** of a track's per-frame world-space displacement from its median center exceeds a threshold. p95 is the robust "did it ever move?" test: a *median* reducer averages brief motion away (the static-then-moving case), a raw *max* trips on single-frame lift glitches. Static tracks → static bg Gaussians at their median position (their segment-fg-masked regions would otherwise be holes). Threaded through `NativeMotionInitializer` + the assembler, **opt-in** (`scene.dynamic_motion_gate`; default off → F0b gate + suite unchanged).
- **`render/gaussian4d/dynamic_field.cloud_at`** — clamp the frame index to the motion-bases range: an out-of-range time-id (held-out camera beyond trained frames) degrades to the nearest frame instead of an out-of-bounds CUDA assert.
- **`scripts/f0_validate_windmill.py`** — `--fg-source {dataset,sam2}` (SAM 2 union masks via a wrapped oracle-geometry provider), `--motion-gate`, `--tracker-grid-size`, `--max-frames` — the native-swap ladder driver.

#### Tests

Motion gate: p95 ever-moved fg vs static-rerouted-to-bg; the static-then-moving case median misses but p95 catches; no-op when all dynamic. **Full suite: 291 passed, 1 skipped.**

#### Note
All GPU validation ran under a 400 W power cap + temp watchdog (peak 68°C).

VERSION `0.1.21.0` → `0.1.22.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-32"></a>room4d pull request 32: feat(FP3): DINOv2 cross-occlusion re-ID top rung (mutual-NN merge_map)

Merged 2026-06-30 · `feature/perception-redesign-fp3` → `develop` · 1 commit, +690 −12 in 9 files · merge commit `0ec2a00`

#### FP3 — DINOv2 cross-occlusion re-ID top rung

Climbs the `reid` ladder (design §4A / §7). Adds a frozen DINO ViT appearance rung **above the ResNet18 floor** that emits a `merge_map` re-acquiring identity *across occlusion gaps* — two tracklets the segmenter gave separate ids but that are the same entity (it left, came back) fold onto one canonical id. The `ReidArtifact` contract was already in place (FP0); FP3 fills the previously-empty `merge_map`.

##### What's in it
- **`perception/reid/dino_reid.DinoReid`** — per-tracklet mean DINO features (decode-once-per-frame, masked crops, L2-normalized). Lazy model load → constructing the rung is import-safe when `transformers` is absent.
- **`build_cross_occlusion_merge_map`** — a **pure** (no-torch) merge builder. Candidates are same-stream, span-**disjoint** pairs (co-visible tracklets are a different problem and never merged); a pair merges only if it's a **mutual nearest neighbour** (+ optional Lowe-style ratio test, off by default), folding onto the earliest-appearing canonical. Mutual-best matching is a *matching* → merges pairs, not multi-gap chains (a deliberate v1 precision choice; chaining is a documented follow-up).
- **`api/pipeline.compute_reid`** — prepends the `dinov2` rung above `resnet18`, gated on `importable('transformers')`. The reid runner contract is now `(embeddings, merge_map)` (mirrors segment's `(observations, mask_arrays)`); missing deps/weights degrade to the ResNet18 floor, so the pipeline never breaks.
- **`scripts/setup_room4d.sh --with-dinov2`** — installs HuggingFace transformers + pre-caches DINOv2-L weights (idempotent).
- **DINOv3-L is a `model_id` swap** — the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` loads through the same `AutoModel.from_pretrained` call once HF access lands (default is ungated `facebook/dinov2-large`).

##### Validation (synthetic occlusion on real DINOv2 features)
`scripts/fp3_validate_reid_merge.py` splits real SAM 2 windmill tracks into time-disjoint halves (the simulated occlusion) and measures merge precision/recall. `reid` does not feed render, so there's no mPSNR gate — the bar is intrinsic merge quality.

| Merge rule | Recall | Precision |
|---|---|---|
| Greedy threshold (initial) | 3/5 | 2/20 false ❌ |
| **Mutual-NN (shipped)** | 2/5 | **0/20 false** ✅ |

**Precision 1.0 — zero false merges.** Recall 2/5 reflects the *adversarial* worst case (five near-identical spinning petals, indistinguishable to appearance-only re-ID across rotation); the rung correctly **abstains** rather than mis-joining. Cross-entity cosine floor of 0.08 confirms clean separation when entities actually differ. A false merge corrupts an entity node while a missed merge leaves recoverable fragments, so precision-over-recall is the right bias here. Cross-occlusion re-acquisition "remains the open risk" per the design doc.

##### Tests
Full suite green (**300 passed, 1 skipped**). New `tests/test_dino_reid.py` covers the mutual-NN + ratio builder (mutual/ratio/co-visible/cross-stream exclusion, earliest-canonical, empty) and a fake-embed `DinoReid.run` integration (no weights); `tests/test_reid_stage.py` updated for the tuple contract.

VERSION → 0.1.23.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-33"></a>room4d pull request 33: feat(FP4a): SigLIP2 open-vocab semantics stage (names tracklets)

Merged 2026-06-30 · `feature/perception-redesign-fp4a` → `develop` · 1 commit, +843 −12 in 12 files · merge commit `1998959`

#### FP4a — SigLIP2 open-vocab semantics rung

Climbs the semantics slot (design §4A / §7). Names the class-agnostic instances `segment` produces: a new **`semantics` stage** attaches an open-vocab **category** (+ a SigLIP image embedding) to each canonical (reid-merged) tracklet. Previously, when the SAM 2 rung ran, `ObjectState.category` was the empty `concept` string and flowed straight to an empty graph-node label — this fills that gap. Qwen2.5-VL free-form attributes/captions arrive at FP4b.

##### What's in it
- **New `semantics` stage** between `reid` and `objects` (`STAGE_ORDER` + `stages.py`; deps ingest+segment+reid). `pipeline.compute_semantics` runs `BackendLadder('semantics', [siglip2 → coco80 → unknown])`: the SigLIP2 top rung (probe `importable('transformers')`) zero-shot matches masked tracklet crops against a candidate vocabulary; the floor carries the detector's COCO-80 `concept`, degrading to `"unknown"`. So a missing dep/weights degrades to the COCO floor — the pipeline never breaks.
- **`SemanticsArtifact`** — per-canonical-track `SemanticRecord` (category, score, top-k, attributes, caption) + SigLIP embeddings (npz), mirroring `ReidArtifact`.
- **`SiglipSemantics`** (lazy transformers SigLIP2, masked-crop image features, zero-shot) + **`FloorSemantics`** (COCO concept → `"unknown"`) + the pure unit-testable **`score_zero_shot`**. Default `google/siglip2-so400m-patch14-384` (ungated/Apache); a `_pooled` shim handles the transformers-5.x `BaseModelOutputWithPooling` return.
- **`perceive_objects`** reads the semantics category (falling back to the COCO concept) into `ObjectState.category` → graph node; top-k stashed in attrs.
- **`setup_room4d.sh --with-siglip2`** + **`scripts/fp4_validate_semantics.py`** (two-gate driver).

##### Validation (`semantics` doesn't feed render → no mPSNR gate; intrinsic label quality)
| Gate | Result |
|---|---|
| **COCO-agreement** (spin, SigLIP2 vs Mask R-CNN over COCO-80) | top-1 **62%** / top-5 **77%** raw; **~85%** on persistent objects |
| **Windmill open-vocab** (SigLIP2 over extended vocab on SAM 2 tracks) | loose — fragmented petals only |

On the COCO gate, every persistent object agrees (person 4/4, chair 3/4); the raw 62% is dragged down by one-frame Mask R-CNN false positives on background clutter (`car→knife`, `clock→suitcase`) — restricting to multi-frame tracks lifts top-1 to ~85%. So SigLIP2 is a **faithful labeler on clean objects**. The windmill is the honest limit: SAM 2 over-segments the pinwheel into petal *fragments*, and small non-COCO part-crops give flat low-confidence scores — whole-object naming wants the reid-merged whole mask (the FP3 limitation resurfacing), not an FP4a defect.

##### Tests
Full suite green (**307 passed, 1 skipped**), incl. the vertical slice running the real SigLIP2 `semantics` stage end-to-end. New `tests/test_siglip_semantics.py` (pure scorer + floor + fake-embed integration, no weights); `test_resume_pipeline.py` / `test_stage_window.py` updated for the new stage in `STAGE_ORDER`.

VERSION → 0.1.24.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-34"></a>room4d pull request 34: feat(FP5a): Sapiens dense human rung + F3 de-fabrication (ladder-ify humans)

Merged 2026-07-01 · `feature/perception-redesign-fp5a` → `develop` · 1 commit, +743 −243 in 15 files · merge commit `34b3972`

#### FP5a — Sapiens dense human rung + F3 de-fabrication

Climbs the `humans` dense sub-ladder (design §4A / §7) with Meta's **Sapiens**, and executes the **F3 subsumption** by making the humans wrappers honest. **FP5b (GVHMR world-SMPL) is deferred** — its blockers (MPI-registered SMPL assets + gravity-aligned camera poses, both net-new) and its payoff both live near the back-half graph re-cut, so it's better sequenced there.

##### What's in it
- **`sapiens_wrapper.SapiensEstimator`** — wraps Sapiens-lite **TorchScript** (pose / body-part seg / depth / normals), top-down per person crop: aspect-pad to 768:1024, ImageNet-normalize, decode heatmaps→keypoints (Goliath-308) + seg (28 parts) + depth + normals back to image space. Ungated HF checkpoints (`facebook/sapiens-{task}-0.6b-torchscript`), **CC-BY-NC-4.0** (non-commercial — fine for this no-ship repo). Lazy + import-safe.
- **`dense_ladder`** — the rung adapters (`SapiensDenseRung` / `KeypointRcnnRung` / `BboxOnlyRung`): thin glue that does the decode-once-per-frame loop and calls each **tool wrapper's** `predict(frame, boxes)`. The tool wrappers stay separate + swappable (per the wrapper-per-tool convention).
- **`pipeline.compute_humans`** (renamed from `perceive_humans`) — `BackendLadder('humans', [sapiens → keypoint_rcnn → bbox_only])` gated by a Sapiens **checkpoint-presence** probe (`--with-sapiens`), degrading to the honest Keypoint-RCNN floor. `HumanArtifact` gains a `CapabilityManifest` (the last stage that lacked one); `HumanState`/`HumanArtifact` carry a `dense` payload.
- **F3 subsumption (`### Changed`)** — `pose2d` no longer fabricates a skeleton for unmatched boxes and stamps it `'real'` (returns `None`, honest label); `densepose_wrapper` drops the fake bbox-grid colorizer (honest `unavailable`); `smpl_fitter` drops the fabricated betas/pose (honest `skip`). All kept as swappable seams — now consistent with the honest `backend_used` every other wrapper already reports.
- **`setup_room4d.sh --with-sapiens`** (pre-caches the 4 0.6B checkpoints, ~5 GB) + **`scripts/fp5_validate_humans.py`**.

##### Validation (no render gate — humans is dead data downstream until P3)
Ran the Sapiens rung on real segmented `spin` people (`backend=sapiens_0.6b`):

| Track | keypoints | in-box | seg parts | depth | normals |
|---|---|---|---|---|---|
| real person (`:N:0`) | 308 (~250 vis) | **1.00** | **8–9** | well-formed | unit |
| spurious detections | 308 | mixed | 1 (bg) | — | unit |

The real person is fully and correctly analyzed; spurious Mask R-CNN "person" boxes correctly yield near-empty. (The dense outputs aren't consumed downstream yet — render/scene ignore humans; the graph reads only presence flags — so this rung enriches data the **P3 graph re-cut** will consume.)

##### Consistency note
The humans stage is now on the §4A pattern (swappable wrappers + `BackendLadder` + `CapabilityManifest`). Remaining cross-stage naming/structure drift (e.g. `perceive_objects` → `compute_objects`, the FP4a single-file semantics) is deferred to a **post-front-half consistency pass**.

##### Tests
Full suite green (**315 passed, 1 skipped**), incl. the vertical slice through the new dense ladder. New `tests/test_humans_dense.py` (honest wrappers, dense-ladder rungs, pure Sapiens decode math); `test_human_perception.py` updated for the rename + floor path.

VERSION → 0.1.25.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-35"></a>room4d pull request 35: feat(FP6a): MoGe-2 metric-depth rung + geometry backend ladder/manifest

Merged 2026-07-01 · `feature/perception-redesign-fp6a` → `develop` · 1 commit, +682 −14 in 9 files · merge commit `58d29c2`

#### FP6a — MoGe-2 metric-depth rung (native-swap) + geometry backend ladder

Adds **MoGe-2** (Microsoft, MIT, ungated) single-image metric depth as an explicitly-selectable `geometry` rung, and gives the geometry stage the observable-degradation machinery (`CapabilityManifest` + graceful-empty `BackendLadder`) every other stage already had.

##### What's in it
- **`geometry/moge_wrapper.MoGeInitializer`** — per-frame MoGe-2 → `depthmaps_by_frame` + camera-space `pointmaps_by_frame` + per-frame `intrinsics_by_stream` (normalized→pixel). Depth-only & honest: MoGe has no cross-frame poses, so `camera_states_by_stream` stays empty. Lazy/import-safe; shared `infer_frame()`; releases VRAM after the frame loop.
- **`GeometryArtifact.attach_manifest()` / `capability`** — the last geometry artifact that lacked a `CapabilityManifest`.
- **`pipeline.initialize_geometry`** — now runs `BackendLadder('geometry', …)`: config-selected backend on top + a **graceful-empty floor**, manifest stamped. `backend_name='moge'` force-selects MoGe; only that rung is probe-gated (`importable('moge')`) so the load-bearing vggt/dataset rungs never false-skip.
- **`setup_room4d.sh --with-moge`** — vendors `microsoft/MoGe` at a pinned commit (`--no-deps` + utils3d/opencv/scipy, self-contained), pre-caches **and smoke-infers** the `Ruicheng/moge-2-vitl-normal` weights.
- **`scripts/fp6_validate_moge.py`** — the native-swap validator (metric + scale-invariant depth error vs the GT oracle).
- **Design doc §4A.1 (new) + §4/§4A/§7** — geometry reframed as **two fused pose+depth ladders** (VGGT *joint* vs MoGe/COLMAP *decomposed*), a coarse geometry-free `dyn_mask` capability (breaks a real `segment`→`geometry` cycle), a static-superset input-ordering invariant, and a FP6 → FP6a/FP6b resequence — all after an adversarial review that caught the original FP6a having no live MoGe path.

##### Native-swap result (MoGe depth vs GT oracle)
| sequence | structure (scale-aligned δ<1.25) | metric scale | focal |
|---|---|---|---|
| spin | **0.985** | 1.13 (δ 0.886) | ×1.05 |
| paper-windmill | **0.998** | 0.645 (δ 0.002) | ×1.02 |

Depth *structure* is excellent; absolute *metric* scale carries a per-scene offset — empirically confirming MoGe is a strong **structural** backstop, not a drop-in **metric** one. This motivates FP6b's scale reconciliation.

##### Scope boundary
Auto-selecting MoGe below VGGT is **inert** (VGGT/GT already emit depth), so the auto-probing pose/depth split, the calibration-arc SfM pose rung, and COLMAP↔MoGe scale reconciliation are deferred to **FP6b** (design §4A.1 / §7).

##### Tests
`tests/test_moge_geometry.py` (5 new: infer shapes/intrinsics, empty-in→empty-out, depth-filled/no-poses, ladder selects MoGe + manifest, graceful-empty on rung failure) + full-suite regression: **320 passed, 1 skipped**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-36"></a>room4d pull request 36: feat(FP6b-step1): COLMAP-SfM pose rung + geometry-ladder wiring

Merged 2026-07-01 · `feature/perception-redesign-fp6b-step1` → `develop` · 1 commit, +454 −7 in 7 files · merge commit `59edd41`

#### FP6b Step 1 — COLMAP-SfM pose rung (native-swap)

The **pose** mirror of FP6a's MoGe depth rung (design §4A.1): monocular hloc+COLMAP camera egomotion — the first piece of the geometry pose/depth split. Where MoGe gives metric depth but no poses, COLMAP gives poses but no dense depth; step 4 fuses them.

##### Spike-gated (the honest part)
Before writing any of this, a throwaway spike (`spike_fp6b_egomotion/`, sibling of the repo) answered the one regime the calibration arc's S1 spike **never tested**: monocular egomotion on *dynamic handheld* clips (S1 was multi-cam static-rig, where dynamic content *helps*). Verdict: **cleared** — and it refined the plan (dense sampler is load-bearing; `dyn_mask` is a size-gated robustness rung, not a hard gate). Findings folded into design §7 FP6b + §4A.1.

##### What's in it
- **`geometry/colmap_wrapper.ColmapPoseInitializer`** — runs `HlocColmapSfMRunner` (SuperPoint + SuperGlue + NetVLAD) over dense frames, **intrinsics fixed from the focal prior** (`camera_mode=SINGLE`), reads the **binary** model via `pycolmap`, fills `camera_states_by_stream` + `intrinsics_by_stream`. **Pose-only, honest:** `depthmaps`/`pointmaps` stay empty. Shared `reconstruct()` core (mirrors MoGe's `infer_frame`).
- **`pipeline.initialize_geometry`** — `backend_name='colmap'` force-selects the rung (fidelity `sfm_poses_no_depth`, probe `importable('pycolmap','hloc')`).
- **`scripts/fp6b_validate_colmap.py`** — native swap vs GT extrinsics through the shipped `reconstruct()`.

##### Validation (60 frames, fixed intrinsics, via the shipped path)
| sequence | registered | pairwise-rel rot (med/max) | Sim(3)-APE rot | APE trans |
|---|---|---|---|---|
| spin | 60/60 (100%) | 0.19° / 1.01° | 1.53° | 0.7% |
| paper-windmill | 60/60 (100%) | 0.09° / 0.27° | 1.04° | 0.3% |

##### Scope boundary
Step 1 lands COLMAP as a config-selectable **pose rung validated vs the GT oracle**. It does *not* yet: combine pose+depth (**step 4**), add `dyn_mask` (**step 3** — unmasked already works at density), or do COLMAP↔MoGe scale reconciliation (**step 5**).

##### Note
Decision #3 adjusted: `calibration/model_io` is text-only but hloc writes a **binary** model, so poses are read via `pycolmap` directly (the proven spike path).

##### Tests
`tests/test_colmap_geometry.py` (5, SfM mocked) + full-suite regression: **325 passed, 1 skipped**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-37"></a>room4d pull request 37: feat(FP6b-step2): geometry pose/depth split + COLMAP↔MoGe scale reconciliation

Merged 2026-07-01 · `feature/perception-redesign-fp6b-step2` → `develop` · 1 commit, +1,110 −45 in 8 files · merge commit `37d2568`

#### What

Fuses the two decomposed geometry rungs — **FP6a MoGe depth** + **FP6b-step1 COLMAP poses** — into a single, *complete* **GT-free, VGGT-free metric** `GeometryArtifact` (design §4A.1). Monocular COLMAP is scale-free, MoGe is metric; a single global scale ties them together. This is the payoff of the pose/depth split: the two single-purpose rungs become complete geometry together.

#### How

- **`geometry/scale_reconcile.py`** — solves the COLMAP→metric scale from sparse-point reprojections vs MoGe depth (robust median of per-observation depth ratios). The solve is a **per-frame camera-space** comparison → **gauge-free**, so it runs *before* ground-plane canonicalization (scale is the one gauge DOF the ground plane doesn't pin, §4A.1). Pure `solve_global_scale` (unit-tested, COLMAP-free) + thin, probe-gated `extract_colmap_observations` (`pycolmap`). Emits the **per-frame residual** (→ F2); identity fallback when under-sampled — never a crash.
- **`geometry/compose.py` (`GeometryComposer`)** — merges pose + depth channels, applies the reconciled scale to camera centers, and **rigidly** transforms MoGe's camera-space points into world via the metric poses (MoGe's own per-frame intrinsics kept self-consistent; never re-projected through COLMAP's focal, §4A.1(a)). Resulting `pointmaps_by_frame` are world-space (VGGT convention) so ground-plane + downstream are unchanged. Graceful `depth_only`/`poses_only`/`empty` fallbacks.
- **`api/pipeline.py`** — `geometry.pose_backend` + `geometry.depth_backend` select the **two fused ladders**; each channel is a 2-rung ladder (backend → graceful-empty) and the composite `CapabilityManifest` records both channels (`pose:`/`depth:`) + resolved fidelity. A lone `backend_name` still selects the single-backend path — **strict superset, existing configs unchanged.**
- **`geometry/colmap_wrapper.py`** — exposes the persisted sparse model dir for the reconciler.
- **`scripts/fp6b_validate_fused.py`** — native-swap validation vs GT.

#### Validation (spin + windmill, 60 frames, healthy COLMAP draws, reproducible across 2 runs each)

| seq | reconcile residual (p10–p90, med) | Sim(3)-APE trans | metric SE(3)-APE (rot / trans) | MoGe scale vs GT |
|---|---|---|---|---|
| spin | [0.92, 1.08] med 1.00 | **0.7%** | 0.8–1.4° / 3.3–3.4% | 1.14 |
| windmill | [0.95, 1.10] med 1.02 | **0.3%** | ~1.1° / 12.9% | 0.686 |

- **Reconciliation collapses the per-scene offset to one global scale** (±10% per-frame residual) — FP6a needed a *per-frame* oracle scale; this needs one global scale + a small F2 residual.
- **Trajectory shape is excellent** (Sim(3)-APE trans 0.3–0.7%, rot ~1°). The entire remaining metric SE(3) error is **MoGe's own per-scene depth bias** (spin +14%, windmill −34%) — exactly the F2 person-anchored-scale hand-off, not a fusion defect.

#### Known issue (documented, → step 3)

Monocular COLMAP is **intermittently unstable** on these low-parallax dynamic clips: incremental BA occasionally hits `Linear solver failure` (dense Cholesky) and lands a **degenerate** reconstruction (100% registered, low reproj error, but geometrically wrong — ~once each on spin/windmill across ~8 runs; healthy draws are the norm and reproduce tightly). **Not introduced by step 2** (a property of the step-1 rung; fusion is correct on healthy draws). Mitigation belongs to **FP6b step 3** (`dyn_mask` robustness rung + dense sampler attack the *cause*) plus a **per-stream, self-calibrating reconstruction health-check** (a global pairwise-rotation threshold would false-positive across streams and can't catch a globally-bent-but-locally-smooth draw — so it's deferred to be designed properly, not bolted on here).

#### Tests

`tests/test_fused_geometry.py` (11): pure scale solver; composer full fusion + all three degradations; pipeline fused-mode wiring + composite manifest; pose-channel failure ⇒ `depth_only` (no crash); back-compat (lone `backend_name` not fused). **Full suite: 336 passed, 1 skipped.**

VERSION `0.1.27.0` → `0.1.28.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-38"></a>room4d pull request 38: feat(FP6b-step3): dyn_mask robustness rung + static-superset reorder + pose health-check

Merged 2026-07-01 · `feature/perception-redesign-fp6b-step3` → `develop` · 1 commit, +932 −20 in 14 files · merge commit `34d4560`

#### What

Closes the two robustness gaps step 2 surfaced: **protect the COLMAP pose rung** from dynamic-content correspondences (a coarse geometry-free `dyn_mask`), and **make its intermittently-degenerate draws observable** (a GT-free health-check). Plus the static-superset reorder that lands `dyn_mask` as a real stage.

#### How

- **`perception/dyn_mask/`** — new coarse, **geometry-free** stage → `DynMaskArtifact` (masks persisted as `.npz`). Ladder: **detector** (COCO Mask R-CNN movable-category masks, reuses `ObjectDetector`) → **all-static** floor (no-op → VGGT-on-top stays a no-op). Consumes only raw frames — never `segment`/geometry — which is what breaks the `segment→geometry` cycle (§4A.1). The general optical-flow motion-seg top rung is a deferred follow-up (native-swap discipline).
- **`geometry/colmap_wrapper.py`** — the pose rung consumes `dyn_mask`, **size-gated**: blacks out dynamic pixels (`rgb[mask>0.5]=0`, the spike's mechanism) only when the median per-frame dynamic fraction clears `mask_size_gate` (default 0.10). Per the spike, masking is a *size-gated* help (tightens large-dynamic scenes, neutral on small ones).
- **`geometry/reconstruction_health.py`** — GT-free `assess_pose_health` flags degenerate SfM draws via a **per-stream, self-calibrating** outlier test on consecutive pairwise-relative rotation (per-stream so multi-cam angles don't false-positive; median + k·MAD *and* an absolute cap). Stamped into `metadata['pose_health']`; the composer downgrades fused fidelity to `fused_metric_suspect` when suspect.
- **Pipeline reorder (§4A.1 fourth invariant)** — `dyn_mask` between `sync` and `geometry`; geometry deps → `(ingest, sync, dyn_mask)`. Ladder gates work, never reorders. Threaded only to the COLMAP rung; default `all-static` → strict superset, existing configs unchanged.

#### Validation (`scripts/fp6b_validate_dynmask.py`; spin 8 runs/arm, windmill 4)

| seq | detector fraction → gate | arm | degenerate | pairwise-max | Sim(3)-APE trans |
|---|---|---|---|---|---|
| spin | 0.351 → **MASK** | unmasked | **1/8** | 1.09° | 0.7% |
| spin | | detector-masked | **0/8** | **0.33°** | 0.5% |
| windmill | 0.000 → **unmasked** | unmasked | 0/4 | 0.26° | 0.3% |
| windmill | | detector-masked | 0/4 | 0.24° | 0.3% |

- **Size-gate fires correctly** — spin (large-dynamic person) masks, windmill (small pinwheel) is left unmasked.
- **Masking reduces the large-dynamic instability** — degenerate draws **1/8 → 0/8**, pairwise-max tail **1.09° → 0.33°** (exactly the oracle-mask spike's masked-spin number), and the **health-check caught the unmasked degenerate draw**.
- **Windmill neutral** — no regression.

**Caveats (honest):**
- **Small N** — 1/8 → 0/8 is suggestive, not statistically conclusive; the pairwise-max tail tightening is the clearer, spike-consistent signal.
- **Windmill gates off via detector fraction = 0.000** — the COCO detector is blind to the pinwheel (not a movable category), so the gate's *small-but-nonzero* branch isn't exercised here; that's what the deferred **flow-seg** rung would exercise (it'd see the pinwheel, and the gate would then leave it unmasked as small).
- **Health-check scope** — catches the outlier-jump degenerate type, not a globally-bent-but-locally-smooth one (no cheap pose-only signal does).

#### Tests

`tests/test_dyn_mask.py` (11): `DynMaskArtifact` npz round-trip; all-static empty masks; detector `mask_frame` unions only movable categories; health-check ok/flagged/too-few; COLMAP size-gate masks-when-large / skips-when-small (verifies pixels blacked out) + stamps `pose_health`; pipeline `dyn_mask` stage; composer downgrade to `fused_metric_suspect`. Three stage-order tests updated for the new stage. **Full suite: 347 passed, 1 skipped.**

VERSION `0.1.28.0` → `0.1.29.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-39"></a>room4d pull request 39: fix(e2e): runner stage-label fix + end-to-end pipeline run findings

Merged 2026-07-02 · `feature/e2e-integration-fixes` → `develop` · 1 commit, +95 −12 in 4 files · merge commit `a98b3bc`

#### What

First **full end-to-end run of the pipeline on real data** (spin), across the geometry-backend matrix, after FP6b completed the geometry backend. Goal (design §7): prove the chain runs → renders a novel view, and surface integration gaps — the F4/G1 discovery mechanism. This PR lands the one **trivial** fix that unblocked the run, plus the findings doc; the deeper gaps it surfaced are **documented, not applied** (they're behavior/design changes that need sign-off).

#### Code change (trivial, display-only)

`run_single_video_room4d.py` kept a hardcoded `stage_labels` map (labeled "/11") that had drifted from `STAGE_ORDER` — missing `dyn_mask` (FP6b step 3) plus the earlier `segment`/`reid`/`semantics` — so a full run hit `KeyError: 'dyn_mask'`. Labels now **derive from `STAGE_ORDER`** (single source of truth) so numbering/coverage can't drift. No stage runs differently.

#### Findings (`docs/design/2026-07-02-e2e-pipeline-run-findings.md`)

| Combo | Geometry | Outcome |
|---|---|---|
| **A** | `vggt` | **Full end-to-end success** → `flythrough.mp4` (60-frame novel view) + trained gaussians. Resume-from-prior works. |
| **B** | fused `colmap`+`moge` (FP6b) | **Runs but not pipeline-robust** — MoGe reliable every frame; COLMAP erratic on lossy video frames → composer degrades to `depth_only` correctly. |
| **C** | `dataset` GT | **Incompatible with video ingest** (`stream_id=None` → GT-name mismatch) → graceful-empty, no crash. |

**Proven:** the chain runs end-to-end and renders on real data (combo A); the new `dyn_mask` stage, FP6b manifests, resume, and gaussian4d render all integrate; and the ladder's **graceful degradation held** (B and C degraded instead of crashing).

**Key gap:** FP6b's fused geometry — clean in the validators (lossless PNGs + fixed intrinsics) — isn't robust through the real pipeline's lossy-video ingest. The weak link is **COLMAP** (a known theme from FP6b steps 2–3), not MoGe.

##### Gap ledger (severity-ranked in the doc)
- **P1 · fixed:** runner label drift (this PR).
- **P1 · open (needs sign-off):** fused/COLMAP not pipeline-robust; no dataset/frame-dir ingest path.
- **P2 · open (needs sign-off):** runner CLI can't select the fused backend.
- **P2 · known roadmap:** `optimization` (F2) stub; `graph` passive builder (P3).
- **P3 · open:** static-only scene on sparse keyframes.

##### Recommended next
A focused **FP6b-hardening pass**: a lossless frame-sequence ingest adapter + intrinsics-prior/density as first-class fused config would make **both** combo B (fused) and combo C (dataset) viable end-to-end. Not in this PR — flagged for your sign-off.

#### Tests
Full suite **347 passed, 1 skipped** (label fix is display-only; the failure mode is removed structurally — the map now covers all of `STAGE_ORDER` by construction).

VERSION `0.1.29.0` → `0.1.29.1` (MICRO — fix + docs).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-40"></a>room4d pull request 40: feat(ingest): lossless frame-directory ingest + intrinsics-prior wiring

Merged 2026-07-02 · `feature/frame-dir-ingest` → `develop` · 1 commit, +237 −9 in 6 files · merge commit `e955c97`

#### What

Closes the two **P1** gaps the end-to-end run ([findings doc](docs/design/2026-07-02-e2e-pipeline-run-findings.md)) surfaced — **fused geometry not pipeline-robust** and **no dataset/frame-dir ingest** — which turned out to be **one root story**: the pipeline only ingested *videos*, so dataset frame dirs got re-encoded to a **lossy mp4** (degrading COLMAP) with a `stream_id` that didn't match GT filenames.

#### How

- **`ingest/decoder.py`** — `decode()` now accepts a **directory** and ingests it as a **lossless frame sequence** (no re-encode). `stream_id` + `frame_idx` are **parsed from filenames** (`0_00060.png` → cam `"0"`, idx `60`) so dataset ground-truth (`{cam}_{idx}.ext`) aligns. `get_frame`/`iter_frames` read the originals via `cv2.imread`. Single-camera-per-dir scope (multi-camera split deferred). **Videos are unchanged.**
- **`run_single_video_room4d.py`** — `--frames-dir` (lossless, alternative to `--video`; exactly one required — you can now **A/B the same data as frames vs a video**) + `--geometry-camera-params "fx,fy,cx,cy"` (sets `geometry.camera_params` + `fix_intrinsics` for the COLMAP pose rung).

#### Validated on spin (from raw frames + GT-focal prior)

| | lossy video (before) | **raw frames + intrinsics (this PR)** |
|---|---|---|
| **Combo B · fused** COLMAP registered | erratic (0–71) | **71/71** |
| **Combo B · fused** reconciliation | 3 frames / 445 obs | **71 frames / 40,189 obs** (residual 0.996, healthy) |
| **Combo C · dataset GT** backend | `empty` (stream_id mismatch) | **`dataset_gt`** (`stream_id="0"` aligns) |

**One change fixed both previously-broken geometry combos.** The lossy re-encode was a **test-harness artifact, not a pipeline defect** — real device videos are native quality; datasets should just be ingested as frames. Findings-doc ledger updated (gaps #2/#3 → **FIXED**; #4 partial — `--geometry-camera-params` added, pose/depth-backend selection still config-only).

#### Tests

`tests/test_frame_dir_ingest.py` (9): filename parsing; frame-dir decode (camera + indices + size); **lossless** `get_frame` (exact match to source, by parsed index not position); missing-index/empty-dir errors; `iter_frames` order; no-camera-prefix fallback; multi-camera single-camera scope; `preprocess` end-to-end on a frame dir. **Full suite: 356 passed, 1 skipped.**

VERSION `0.1.29.1` → `0.1.30.0` (PATCH — new ingest capability).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-41"></a>room4d pull request 41: feat(geometry): zero/NaN depth safety across consumers + complete renderable GT oracle

Merged 2026-07-02 · `feature/depth-safety-and-oracle` → `develop` · 1 commit, +226 −5 in 7 files · merge commit `90b0d58`

#### What

Two related hardening items the end-to-end stress test surfaced: **depth consumers weren't robust to zero/NaN holes**, and the **dataset GT oracle was geometry-incomplete** (no world point maps, no RGB linkage — so it couldn't render). Both fixed; the oracle now flows GT-perfect geometry all the way to a rendered novel view.

#### The core hazard

**Zero depth is *finite*.** A plain `~isfinite(z)` guard lets a depth hole through and back-projects it to a spurious point at the **camera origin**. GT (and sensor) depth carries zeros/NaNs at holes, so this bites in practice.

#### How

**Shared helper — `common/depth.py` (new):** one definition of *valid depth* (`valid_depth_mask` — finite & `> near`, optional `far`) + a dense `unproject_depth_to_world` (world-from-camera, OpenCV pinhole = the VGGT convention) returning **NaN at invalid depth** with a validity mask.

**Consumer safety (defense on the consuming side):**
- `lifting/backproject.py::Backprojector`: `~isfinite(z) | z<=near` → NaN (also fixes the optimizer's `_relift_tracks`, which routes through it).
- `lifting/geometry_provider.py`: the render frame-stack's `valids` now comes from `valid_depth_mask(depth)` instead of **unconditional ones** — a zero/NaN pixel is no longer fed to the render trainer as valid supervision.
- (`render/seed.py` + fused `compose`/`scale_reconcile` were already safe — verified in the audit.)

**Producer side — complete + renderable oracle (`geometry/dataset_wrapper.py`):**
- back-projects GT depth → `pointmaps_by_frame` (NaN at holes, confidence = validity), so ground-plane + every point-map consumer treat the oracle like any other backend;
- wires `metadata['debug_image_paths']` (the GT RGB frames) so scene/render seeding finds images.

#### Validated end-to-end on spin (GT oracle, from frames)

| | before | this PR |
|---|---|---|
| oracle `pointmaps` | 0 | **produced** (finite frac 0.90, NaN holes) |
| `ground_plane_applied` | False | **True** |
| full chain → render | empty (0 gaussians) | **1.5M seed gaussians + `flythrough.mp4`** |

The whole pipeline (segment/tracking/render) runs **rc=0 consuming NaN-hole depth without a single crash** — the consumer hardening holds, and there's now a real **GT-perfect novel-view baseline**.

#### Tests

`tests/test_depth_safety.py` (4): `valid_depth_mask` excludes zero/NaN/negative + far bound; `unproject_depth_to_world` (identity pose + NaN holes); `Backprojector` zero/NaN → NaN (not origin); dataset oracle emits complete safe point maps + RGB linkage. **Full suite: 360 passed, 1 skipped.**

VERSION `0.1.30.0` → `0.1.31.0`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-42"></a>room4d pull request 42: feat(scene): enable dynamic 4D reconstruction path + G1-single/G1-multi gate split

Merged 2026-07-05 · `feature/g1-single-dynamic-4d` → `develop` · 3 commits, +175 −35 in 6 files · merge commit `bbed838`

#### What

Two things, both surfaced by the **G1-single stress test**:
1. **Design:** split the G1 gate into **G1-single** (single-stream, dynamic 4D — unlocks the back-half) and **G1-multi** (multi-stream — a parallel track). The queryable-graph→query→compiler→round-trip-eval contribution is single-stream-viable, so BLOCK 2 (P3–P7) no longer waits on multi-stream (F1/F2/F4).
2. **Fix:** make the **dynamic 4D reconstruction actually run** via the runner — *every* runner-driven render was silently coming out `kind:static`.

#### The dynamic-path bugs (all fixed)

- **`api/stages.py` (the operative one):** the scene stage's registry `run_fn` **never passed the segment `instance_artifact`** (defaulted to `None`), and `segment` wasn't a scene dependency — so the runner always handed the dynamic assembler **zero foreground**. (`pipeline.run()` passed it; the registry the runner uses did not.) Fixed: `segment` is now a scene dep + the masks are passed.
- **`scene/assembler.py` — fg `frame_index`:** `_fg_mask_by_frame` read `frame_index`, which is only built at save-time — so an **in-memory (pre-save)** segment artifact had it empty → no foreground. Now rebuilds from `observations` (mirrors `save()`).
- **`scene/assembler.py` — robustness:** `_assemble_dynamic` crashed at `quantile()` on empty foreground; now guarded (upfront `frames_with_fg<2` + try/except backstop) and **falls back to static** (the documented intent). Extracted `_assemble_static` as the shared fallback.

#### Validated (GT-oracle geometry, spin, 29 dense keyframes)

Scene `kind=dynamic` with **35,257 fg + 100,000 bg gaussians + 10 SE(3) motion bases**; render → dynamic `flythrough.mp4` + `trained_dynamic.npz`. The dynamic path runs end-to-end for the first time via the runner.

#### Honest known issue (→ next branch)

**In-pipeline masked-NVS quality is poor: mPSNR ≈ 5 vs the ~20.7 F0b target.** The reconstruction + render *run*, but held-out novel views (val cameras 1/2, covisible-masked) are geometrically off — almost certainly the **in-pipeline val-camera placement / `scene_norm`** path (F0b's ~20.7 was on windmill, likely via the bundle "gate path", not this normalization path). That's a distinct, deeper investigation and is **explicitly out of scope here** — this PR lands the enablement + gate split; the NVS-quality fix starts on a new branch after merge.

#### Tests

`tests/test_dynamic_scene.py` (3): scene depends on `segment`; `_fg_mask_by_frame` rebuilds from observations when `frame_index` empty; multi-instance union. **Full suite: 363 passed, 1 skipped.**

VERSION `0.1.31.0` → `0.1.31.1` (MICRO — bug fixes + docs; the NVS-quality payoff is the next branch).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-43"></a>room4d pull request 43: fix(scene): place held-out cameras through the ground-plane transform (dynamic NVS 5→15.5)

Merged 2026-07-06 · `feature/dynamic-nvs-quality` → `develop` · 2 commits, +63 −1 in 4 files · merge commit `4fc45b7`

#### What

Fixes the **dynamic masked-NVS** being garbage (mPSNR ~5 vs the ~20.7 target). The investigation branch after #42 confirmed this is a **held-out-camera placement bug**, not reconstruction quality.

#### Diagnosis (the decisive step)

Scoring the trained scene from its **own train cameras** gives **mPSNR 24.3** — while the held-out **val** cameras give ~5. So the reconstruction is good; the val cameras are misplaced. Root cause: the dynamic init computes `scene_norm` on the **already ground-plane-canonicalized** geometry (the provider serves canonicalized poses/points), so the bundle's `scene_norm.transfm` maps only *canonicalized → scene*. But at scoring time the val cameras are loaded as **raw GT** poses and placed via that transfm — **missing the ~45° Z-up ground-plane rotation** (verified: the transform is non-identity and absent from the bundle) → they render the wrong place.

#### Fix

`scene/assembler.py::_fold_ground_plane_into_norm` — compose the geometry's ground-plane transform (`metadata['world_frame']['transform_world_old_to_new']`) into `scene_norm.transfm` so the bundle carries the **full raw-world → scene** transform:

`full_transfm = scene_norm.transfm @ T_gp`  (convention from `world_frame.apply`: `w2c_new = w2c_old @ inv(T_gp)`)

No-op when no ground plane ran.

#### Validated (GT-oracle geometry, spin, 29 dense keyframes)

| | mPSNR | SSIM | LPIPS |
|---|---|---|---|
| held-out val, **before** | 4.99 | 0.36 | 0.85 |
| held-out val, **after** | **15.50** | 0.49 | 0.43 |
| (train-view reference) | 24.3 | 0.69 | 0.36 |

Confirmed **two ways**: an isolated orig-vs-fixed comparison on the same trained scene (15.38), and a **full in-pipeline scene→render→score re-run** with the fixed code (15.50).

**Residual gap to ~20.7** (train 24.3 → val 15.5, ~9 dB) is a separate, smaller matter — likely genuine monocular novel-view difficulty (spin is harder than F0b's windmill; SOM-family methods report ~16–19 on these iphone sequences) plus training-iteration headroom. Not another dominant bug; noted for follow-up.

#### Tests

`tests/test_dynamic_scene.py`: the composition (identity `scene_norm` ∘ `T_gp` == `T_gp`) + no-op when `world_frame` absent. **Full suite: 364 passed, 1 skipped.**

VERSION `0.1.31.1` → `0.1.31.2`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-44"></a>room4d pull request 44: fix(pose_refine): re-anchor BA gauge to input scale (dynamic NVS 4.6→11.95)

Merged 2026-07-06 · `fix/pose-refine-gauge-reanchor` → `develop` · 1 commit, +152 −1 in 5 files · merge commit `2741ed7`

#### Summary

The honest **VGGT G1-single pass** (spin, dynamic path) surfaced a bug in `pose_refine`: with BA enabled, held-out masked-NVS collapsed to **~4.6** and the background gaussian count exploded (~2.4M) despite high train-view PSNR (~26). This is a **BA gauge-scale drift**.

pycolmap BA optimizes reprojection, which is invariant to a global similarity of `{poses, points}`; with an unpinned datum the overall **scale walks freely along that null-space** (~4,000–20,000× the input depth scale). The scene assembler places gaussians from the *untouched* VGGT depth at the input scale, so refined poses at the drifted scale wreck the reconstruction. Confirmed **uniform** (per-camera radius ratios tight around the median), i.e. a pure gauge walk, not outliers.

#### Fix

`geometry/pose_refine.py::_refine_stream` — after BA, **re-anchor the refined poses onto the input-geometry frame with a single orientation-based Sim(3)** (`_sim3_src_to_dst` / `_apply_sim3_to_c2w`). Because it is a global similarity it **preserves BA's relative pose gains**, and because reprojection is invariant to it **no re-BA is needed**. Depth/point maps are untouched (already at input scale). Recovered scale is logged.

#### Result (VGGT geometry, spin, 29 keyframes, dynamic → render → masked-NVS)

| Config | val mPSNR | cam span | bg gaussians | rot vs GT |
|---|---|---|---|---|
| pose_refine **off** | 10.29 | 0.61 | ~292k | 0.88° |
| pose_refine **on**, before fix | 4.6 (degenerate) | 12886 | 2.4M | — |
| pose_refine **on**, **+ re-anchor** | **11.95** | 0.607 | ~300k | 0.615° |

Clean **+1.66 dB** pose-refinement win.

#### Diagnostics captured (in CHANGELOG)

- `refine_focal` is **not** the scale trigger (still ~4,300× with it off; and fixing focal makes poses *worse* — 0.615°→1.25° — as calibration error aliases into pose).
- Pairing BA's refined poses with the **depth-native** focal *hurts* (11.95→10.66): BA optimizes poses + shared focal jointly, so refined poses are only consistent with the refined K. `pose_refine` keeps BA's refined focal downstream (unchanged behavior, now documented in `pipeline.py`).

#### Tests

`tests/test_pose_refine.py::test_reanchor_sim3_recovers_gauge_drift` — applies a known global-similarity drift (5000× scale + rotation + translation) and asserts the re-anchor inverts it exactly. Full suite: **365 passed, 1 skipped**.

VERSION 0.1.31.2 → **0.1.31.3**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-45"></a>room4d pull request 45: feat(runner): --keyframes-all flag (dense keyframes for benchmark parity)

Merged 2026-07-06 · `feature/keyframes-all-flag` → `develop` · 1 commit, +96 −4 in 7 files · merge commit `60580be`

#### Summary

Adds an explicit **`--keyframes-all`** runner flag: use *every* frame as a keyframe. Previously the only way to approximate dense keyframe selection was to game a tiny `--keyframe-stride-sec` + a huge `--max-keyframes` — an implicit hack. This makes it a first-class, documented option.

**Motivation:** the NVS investigation needs benchmark-protocol parity. Comparing our dynamic reconstruction to SoM's per-scene DyCheck numbers (e.g. spin **17.42** mPSNR) requires training on the sequence's **full** train split, not our sparse ~29-keyframe subset — the sparse subset is itself a large NVS penalty and confounds the comparison.

#### Changes

- `data/dataset.py`: `sample_keyframes(all_frames=False)` — when True, stride 1 and no cap.
- `api/pipeline.py`: ingest passes `keyframes_all` from cfg.
- `scripts/run_single_video_room4d.py`: `--keyframes-all` arg (overrides `--keyframe-stride-sec` / `--max-keyframes`, documented in help) + `cfg['keyframes_all']`.
- `api/resume.py`: `keyframes_all` is recorded in the ingest stage signature **only when enabled** — so the default (strided) signature stays byte-identical to pre-flag artifacts (they resume unchanged), and only an actual all-frames run gets a distinct signature that correctly invalidates a strided cache. (Avoids breaking resume-from-old-artifacts, which a naive unconditional key would have done.)

#### Tests

`tests/test_keyframe_sampling.py` (new): `all_frames=True` emits every frame and ignores stride/cap (single/multi-stream + zero-frame guard); resume-signature backward-compat invariant (default == pre-flag, enabled is distinct). Full suite: **370 passed, 1 skipped**.

VERSION 0.1.31.3 → **0.1.31.4**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-46"></a>room4d pull request 46: perf(scene): cut 4D scene-assembly all-pairs tracking 8.3h→12.5min (40×)

Merged 2026-07-09 · `feature/scene-allpairs-speedup` → `develop` · 1 commit, +156 −10 in 7 files · merge commit `6a134b7`

#### Summary

4D scene assembly took **~8.3 h** on the full-frame (426f) spin oracle — a super-linear (~O(T²)) blowup in the in-assembler all-pairs point tracker (`TAPIRWrapper.compute_all_pairs`), **not** the motion init. The tracker (a faithful port of Shape-of-Motion's *offline* preproc) seeds a dense foreground grid in **every** frame, tracks each point to **every** frame, and recomputed the full-clip ResNet feature pyramid on **every** 128-point forward — ~10⁴ redundant passes over ~850k query points, ~95% of which the native lift discards (it keeps only ~`num_fg/T` per frame, at random).

Four fixes take wall-time **8.3 h → 12.5 min (~40×)** while *holding* held-out NVS quality.

#### Fixes

| lever | effect | changes tracks? |
|---|---|---|
| **A** — hoist the feature pyramid (`TAPIR.forward` gains optional `feature_grids=`; computed once per clip) | removes the redundant ResNet passes | no — 0.0px under matched RNG |
| **D** — cap tracked queries to `max_total_queries` (subsample per frame **before** tracking) | **~7× (dominant)** | yes (parity) |
| **B** — dynamic grid stride 4→8 | ~4× fewer seeds | yes (parity) |
| **E** — seed the tracker RNG | deterministic (was ~8px jitter run-to-run) | pins output |

All four ship together (A+B+D+E). A+B+E alone was an intermediate waypoint (78 min); **D** is the dominant lever that takes it to 12.5 min.

#### Validation (spin oracle, full-val n=722)

| run | scene time | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|---|
| baseline (grid4) | 8.3 h | 19.639 | 0.585 | 0.331 |
| A+B+E (grid8) | 78 min | 19.797 | 0.589 | 0.334 |
| **A+B+D+E (this PR)** | **12.5 min** | **19.857** | **0.595** | **0.325** |

Every NVS metric is ≥ baseline; `dynamic_fg` held at 35,687 (vs 35,835). Because the lift already subsamples randomly, D just does the same draw earlier and cheaper.

#### Impact — unblocked the G1-single full-frame capstone

This was the gating cost on running the single-stream dynamic-4D pipeline at full frames on *native* geometry. With it fixed, the full VGGT-Long pipeline runs in **~34 min (not ~8 h)**, which let us establish the honest full-frame native number and close the G1-single quality story:

- Held-out NVS on native geometry is **geometry-limited** (VGGT-Long chunk-stitch drift), not a dynamic-path/scene/scoring problem — proven via a GT-oracle control (0.003° residual → 19 dB on the same scorer) and shown *not* to be a repeat of the ground-plane placement bug.
- **`pose_refine` (BA) is the lever** — it cuts the drift 2.3°→0.71° (native-competitive with single-pass VGGT ~0.6°); loop closure turned out nearly redundant once BA runs.
- **Honest number: VGGT-Long full-frame + pose_refine = 12.56 mPSNR** (up +3.25 dB from the raw capstone's 9.31), which beats the sparse 29-kf number (11.95) and trails GT-oracle (~19) / SoM (17.42) by exactly the now-characterized native-geometry cost.

Net: the dynamic-4D reconstruction is no longer the risk in G1-single — it holds, it's fast, and its quality ceiling is understood and liftable.

#### Compatibility

Standalone `TAPIRWrapper` defaults unchanged (`grid_size=4`, no budget) to preserve SoM shipped-track parity. Dynamic defaults (`grid_size=8`, `max_total_queries=3*dynamic_num_fg`, `seed=0`) live on the `SingleVideoSceneAssembler` path and are overridable via `scene.tracker.*`. `TAPIR.forward`'s new arg defaults to `None` (original behavior).

VERSION `0.1.31.4 → 0.1.31.5`. Full suite: **373 passed, 1 skipped**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-47"></a>room4d pull request 47: chore(release): VERSION 0.1.31.5 → 0.2.0.0 (G1-single milestone)

Merged 2026-07-09 · `chore/release-v0.2.0.0` → `develop` · 1 commit, +18 −1 in 2 files · merge commit `76df65f`

MINOR bump to mark the **first release to `main`** and **passing the G1-single-stream gate**. No code change over `0.1.31.5` — this only moves `VERSION` and adds the `0.2.0.0` CHANGELOG release header.

Merge this to `develop` **first** (auto-creates the `v0.2.0.0` develop tag), then the `develop → main` release PR ships it (auto-creates `release-v0.2.0.0` + a GitHub Release).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-48"></a>room4d pull request 48: Release v0.2.0.0 — first release to main

Merged 2026-07-09 · `develop` → `main` · 124 commits, +28,766 −4,568 in 182 files · merge commit `b3abc90`

### room4d 0.2.0.0 — first release to `main`

The **first release** cut to `main` (which was still at "Add license", ~37 commits behind). It ships the accumulated `develop` batch and marks a milestone: **single-video (monocular) dynamic-4D reconstruction now works end-to-end on the pipeline's own reconstructed geometry.**

> This ships `VERSION` 0.2.0.0 (already on `develop`). On merge, `tag-on-release.yml` auto-creates `release-v0.2.0.0` + a GitHub Release rendered from these notes.

#### Headline

Reconstructing a **moving scene from a single video** into a renderable 4D (dynamic) representation now runs end-to-end on the pipeline's *own* geometry — not just ground-truth "oracle" geometry: **it holds, it's fast, and its novel-view quality is characterized and improvable.** That clears the way for the next phase — the **queryable 4D scene graph**: perceive a moving scene into a graph, query it, render an answer, and diff the re-perceived result against the original.

#### What's in this batch

- **Perception-stack redesign** + end-to-end robustness (lossless frame-directory ingest, zero/NaN depth safety, a complete renderable ground-truth oracle).
- **Dynamic-4D enablement** — the single-video pipeline now produces a real dynamic scene (moving foreground + rigid-motion bases), with a robust static fallback.
- **Novel-view-quality fixes:** correct held-out camera placement through the ground-plane transform (5→15 mPSNR), and a bundle-adjustment gauge-scale re-anchor in `pose_refine` (4.6→11.95).
- **`--keyframes-all`** (train on the full frame set for benchmark parity).
- **Scene-assembly speedup (#46):** the in-assembler all-pairs point tracker scaled ~O(T²); hoisting the feature pyramid + a query budget + a coarser grid + a seeded RNG took it **8.3 h → 12.5 min (~40×)**, with novel-view quality held.

#### Single-video dynamic-4D — the evidence

| aspect | result |
|---|---|
| **It holds** | a dynamic scene on the pipeline's own (VGGT-Long) geometry, full 426 frames: ~36k moving-foreground gaussians + 10 rigid-motion bases + a novel-view flythrough (not a static fallback) |
| **It's fast** | full native pipeline ~34 min (scene assembly 12.5 min, was 8.3 h) |
| **Quality is understood** | held-out novel-view quality is *limited by the reconstruction's geometry*, not the dynamic modeling — proven with a ground-truth control (near-zero alignment error → 19 dB on the same scorer). Bundle adjustment (`pose_refine`) cuts the geometry's pose drift **2.3°→0.71°**, competitive with a single-pass reconstruction. |

**Honest full-frame native number** (masked novel-view PSNR, "spin" sequence, full held-out set):

| config | geometry | frames | mPSNR |
|---|---|---|---|
| single-pass VGGT + pose_refine | 0.6° pose error | 29 keyframes | 11.95 |
| **VGGT-Long + pose_refine** | **0.71° pose error** | **426 (all)** | **12.56** |
| ground-truth oracle | 0.003° | 426 | ~19.0 |
| Shape-of-Motion reference | estimated | all | 17.42 |

Using all frames + adjusted geometry beats the sparse 29-frame number; the remaining gap to the oracle / the Shape-of-Motion reference is exactly the cost of using self-reconstructed rather than ground-truth geometry.

#### Versioning / tags

No code change over `0.1.31.5`; the MINOR bump (`0.1.31.5 → 0.2.0.0`) marks the release milestone. Per-feature detail is under the `0.1.x` CHANGELOG entries. On merge: `release-v0.2.0.0` + a GitHub Release.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-49"></a>room4d pull request 49: docs(changelog): plain-English wording for the 0.2.0.0 release entry

Merged 2026-07-09 · `chore/changelog-plain-wording` → `develop` · 1 commit, +17 −14 in 1 file · merge commit `6e95c0b`

De-jargons the `[0.2.0.0]` CHANGELOG entry — swaps internal design-doc codes (G1-single, P3–P7, FP6, BLOCK 2) for plain descriptions, keeping a parenthetical breadcrumb to the internal terms (`(internally: the G1-single gate)`) for traceability. **Docs-only; `VERSION` unchanged (0.2.0.0).**

Merge this to `develop` **before** the release PR (#48) so the release ships the cleaned CHANGELOG. (The GitHub Release notes on #48 are already plain-only.)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-50"></a>room4d pull request 50: docs(readme): refresh to the current pipeline (ships with 0.2.0.0)

Merged 2026-07-09 · `chore/readme-refresh` → `develop` · 1 commit, +87 −60 in 2 files · merge commit `263dc79`

The README was last updated at **v0.1.4.0** (97 commits ago) and still framed room4d as "intentionally a skeleton" of stubs. This refreshes it to match the shipped **v0.2.0.0** reality.

**Rewritten (stale):**
- Intro + **Status** — a working single-video dynamic-4D reconstruction pipeline (research project), with an honest **works-today / partial-scaffold / designed-not-built** split (graph relations + joint optimization are still scaffold; the queryable-graph back-half + multi-stream are ahead).
- **Pipeline** — the real 15-stage flow from `room4d/api/stages.py`.
- **Running** — `run_single_video_room4d.py` (supersedes the README's `run_single_video_vggt.py`), plus a v0.2.0.0 results table.
- Fixed a now-stale line in the reference-benchmark section ("renderer will emit … raw PSNR only" — the renderer landed and now computes masked-NVS).

**Kept (still accurate):** setup (`setup_room4d.sh`), RealSense capture, the Shape-of-Motion reference benchmark, and calibration-GT scoring.

**No version bump** — this doc was meant to ship with `0.2.0.0`; the `[0.2.0.0]` CHANGELOG entry gets a `### Docs` note instead. After this lands on `develop`, a `develop → main` merge syncs it to `main` (no new release tag, since `release-v0.2.0.0` already exists).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="room4d-pr-51"></a>room4d pull request 51: docs: sync refreshed README to main (0.2.0.0, no version change)

Merged 2026-07-09 · `develop` → `main` · 2 commits, +87 −60 in 2 files · merge commit `f4f31e2`

Syncs the refreshed **README** (and the `[0.2.0.0]` CHANGELOG `### Docs` note) from `develop` to `main`. This is the doc that was meant to ship with the `v0.2.0.0` release but landed just after it.

**Docs-only, no version change** (`VERSION` stays `0.2.0.0`). On merge, `tag-on-release.yml` fires but **no-ops** — `release-v0.2.0.0` already exists, so no new tag or GitHub Release is created; `main` simply picks up the updated README.

Diff is exactly the README refresh merged via #50:
- v0.1.4 "skeleton / stubs" framing → the current single-video dynamic-4D pipeline with an honest works-today / partial / not-yet-built status;
- the real 15-stage pipeline + `run_single_video_room4d.py` invocation + v0.2.0.0 results;
- corrected the stale "renderer will emit … raw PSNR only" line in the reference-benchmark section.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## reifyscene, first repository

### <a id="reifyscene-pr-1"></a>reifyscene pull request 1: chore: rename package room4d → reifyscene

Merged 2026-09-10 · `chore/rename-to-reifyscene` → `develop` · 1 commit, +1,022 −974 in 336 files · merge commit `d6e6af2`

"room4d" ("room-sized 4D") was a working name. The scope has grown to an **interpretable, queryable, semantically editable** 4D reconstruction, and *reify* — make something abstract concrete and manipulable — names that goal.

**Literal rename, not a rebrand.** No behavior changes.

#### Verification

| | Before | After |
|---|---|---|
| Test suite | 373 passed, 1 skipped | **373 passed, 1 skipped** |

Same single env-gated skip (`REIFYSCENE_RUN_VGGT`). Also verified: `pip install -e .` succeeds, `import reifyscene` works, `ReifyScenePipeline` resolves, and `reifyscene --help` runs.

#### Mechanical

- `room4d/` → `reifyscene/` (226 files); all 650 imports updated
- `Room4DPipeline` → `ReifyScenePipeline`
- `pyproject.toml`: distribution name, packages include, ruff exclude, CLI entry — **the command is now `reifyscene`**
- `setup_room4d.sh` → `setup_reifyscene.sh`, `run_single_video_room4d.py` → `run_single_video_reifyscene.py`, `run_calibration_room4d.py` → `run_calibration_reifyscene.py`, matching test file
- `ROOM4D_*` → `REIFYSCENE_*`
- benchmark `method_name` + its two Python fallbacks in `run_render_proof.py` — display label only, so pre-rename `reference_benchmark.json` records stay directly comparable
- default run dir → `artifacts/runs/reifyscene_dev`

#### Deliberately NOT renamed

These name state *outside* the codebase:

- **Conda envs** `room4d` / `room4d_som_ref`, their lockfiles, and all references (`gen_env_locks.sh`, `DEFAULT_CONDA_ENV`, `setup_reference_tool.sh`). Renaming means recreating the environments — separate decision.
- **Working directory** `the project folder` — planned for after this lands. This is why `tests/test_gt_scoring.py`'s `data` default and one `TODOS.md` path still say `project`.
- **Cache**: path moves to `.reifyscene_cache` so a fresh cache is built. The existing 632 MB `.room4d_cache/` stays on disk (still gitignored) until confirmed disposable. Nothing reads it any more.
- **CHANGELOG history** — prior entries describe what shipped *as* `room4d`. Rewriting them would falsify the record; the new entry says so explicitly.

#### Two over-reaches caught

The blanket substitution rewrote things it shouldn't have, both reverted:

1. **Conda env names** — would have pointed the reference-benchmark tooling at environments that don't exist.
2. **The absolute path** in `test_gt_scoring.py`, which silently turned two passing real-data tests into skips (373/1 → 371/3). Caught by comparing against the baseline rather than trusting "no failures".

VERSION `0.2.0.0` → `0.2.0.1` (MICRO: wide but mechanical `### Changed`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-2"></a>reifyscene pull request 2: fix(setup): make a from-scratch environment actually reproducible

Merged 2026-09-10 · `feat/setup-create-env` → `develop` · 10 commits, +523 −71 in 20 files · merge commit `5414911`

Building a fresh conda env to validate the rename surfaced **eight bugs**, none visible from the long-lived working environment — including that the lockfile had never described the environment it was meant to reproduce, and that nothing validated the lock it produced.

| | Old `room4d` env | New `reifyscene` env |
|---|---|---|
| Test suite | 373 passed, 1 skipped | **373 passed, 1 skipped** |

Setup now exits **0**, repeatedly; all 16 smoke tests pass including a real gsplat CUDA rasterization; `pip check` is clean apart from two pre-existing optional-dep notes the old env reports identically.

#### The lockfile did not lock the environment

Three packages were installed **after** the pinned tree with open-ended specs, so they escaped it entirely — a fresh machine resolves whatever is newest that day:

| Package | Spec | Fresh machine got | Pinned anchor wants | Old env had |
|---|---|---|---|---|
| `transformers` | `>=4.44`, absent from lock | 5.17.0 | needs safetensors ≥0.8.0 vs pinned 0.7.0 | 5.12.1 ✓ |
| `setuptools` | `--upgrade` (unbounded) | 84.0.0 | torch declares `<82` | 81.0.0 ✓ |
| `mypy` | `>=1.10` | 2.3.1 | needs `pathspec>=1`, had 0.12.1 | *not installed* |

In every case the long-lived env was correct **only by accident of when it was built**. New `pip_install_constrained` passes the lockfile as a pip constraints file; `SETUPTOOLS_MAX_SPEC` bounds the bootstrap.

**Locks regenerated:** `reifyscene` 110 → **140 pins, nothing removed** — the old lock was missing 30 packages the installer adds. Only 2 existing pins changed, both improvements (`torch`/`torchvision` now carry `+cu130`). Header Python now 3.11.16, matching what `--create-env` produces. `room4d_som_ref`: content **identical**, 261 pins — that env hasn't drifted.

#### Nothing validated the generated lock

`pip freeze` faithfully records a broken environment and emits unportable direct-URL lines for local-path installs. Both shipped here: `mypy==2.3.1` pinned beside `pathspec==0.12.1` (unresolvable), and `detectron2 @ file://` line pointing at an already-deleted build dir — the latter **caused by my own fix** making detectron2 non-editable.

`gen_env_locks.sh` now rejects `@ file://` lines and dry-run-resolves the lock before writing it. Both failure modes would now fail at generation instead of at someone's next fresh install.

#### `cmd | grep -q` is unsafe under `set -o pipefail`

`grep -q` exits at the first match and closes the pipe; the producer dies of SIGPIPE (141); pipefail reports failure **even though the match succeeded**. Two permanent — not flaky — consequences: the ffmpeg **install guard** never fired (conda reinstalled ffmpeg every run), and the ffmpeg **smoke test** could never pass (script always exited 1 at the final step). Same pattern fixed in `gen_env_locks.sh` and `setup_reference_tool.sh`.

#### The script could only run once per environment

conda's `activate.d` hooks aren't `set -u` safe — the cuda-nvcc hook expands `${NVCC_PREPEND_FLAGS}` unconditionally — so activation aborted with `unbound variable`, but only once cuda-nvcc was installed. The first run activates a bare env and *installs* the hook; every later run died. This masked the pipefail bug: **you could not re-run the script to discover that re-running was broken.**

#### detectron2: dropped from the standard install, repointed to `main`

`v0.6` (Nov 2021) pins `black==21.4b2` → `pathspec<1`, which is what made the lock unresolvable; on `main`, black is unpinned in `install_requires`. It had been compiling on every full setup since the FP5a Sapiens rung landed, **for a backend that has never existed** — `detectron2` is imported by no Python file in the repo in any commit, and `densepose_wrapper.py` reports `unavailable` unconditionally.

The flag is **retained, off by default**: DensePose is the **permissively-licensed dense top rung** — what you reach for when Sapiens is ruled out by its CC-BY-NC-4.0 licence, its gated weights, or its size — since the ladder currently drops from Sapiens' four dense modalities straight to 17 sparse Keypoint-RCNN points. The Sapiens-unavailable fallback is unaffected: `pose2d.py`'s Keypoint-RCNN rung is **torchvision**-backed, not detectron2.

*Follow-up PR:* wire a `DensePoseRung` between Sapiens and Keypoint-RCNN, add the real backend path, and resolve detectron2's `iopath<0.1.10` cap against SAM-2's `iopath>=0.1.10` (a declared conflict, not a functional one — SAM-2 uses iopath only for `g_pathmgr` and runs on 0.1.9).

#### `__version__` stale at 0.1.0 for 49 releases

`VERSION` is the single source of truth — pyproject reads it at build time, the release workflow reads it to name the tag. A hardcoded `__version__` was a fourth copy nothing kept in sync. Now resolved: VERSION file first (never stale between a bump and the next reinstall — i.e. every version-bump PR), dist metadata as fallback for a built wheel, which doesn't ship VERSION.

#### Added

- **`--create-env`** — the installer could previously only *activate* an env, so "run the setup script" silently assumed one built by hand. Opt-in by design: a missing env without the flag is a hard error naming it, so a typo in `--conda-env` fails loudly instead of silently building a multi-GB environment. `--conda-env` defaults to `reifyscene`.
- Python version read from the lockfile header. MAJOR.MINOR must match (wheels are `cp311`-tagged); a PATCH difference is reported and allowed, since the patch level isn't part of the ABI. A fresh `--create-env` produced 3.11.16 against a lockfile captured under 3.11.15 — exercising exactly that path on its first real use.
- `scripts/locks/reifyscene.lock` replaces `room4d.lock`, now removed. `room4d_som_ref.lock` keeps its name — that reference env isn't being renamed.

VERSION `0.2.0.1` → `0.2.0.2` (MICRO — dominated by fixes).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-3"></a>reifyscene pull request 3: Score native and static scenes, match photos to cameras, and make resume trustworthy

Merged 2026-09-12 · `feat/checkpoint-view-export` → `develop` · 23 commits, +8,854 −1,236 in 71 files · merge commit `81730ed`

#### Summary

This PR started as "score self-reconstructed scenes" and became the last step before the spike:
everything the scoring work turned up lands here.

1. **Native scenes can be scored** (aligned scoring, as before in this PR).
2. **Static scenes are scored correctly too** — they were rendered from misplaced cameras.
3. **VGGT's photos now match its cameras** — six places paired the model's 518² crop with the
   full frame. Native dynamic: **12.37 → 14.96** on the same geometry.
4. **A resume loads only what the run would make, and says why** — an audit of all 15 stages
   found the old check wrong both ways; it is now enforced for every stage.

#### Numbers (spin, all 722 held-out views, masked mPSNR)

| geometry | scene | before this PR | now |
|---|---|---|---|
| VGGT-Long + pose_refine (native) | dynamic | 4.55 (black renders) | **13.97** |
| VGGT-Long + pose_refine (native) | static | 11.46 (misplaced) | **13.34** |
| GT oracle | dynamic | 19.64 | **19.65** |
| GT oracle | static | — | **16.22** (fit exact) |
| Shape-of-Motion reference | dynamic | 17.42 | 17.42 |

Both native rows are one fresh run of the config with every fix in. The photo fix's own effect is
measured on one geometry before and after: **12.37 → 14.96**. Pose refinement does not repeat
exactly (GPU kernels, a multithreaded BA), and where it lands moves the score by about a dB:
13.97 here, 14.96 on the earlier geometry, both converged. Re-scoring the old checkpoints with
this PR's scorer is unchanged: oracle **19.638906478881836** (bit-identical), native 12.37.

#### What changes

##### Scoring
- Aligned scoring for self-reconstructed scenes (`align: sim3`, fitted on the training cameras,
  refused when untrustworthy, recorded with every score); sparse-keyframe time fix;
  `scripts/export_checkpoint_views.py`; `configs/experiment/spin_native_full.yaml`.
- **Static scenes** (`4c7a1ca`): same fit through the scene's training views; the default for
  every static scene, the oracle's included (its world is ground-plane rotated). The export tool
  handles static runs.

##### Photos, masks and track positions in the geometry's pixels (`ca0c7ce`)
- `geometry/image_transform.py` records the exact frame → model-pixels ops per stream (both VGGT
  loaders reproduced pixel for pixel); every geometry backend writes it.
- Static training photos, seed colours, dynamic training photos, the dynamic init's frames and
  masks, and track positions all go through it. Nothing is squashed; tracks outside the crop get
  no 3D point.
- One lift for tracking and the optimizer (depth + pose-refined camera through the crop).
- The VGGT input cache is content-keyed, so runs no longer overwrite each other's photos.

##### The graph holds things, not tracks (`a89de66`)
- Nodes are the cameras, the people and objects the perception stages found, and the background.
  A point track is evidence for the entity whose mask it sits inside on at least 2 frames and at
  least half the frames the two share, and for at most one -- the one it is inside most often.
- Each entity carries its supporting tracks, how many frames each was inside, whether the
  tracking stage called them static or moving, and **where it was per frame** (the median of its
  tracks' lifted 3D points) -- the handle a query like "where did the person go?" needs.
- Tracks inside nothing are the background's evidence; tracks inside something but never enough
  are named as ambiguous rather than swept into the room. Thresholds are `graph.*` settings.
- On spin: **13 nodes instead of 1593**; the person supported by 294 tracks (mean ratio 0.99)
  with a position on 421 of 426 frames; 411 background tracks, 72 ambiguous. Nothing read the
  graph before this (`solver.py` copies the nodes into a dict no code touches; `reifyscene/query/`
  is stubs), so the node-shape change breaks nothing downstream.

##### Pose refinement repeats (`338b669`)
- A 6-DOF pose needs 3 observed points. On spin **9 of 426 cameras had none** and 14 had fewer
  than 3 (median 204) -- and COLMAP registers them anyway, so they entered the bundle adjustment
  as free parameters and CHOLMOD reported a rank-deficient normal matrix 128-154 times per solve.
- Replaying one run's saved seed model 8 times: the worst camera landed **9.5 to 143 degrees**
  from its seed, one replay ended at 1.25e150 px, and one collapsed the scene's scale 60x
  **while its reprojection error stayed a healthy 0.88 px** -- invisible to any reproj check.
- `min_obs_per_camera` (3) refines only the cameras the data constrains (411 of 426 on spin; the
  rest keep their seed pose and are named); `ba_threads` (1) makes Ceres repeat; the divergence
  guard now measures reprojection before the re-anchor and rotations after; and the re-anchor
  covers only the cameras the solve moved -- correcting a skipped one displaced it by the drift
  it never had.
- **The tracker was never the cause**: three runs of it, three processes, days apart, wrote
  byte-identical seed models. The PR's earlier claim that GPU kernels were to blame is retracted
  in the CHANGELOG.
- Two full GPU runs of the stage are now bit-identical: one digest over all 426 poses, the same
  focal to 9 decimals, the same reprojection error to 16.
- pose_refine is contract version 2: a resume upgrades a pass-through for free and refuses a run
  that actually refined, naming pose_refine as the restart. **The four runs on disk therefore
  resume only to pose_refine** -- re-running it is ~10 min of GPU, retraining a scene is hours.

##### Resume (`a885187`, `90d6d12`)
- Every stage declares the settings its output depends on (any config section), and every
  artifact records that fingerprint, the input frames' identity (not the folder's path), a
  contract version and the upstream artifacts it was built from.
- The whole resume is checked read-only first; `--check-resume` prints the report.
- Each stage lists what a loaded artifact must carry: derivable gaps are derived (moved input or
  dataset folder, missing photos, crop box, re-lifted tracks), cheap stages are rebuilt, the
  rest is refused with the stage to restart from.
- Bugs fixed: resuming into the prior's own folder deleted it; a rejected resume had already
  copied; object embeddings were lost on every resume; missing files were skipped silently.
- All three old spin runs (two from the pre-rename paths) resume at `scene`.

#### Findings (in the CHANGELOG, not changed here)
- **Pose refinement could diverge silently** (one run: final reprojection 6·10¹⁴⁹ px, a camera
  turned 160°, passed downstream; 12.67) — refused now (`1a444fc`). It still does not repeat
  exactly run to run: seeding the tracker's shuffle is not enough (GPU kernels, multithreaded BA).
- **The dynamic scene barely beats a static one on held-out views** (13.97 vs 13.34 here; 14.96
  vs 14.56 on the earlier geometry) though it fits its training views 3-4 dB better.
- **The object labels on spin are wrong, and one node per entity makes it obvious**: four separate
  `tie`s, plus `vase`, `spoon`, `couch`. The biggest carries 397 tracks, 337 of them static -- a
  large static region, not a tie. The labels are the semantics stage's, unchanged here.
- **The static/dynamic partition disagrees with itself about the moving person**: 168 static /
  123 moving on the native run, 29 / 38 on the oracle, same clip.
- VGGT-Long's focal is 38% long in its own pixels; pose refinement's shrink is correct.
- Our VGGT-Long no-upscale loader squashes portrait frames (only below 518; unused here).

#### Commits
- earlier in this PR: `e4beef8` export tool · `a0ed009` config · `6658f93` ignore rule · `4c2f355` aligned scoring
- `4c7a1ca` fix(eval): static scenes scored through their training cameras
- `a885187` feat(resume): fingerprints, lineage, check first
- `90d6d12` feat(resume): per-stage requirements, derive or reject
- `ca0c7ce` fix(geometry): photos, masks and track positions in the geometry's pixels
- `c143d86` docs: the numbers
- `ca418ef` fix(pose_refine): BA seed model in its own folder
- `5479677` fix(optimization): artifact saved under the stage registry's name
- `d0778a4` refactor: remove evaluation code nothing ran
- `a2f8ca5` fix(resume): compare recorded settings as today's contract fingerprints them
- `19cdfec` fix(geometry): each VGGT-Long run gets its own scratch folder, removed after
- `4ddc2fb` fix(geometry): the no-upscale VGGT-Long loader crops tall frames instead of squashing them
- `1a444fc` fix(pose_refine): seed the tracker, refuse a bundle adjustment that diverged
- `bd5ccb4` fix(tracking): track every frame; no machine profile caps the frame count
- `1de17b8` fix(pipeline): a backend asked for as real fails instead of falling back; every fallback is reported
- `4edc089` fix(geometry): the GT-oracle's cameras on the ingest clock; old oracle runs re-timed on resume
- `5290f0d` docs: the seeding claim corrected, numbers from a fresh run with every fix in
- `3eacd00` docs: remove the CHANGELOG block `5290f0d` duplicated
- `a89de66` feat(graph): entities carry the point tracks that are evidence for them
- `338b669` fix(pose_refine): refine only the cameras the observations constrain, and repeat

#### Follow-ups (one commit each, in progress)
- [x] pose_refine wrote into the geometry stage's folder
- [x] optimization's stage name
- [x] delete dead code that carried the crop bug (`rendering_metrics`, `tracking_metrics`, the human-centre projection)
- [x] VGGT-Long's output cache: one scratch folder per run, removed after (it was ~11 GB, shared)
- [x] our no-upscale VGGT-Long loader crops portrait frames like upstream instead of squashing
- [x] pose refinement refuses a diverged bundle adjustment, and is seeded (it still does not repeat exactly: see Findings)
- [x] tracking is no longer capped at 96 frames (every profile tracks in windows)
- [x] fallbacks to weaker backends: fail when `real` was asked for, summarize when `auto`
- [x] the dataset (oracle) backend's cameras run on the ingest clock; old oracle runs re-timed on resume
- [x] the graph holds entities, with their point tracks as evidence (and a background node)
- [x] pose refinement repeats: the under-constrained cameras that made it wander are out of the solve

Version is decided at the end.

Validated on the GPU: a fresh run with every fix in (tracking all 426 frames, refinement
converged and unrefused, no fallbacks), and read-only resume checks of all four runs on disk --
the diverged run is now refused at pose_refine, the old oracle is re-timed and followed
downstream, the rest resume at scene.

#### Tests
Full suite: **530 passed, 1 skipped**.

#### Merge order
This is the only open PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-4"></a>reifyscene pull request 4: Version 0.3.0.0

Merged 2026-09-12 · `chore/version-0.3.0.0` → `develop` · 1 commit, +3 −3 in 3 files · merge commit `2d0f6a2`

#### Summary

PR #3 merged without a version bump, so the tag automation cut **`v0.2.1.0`** at the merge
commit. That tag has been deleted (origin and local), and this names the batch it shipped
`0.3.0.0` so the automation cuts `v0.3.0.0` in its place.

The batch: aligned scoring for native and static scenes, VGGT's photos put into its own
pixels, per-stage resume contracts with derive-or-reject, the graph holding entities with
point tracks as their evidence, and a pose refinement that repeats bit for bit.

#### What changes
- `VERSION`: `0.2.1.0` → `0.3.0.0`.
- `CHANGELOG.md`: the `## [0.2.1.0]` heading is **retitled**, not added to — nothing was ever
  released under that number, and the section's content is unchanged.
- `configs/experiment/spin_native_full.yaml`: the one reference to `v0.2.1.0` outside the
  CHANGELOG (a recorded score) moves with it.

No code changes; the suite is unaffected (530 passed, 1 skipped on the merged develop).

#### Merge order
This is the only open PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-5"></a>reifyscene pull request 5: fix(semantics): abstain on a flat ranking instead of asserting the winner

Merged 2026-09-15 · `fix/semantics-abstention` → `develop` · 3 commits, +322 −31 in 6 files · merge commit `275f716`

#### What

On `davis_lab-coat_full` the `semantics` open-vocab rung named **8 of 8** tracks — `cup`, `bird`, `handbag` — for a scene of people in lab coats behind a fence. The rung was working. It was being asked a **closed-vocabulary question**: `DEFAULT_VOCAB` is COCO-80 so the SigLIP rung stays A/B-comparable against the detector floor. With no right answer available the model still returns a winner, and nothing downstream could tell that winner from a real one.

Three defects, all in what the graph is entitled to assert.

#### 1. Record separation — and act on it only when asked

Absolute score does not separate. The junk label `handbag` scored **0.1170, the highest in the run**, above both correctly-named people. Separation does:

| track | top-1 | score | margin | **z-score** |
|---|---|---:|---:|---:|
| `:2` | **person** | 0.1151 | 0.0330 | **4.76** |
| `:3` | **person** | 0.1080 | 0.0449 | **4.71** |
| `:6` | handbag | 0.1170 | 0.0105 | 3.00 |
| `:0` | cup | 0.1020 | 0.0051 | 2.53 |
| `:7` | bird | 0.1078 | 0.0063 | 2.26 |
| `:4` | bird | 0.1077 | 0.0022 | 2.25 |
| `:5` | cup | 0.0850 | 0.0002 | 1.89 |

Every record now carries `margin` (raw top1−top2 cosine) and `margin_z` (standard deviations above the typical candidate, over the whole vocabulary), and `score_zero_shot` returns a `ZeroShotResult` carrying both plus an abstention reason.

**Abstention is off by default.** A raw gap is not a quantity a constant can be set against — it shrinks as the candidate list grows, so a threshold tuned on COCO-80 would over-abstain on a 500-word vocabulary and under-abstain on a 5-word one. That is the same defect this PR is about, and it would have shipped as the guard against it. When a caller enables the gate (`cfg['min_margin_z']`) it acts on the **z-score**, which stays comparable as the vocabulary changes size.

The z-score's limitation is **pinned by a test** rather than left to be rediscovered: it measures the winner against the *spread* of the candidates, so a bulk of near-identical candidates collapses that spread and a negligible gap reads as many sigma. Real vocabularies fan out, which is why it works on them — but a threshold is not transferable between vocabularies of different shape.

#### 2. Stop overwriting an abstention with something worse

`objects` fell through to the detector's COCO `concept` when semantics said `unknown`. `segment` is class-agnostic and emits `concept=''`, so a refusal became an **empty-string category** — indistinguishable downstream from a real label. It now resolves to `unknown`.

#### 3. Record the abstention as a value, not a gap

`SemanticRecord` gains `margin`, `margin_z`, `abstained` and `abstain_reason` (`low_separation` / `low_score` / `no_labels` / `no_concept`), so a consumer can tell **"asked and refused"** from **"never asked"**, and the rejected evidence stays auditable. Records written before this change load with the new fields defaulted.

#### Effect, measured on the shipped mechanism

Recomputing the full 80-label similarity distribution from the stored SigLIP embeddings: the two people score **z = 4.76 and 4.71**, every off-vocabulary label scores **≤ 3.00**. A `min_margin_z` anywhere in **3.0 – 4.7** keeps exactly the two people and refuses everything else.

**No default does that** — this ships the mechanism and the measurements, not a constant. Recording both statistics on every run is what lets a threshold later be chosen from data rather than guessed. The real repair is a scene-appropriate vocabulary, a rung above this one and a separate change.

#### Verification

- `539 passed, 1 skipped` (full suite; this repo has no CI, so that is a local run)
- Separation tests use realistic candidate spreads shaped like the measured lab-coat distributions; the degenerate-bulk caveat has its own regression
- `SemanticRecord` round-trip is pinned by tests, legacy records included
- Ruff delta against `develop`: **zero new findings**
- Merges cleanly: `develop` has not moved, and `git merge-tree` reports no conflict with the active spike branch

Version `0.3.0.0` → `0.3.0.1` (MICRO — bug fix).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-6"></a>reifyscene pull request 6: The queryable scene graph: Gaussians per entity, relations, query and remove, and the local app (0.4.0.0)

Merged 2026-10-16 · `spike/queryable-graph-definition` → `develop` · 79 commits, +33,696 −3,305 in 233 files · merge commit `f9520b3`

#### Summary

The queryable-scene-graph spike, merged into `develop` as one PR: 77 commits (plus 2 merges of
`develop`), version **0.3.0.1 → 0.4.0.0**. The graph's entities own Gaussians, relations hold
between them, a finished run can be queried and an entity removed, and a local app shows all of
it. The full record is the `0.4.0.0` entry in `CHANGELOG.md`.

#### What's in it

- **Entities with Gaussians:** a `handles` stage writes each entity's Gaussians onto the graph.
  Selection is FlashSplat's evidence-majority assignment followed by a floater filter.
- **Relations** between entities per frame, from VLA-3D's relation functions.
- **Queries and `remove(entity)`** over a finished run (`reifyscene query`, `reifyscene edit`).
- **The local web app** (`reifyscene serve`): graph, relations, timeline, removal, and a 3D view.
- **Training:** 8 views per step, a 1,000,000-Gaussian cap per field, and depth supervision on
  by default. Per-Gaussian identity is built and off by default.
- **Parked, opt-in:** the Sapiens body bound (`handles.body_bound`) and per-entity view coverage.
- **Evaluation tools:** DAVIS inventory scoring, an open-vocabulary detector comparison,
  ConceptGraphs on our geometry, an identity removal gate, and the edit-damage benchmark.
- **Refactors:** inputs come from records, never guesses; runs can be moved; one frame-key format.
- **Design docs** for the spike, the identity scope, graph layers and the web app.

#### Clean break

Every compatibility path is removed. Artifacts load exactly or not at all, so runs made before
this version are rebuilt or converted outside the repo, never read by compatibility code.

#### Known limitations (measured)

- **Removal leaves a visible copy of a person.** On `exp_a_lab-coat`, the floater filter puts
  back a coarse layer of large Gaussians that carries about 36% of the person's pixels. A design
  for fixed ownership at birth follows as its own PR.
- **The ten DAVIS slice runs still use the older geometry** (VGGT-Long + pose_refine), which
  leaves a "shadow" behind people; `exp_a_lab-coat` uses single-pass VGGT.
- **The body bound is parked** and has not been checked by eye.

#### Tests

Full suite on `96f5f5f`: **768 passed, 1 skipped**. The skip is the optional real-VGGT
integration test, which runs only with `REIFYSCENE_RUN_VGGT=1`. Run memory-capped (8 GB), GPU
power limit at 400 W.

#### Notes

- Merging into `develop` lets the tag workflow create `v0.4.0.0`.
- The repository will later be recreated with rewritten history (local paths and email
  removed) before it goes public; this PR's description will be exported first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-7"></a>reifyscene pull request 7: Ready to go public: one data directory, no local paths, non-commercial licence (0.5.0.0)

Merged 2026-10-17 · `chore/public-readiness` → `develop` · 3 commits, +2,097 −1,009 in 133 files · merge commit `a3fa55c`

#### Summary

Makes the repository ready to be public. Version **0.4.0.0 → 0.5.0.0**, in three commits:

1. **One data directory; nothing names a folder on one machine** (`54dabc7`)
2. **Licence: PolyForm Noncommercial 1.0.0, with third-party notices** (`8fd79e3`)
3. **Version 0.5.0.0** (`5117358`)

The full record is the `0.5.0.0` entry in `CHANGELOG.md`.

#### What's in it

**The data directory.**
- Everything the pipeline reads or writes lives in one folder, named by `REIFYSCENE_DATA_DIR`.
  Only `reifyscene/common/data_dir.py` reads that setting.
- Its layout is fixed: `datasets/`, `checkpoints/`, `runs/`, `reference/`, `cache/`, `archive/`.
- Commands and configs name things inside it: `--run-dir my_run` is `runs/my_run`, and a
  config's `dataset_root` is a name in `datasets/`. A full path is refused.
- Run records store those names, never a full path, so the whole folder can be moved or
  renamed.
- `reifyscene serve` takes no folder; it serves `runs/`.
- The pipeline's caches (VGGT inputs, VGGT-Long scratch, COLMAP work) and the TAPIR checkpoint
  move out of the repository, into the data directory.

**No local paths anywhere.**
- Mentions of folders on this machine, private notes and personal scripts are gone from the
  code, docs, the changelog's history and `TODOS.md`.
- A guard test fails if one comes back.
- Two more guards: a whole pipeline run's records hold no path of the data directory, and only
  `data_dir` reads the setting.

**Licence.**
- `LICENSE` is now PolyForm Noncommercial 1.0.0, with the line
  `Required Notice: Copyright 2026 evan-wes`.
- Code from other projects keeps its own licence, listed in `THIRD_PARTY_NOTICES.md`, with the
  licence texts in `LICENSES/`:
  - the two VGGT-derived functions move to `reifyscene/geometry/vggt_derived.py`, under the VGGT
    License;
  - the eight files ported from Shape of Motion carry its MIT notice;
  - the vendored TAPIR model stays Apache-2.0, with its one change marked;
  - FlashSplat and the Sapiens keypoint table are credited;
  - each `third_party_patches/` folder is under its upstream's licence.

**README.**
- How to set up and use the data directory.
- What works now: the queryable scene graph, queries, removal and the local app.
- The measured removal limit: on `lab-coat`, a coarse layer the floater filter keeps draws about
  a third of a removed person's pixels.
- A Licence section naming the models that are for non-commercial use only.

#### Before merging

- **Existing runs stop opening until they are converted.** Their records still hold full paths.
  Converting them is the next step on the local machine, done by a script kept outside the
  repository.
- **`vggt-long/0001-dinov2-path-cwd.patch` keeps three unchanged context lines from SALAD
  (GPL-3.0).** The notices say so; removing them would mean changing how every patch is applied.
- **`scripts/verify_reference_data.sh` still checks a folder given by path.** It is a shell
  check of downloaded data, and the README points it inside the data directory.
- **The history rewrite, before going public, must also replace `LICENSE` in every past
  commit**, with this exact file, so no published commit carries Apache-2.0.

#### Tests

- **Full suite:** 783 passed and 6 skipped, both on the final state and on the first commit
  alone. Run under an 8 GB memory cap, with the GPU power limit at 400 W.
- **The skips:** the optional real-VGGT test, plus five real-data tests that need a data
  directory.
- **Against a throwaway data directory that only links to the existing datasets and TAPIR
  checkpoint:** four of those five pass. The fifth needs a converted run and still skips.

#### Notes

- This is the only open PR.
- Merging into `develop` lets the tag workflow create `v0.5.0.0`.
- This PR's description will be exported before the repository is recreated with rewritten
  history.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-8"></a>reifyscene pull request 8: Release 0.5.0.0: the queryable scene graph, one data directory, non-commercial licence

Merged 2026-10-17 · `develop` → `main` · 128 commits, +45,208 −5,317 in 494 files · merge commit `502ef93`

#### reifyscene 0.5.0.0

Everything on `develop` since 0.2.0.0: PRs #1–#7, versions 0.2.0.1 to 0.5.0.0. `CHANGELOG.md` has
the full record of each.

##### Highlights

- **The queryable scene graph (0.4.0.0).**
  - The graph's entities (people, objects, background) own their Gaussians in the trained scene.
  - Relations between them are computed per frame with VLA-3D's relation functions.
  - A finished run can be queried with structured templates (`reifyscene query`: select, count,
    when, relate), and an entity can be removed from the scene (`reifyscene edit remove`).
  - A local web app (`reifyscene serve`) shows the graph, its relations over time and the scene
    in 3D, and runs removals.
  - Training uses 8 views per step and is depth-supervised by default.
- **One data directory (0.5.0.0).**
  - Everything the pipeline reads or writes lives in the folder `REIFYSCENE_DATA_DIR` names,
    laid out as `datasets/`, `checkpoints/`, `runs/`, `reference/`, `cache/` and `archive/`.
  - Commands and run records name things inside it, never by a path on one machine, so a whole
    data directory can be moved.
- **Licence (0.5.0.0).** reifyscene is now licensed under the PolyForm Noncommercial License
  1.0.0. Code adapted from other projects keeps its own licence; `THIRD_PARTY_NOTICES.md` lists
  every such file and the licences of what the pipeline downloads.
- **Resume loads only what a run would make (0.3.0.0).**
  - Every stage records a fingerprint: the settings it used, the input frames and the artifacts
    it was built from.
  - A resume loads a prior stage only when the fingerprint matches, rebuilds cheap stages, and
    refuses the rest with the stage to restart from.
  - `--check-resume` reports all of this without running anything.
- **Scores you can trust (0.3.0.0).**
  - Self-reconstructed scenes are scored with held-out cameras placed through the training
    cameras.
  - VGGT's photos match its cameras.
  - Pose refinement is repeatable, and a diverged bundle adjustment is refused.
  - Fixes to several cache, timing and resume bugs.
- **Semantics abstains (0.3.0.1).** The semantics stage no longer asserts a category it has no
  warrant for.
- **The rename and the installers (0.2.0.1, 0.2.0.2).**
  - `room4d` becomes `reifyscene`.
  - Both installers can create their conda environment (`--create-env`).
  - The lockfiles install what they lock.

##### Breaking changes

- **Runs made before 0.5.0.0 cannot be read.** Records now name places inside the data
  directory, and other formats were changed as clean breaks along the way. Rebuild old runs, or
  convert them outside the repository.
- **Every command takes names inside the data directory**, not paths: `--run-dir my_run` is
  `runs/my_run`, and a config's `dataset_root` is a name in `datasets/`.
- **The licence changed** from Apache-2.0 to PolyForm Noncommercial 1.0.0.

##### Known limits

- **Removing a person leaves a visible copy of them.** On DAVIS `lab-coat`, a coarse layer of
  large Gaussians that the removal's floater filter keeps draws about a third of the person's
  pixels. A fix that gives each Gaussian a fixed owner from training is designed, not built.
- Learned per-Gaussian identity is built but off by default; the body-volume bound for people is
  experimental and off.
- Not built yet: natural-language queries, edits other than removal, and multi-stream capture.

##### Tests

The tree is identical to the last commit of #7. On it, the full suite gives 783 passed and 6
skipped without a data directory set. The skips are the optional real-VGGT test and five
real-data tests. With the data directory set, those five pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-9"></a>reifyscene pull request 9: Release: LICENSE in its wrapped plain-text layout (0.5.0.0)

Merged 2026-10-17 · `develop` → `main` · 1 commit, +76 −18 in 1 file · no merge commit left: its only change, the licence file's line breaks, is in every commit of the rewritten history

#### Summary

Brings one change from `develop` to `main`, with the version unchanged at 0.5.0.0:

- `447185d` **`LICENSE` in the licence's wrapped plain-text layout.**
  - The PolyForm Noncommercial 1.0.0 terms come from the licence's plain-text download, which
    wraps lines at about 60 characters.
  - It matches the previous `LICENSE` word for word: 700 words and 37 paragraphs. Only the line
    breaks differ.
  - The `Required Notice` line is unchanged.

#### Notes

- The version is unchanged, so the release workflow finds `release-v0.5.0.0` already there and
  creates no new tag or Release.
- The history rewrite before going public puts this exact `LICENSE` into every commit, so the
  recreated `release-v0.5.0.0` will carry it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

### <a id="reifyscene-pr-10"></a>reifyscene pull request 10: Release: package metadata names its licence and author handle (0.5.0.0)

Merged 2026-10-17 · `develop` → `main` · 1 commit, +2 −2 in 1 file · merge commit `1228f77`

#### Summary

Brings one fix from `develop` to `main`, with the version unchanged at 0.5.0.0:

- **Package metadata names the licence it has, and the author by handle.**
  - `pyproject.toml` still classified the package as "OSI Approved :: Apache Software License"
    after the move to the PolyForm Noncommercial License 1.0.0, which is not OSI-approved. It
    is now "Other/Proprietary License".
  - The author is now `evan-wes`, the name the licence's copyright line uses.

#### Notes

- The version is unchanged, so the release workflow creates no new tag or Release.
- Checked with `tomllib` and a dry-run `pip install -e .`.
- The history rewrite before going public also replaces the old author name in every earlier
  version of `pyproject.toml`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# TODOS

> **Architecture roadmap → design doc.** The v1 full-pipeline roadmap (F0–P9) now lives in
> `docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md` — **that doc is
> the system of record.** Entries the design absorbs are marked `→ <phase>`; the rest are still-live
> orthogonal backlog.

## Reference Comparison Tool

### Calibration arc next: M2/M3 joint-opt fusion (person + SfM)

**What:** S1 cleared the concept-proof gates (CONTINUE-qualified, D15), so the
justified next build is real joint-opt fusion of the moving-person bridge with
SfM — promoting the spike's two throwaway proof scripts (detect+correct
robustness; metric-scale anchor) into production optimizer factors (full
reprojection + person joint BA is M3 greenfield). Named dependencies surfaced by
the spike:
- **Per-camera intrinsics injection for Panoptic** — THE scale-accuracy
  dependency (the +16% bias traced to free-estimated focal). hloc's
  `image_options` is global, so this needs per-folder imports or COLMAP DB writes
  (D12).
- **De-stub the real machinery the fusion rides on** — the sync stage and ALL
  joint-opt factors are placeholder stubs, and `pose2d.py` silently falls back to
  fabricated keypoints (calibration-grade consumers must hard-require `'real'`).
- (Lower-priority M0-hardening tooling carried over: fingerprinted SfM reuse +
  copied flock for `score_calibration_gt.py` — copy the ~20 lines, do NOT import
  from the wrapper, which must stay stdlib-anywhere, D8/D12.)

**Why:** The arc is justified but not free — M2/M3 is a genuine build comparable
in size to the renderer's step 12, to be sequenced against it (user's per-session
call).

**Superseded by D15:** the original "M1 baseline batch / person-beats-SfM"
framing is dead — on wide-baseline shared-subject SfM already wins; the value is
robustness + scale *complementarity*, not a benchmark batch. Disjoint-FOV (the
arc's real endpoint) still needs own-rig captures (no public GT).

**Effort:** L
**Priority:** P2
**Depends on:** per-camera intrinsics injection; de-stubbing sync + joint-opt factors
**→ Now scoped as F2** (real multi-stream joint-opt fusion) in the full-pipeline design doc; the
per-camera-intrinsics + de-stub-sync/factors dependencies carry over verbatim.

### Shared data-dir cache race between concurrent runs of one sequence

**What:** SoM's dataloader writes caches into the DATA dir
(`flow3d_preprocessed/cache/scene_norm_dict.pth`), so two concurrent
benchmarks of the same sequence into *different* run dirs race
writer-vs-reader on that file (torch.load of a half-written file crashes;
worst case the val split normalizes with the other process's transform —
silently skewed metrics). The run-dir flock (v0.1.3.1) does not cover this.
Also: the data dir must be writable, which nothing documents or validates.

**Why:** Residual risk split out of the lockfile work — the run-dir lock
covers same-run-dir contention but not same-data-dir contention.

**Context:** Found by the /ship adversarial review of
feature/dataset-backends; deferred when the run-dir lock shipped.

**Effort:** S
**Priority:** P3
**Depends on:** None

### Calibration arc: M0-minimal GT scorer + S1 kyotovision spike — DONE

**M0-minimal — DONE (2026-06-12, PR #8).** `room4d/calibration/gt_scoring.py` +
`scripts/score_calibration_gt.py`: score recovered extrinsics against dataset GT
with pairwise-relative errors (primary) + guarded Sim(3) APE (secondary). Turns
the calibration stage's correctness into a number.

**S1 spike — RAN 2026-06-13/14, verdict CONTINUE (qualified).** Standalone
workspace `spike_s1_calibration/` (record
`RESULTS_flute1.md`; design doc D15). On CMU Panoptic `161029_flute1`
(cams 3/12/18/22) kyotovision reproduces (1.48° median pairwise rot), but static
hloc+COLMAP MATCHES it (1.03°) — on wide-baseline shared-subject plain SfM already
solves it, so "person beats SfM" was the wrong kill-gate. D15 reframed the thesis
to **person COMPLEMENTS SfM**, fixing two blind spots, both then validated:
- **Robustness — YES** (Exp A): SfM silently misregistered the isolated cam 12 by
  ~70° with every internal signal green; SfM-vs-person disagreement detected it and
  select-and-transform corrected 35.6° → 2.33°. Inert when SfM is healthy.
- **Scale — QUALIFIED YES** (Exp B): person-anchored metric scale is consistent
  (IQR 5–10%) but +16% biased — the bias traces to SfM free-estimating focal +13%
  (not the person), i.e. it needs per-camera intrinsics (GT-intrinsic round-trip
  validates at 0.998).

Net: the arc cleared its concept-proof gates → M2/M3 (real joint-opt fusion) is a
justified build, to weigh against the renderer (Track 2) priority — user's
per-session call (see the M2/M3 entry above). Full record (D2–D15):
`a private design note`;
memory a private note. Disjoint-FOV (the arc's real endpoint) still
needs own-rig captures — no public GT exists for that regime.

## Renderer (Track 2)

Action plan / decision record: `evan-renderer-track-completion-design-20260614-2256.md`
(gstack project docs). Sequence: PR 1 (dataset geometry backend + Rasterizer
ABC + Checkpoint 0) → PR 2 (static Gaussian trainer + Checkpoints 1–2) → PR 3
(reference_benchmark_v1 eval, below) → PR 4 (dynamic branch + Checkpoint 3) →
PR 5 (CLI flythrough). Step 12 (own VGGT/CoTracker swap-in) is out of scope.

- **PR 1 landed (v0.1.5.0):** `Rasterizer` ABC + `GsplatRasterizer` backend,
  `DatasetGeometryInitializer`, `seed_cloud_from_geometry`,
  `scripts/run_render_proof.py`. Checkpoint 0 (coordinate sanity) PASSES on
  Balloon1.
- **PR 2 landed (v0.1.6.0):** static Gaussian trainer — `GaussianField`,
  `DensificationController` (our clone/split/prune, gsplat thresholds ported),
  `StaticGaussianTrainer` (Adam + L1/D-SSIM), `Rasterizer.rasterize()`
  differentiable path, `run_render_proof.py train` mode, config files. Trainer
  proven on paper-windmill (the proper static regime — Balloon1's train camera
  is a single static view). Static structure reconstructs sharply on train +
  held-out views; PSNR is motion-capped (~22/~20 dB) until the PR-4 dynamic
  branch.
- **PR 3 landed (v0.1.7.0):** covisibility-masked NVS benchmark (DyCheck
  protocol). Eval reorganized into co-located `<stage>/eval/` + a cross-cutting
  `eval/metrics/` (shared masked `MaskedPSNR/SSIM/LPIPS`, ported from SoM) +
  `eval/benchmarks/covisibility_masked_novel_view_synthesis/` (schema + compare
  + README, the single source of truth for `reference_benchmark_v1`).
  `run_render_proof.py train` emits the record over the val split and
  `--compare`s vs SoM. First numbers (room4d static vs SoM dynamic,
  paper-windmill, 419 val frames): masked mPSNR 18.51 vs 19.53 — the residual
  gap is motion, closed by PR 4. (Satisfies the former "emit
  reference_benchmark_v1" P2 item.)
- **PR 4 landed (v0.1.8.0):** dynamic Gaussian branch (SoM SE(3) motion model
  ported) — `transforms`/`motion`/`dynamic_field`/`dynamic_trainer`, the
  `entity_id` channel, `run_render_proof.py dynamic`. Checkpoint 3 PASSES:
  dynamic train-view 25.4 dB (windmill reconstructed); masked NV mPSNR 20.71
  (+2.2 dB over the static 18.51) — beats SoM on all three masked metrics.
- **PR 5 landed (v0.1.9.0):** CLI flythrough — `orbit_cameras` (near-trajectory
  arc) + `write_video` (cv2 mp4); `run_render_proof.py dynamic` emits
  `flythrough.mp4`. **Renderer track (design-doc steps 8–11) COMPLETE.** Next
  phase is step 12 (swap in room4d's own VGGT/CoTracker, one backend at a time)
  plus the two PR-4 follow-ups above (own motion init, dynamic densification).

### Follow-up: own motion initializer (replace the borrowed SoM init)

**What:** PR 4's dynamic init is borrowed from SoM (`scripts/borrow_som_motion_init.py`
runs SoM's cuML-clustering + Procrustes track init in the reference env and dumps
a bundle). Replace it with a native room4d `MotionInitializer` behind the existing
`render/gaussian4d/motion_init.py` seam. Split into two PRs:

- **PR-A — track-lifting bridge: DONE (v0.1.11.0).** `room4d/lifting/native_track_lift.py`
  ports SoM's lift natively (TAPIR parse, scene-norm, `get_tracks_3d`,
  `get_bkgd_points`) onto `DatasetGeometryInitializer`, from aligned inverse-depth
  + COLMAP-refined cameras. Reproduces the borrowed frame on paper-windmill to
  <1% (rotation 3.9e-8, scale 0.4%, track count 0.1%). Validated by
  `scripts/check_native_lift.py`. Only the 2D tracker output is still borrowed.
- **PR-B — clustering + Procrustes + init: DONE (v0.1.12.0).**
  `render/gaussian4d/procrustes_init.py` ports the clustering (sklearn KMeans on
  velocity dirs, no cuML) + per-cluster/per-frame weighted Procrustes +
  `init_fg_from_tracks_3d` / `init_bg`; `NativeMotionInitializer` returns the same
  `(DynamicScene, cameras)` shape as `load_borrowed_init`. Wired as
  `run_render_proof.py dynamic --native` (+ `scripts/native_motion_init.py` npz
  dump). Validated at the full SoM schedule (iters=5000): native masked mPSNR
  **20.74** vs borrowed **20.68**. The only borrowed input left is the 2D tracker
  output (own tracker = separate future work).

**Why:** the borrow is a stopgap; the dynamic init is now room4d's own, like
every other stage. **Status:** DONE.

### Follow-up: own 2D point tracker — DONE (v0.1.13.0)

room4d's own tracker now produces the 2D tracks the native motion init consumes —
the last borrowed input in the motion-init path is closed. `room4d/tracking/tapnet/`
vendors the DeepMind TAPIR torch model byte-for-byte and `TAPIRWrapper` (a real
`PointTracker` backend, sibling to the CoTracker wrapper) faithfully ports SoM's
`compute_tracks_torch.py` all-pairs FG-seeded loop to `(x, y, occ, dist)`.
`TrackArtifact` was extended (additively) with a first-class `AllPairsTracks`
payload + SoM `2d_tracks/{i}_{j}.npy` save/load; `lift_tracks_3d` and friends gained
a `tracks_dir` override, and `run_render_proof.py dynamic --own-tracker` (or
`scripts/compute_tracks_native.py`) runs the whole init with no borrowed tracks.
On paper-windmill the native tracker reproduces the shipped FG seeding exactly
(1739/1739 diagonal coords); e2e masked mPSNR **20.70** (mSSIM 0.668, mLPIPS 0.161)
matches the borrowed/native quality bar (borrowed 20.68 / native-borrowed-tracks
20.74). The remaining dataset-provided inputs (RGB/depth/poses) are dataset ground
truth, not borrowed reconstruction.

### Step 12: native geometry (VGGT-Long) → real renderer — pose rung DONE

Replace the borrowed DyCheck geometry (depth/poses; masks deferred) with room4d's
own VGGT-Long reconstruction, scored against the DyCheck-GT oracle (masked mPSNR
~20.75 on paper-windmill), one input at a time.

- **PR 2 — geometry-provider seam: DONE (v0.1.14.0).** `room4d/lifting/geometry_provider.py`
  (`GeometryProvider` ABC + `DyCheckGeometryProvider`, the permanent oracle);
  `build_native_init_inputs`/`NativeMotionInitializer` take an optional `provider`.
  Byte-identical to the old direct path (lossless refactor).
- **Pose rung — DONE (diagnosed + fix productionized).** Native VGGT-Long poses
  (GT depth) scored **17.01** vs oracle **20.75** — a −3.7 dB gap diagnosed as a
  POSE problem (spike R0/R1 ladder, memory a private note).
  VGGT-Long itself was fixed first — global pose alignment (v0.1.15.2,
  chunk-local→`camera_poses.txt`, 20.5°→1.73°) + loop closure (v0.1.15.3).
  Photometric pose-opt (Path A) failed; COLMAP-BA pose refinement (Path B)
  recovered **17.01→19.58** (windmill 1.73°→0.26°), productionized as the
  **`pose_refine` pipeline stage (v0.1.16.0)** — robust BA, refine focal, pp fixed,
  no FG mask needed (ablation: masking not load-bearing).
- **Remaining rungs (lower priority — poses were the dominant gap):** native DEPTH
  (explored in spike R3, not productionized) and native FG masks (needs a
  segmenter, none wired in). Score each vs the oracle when picked up.
- **The big convergence** — making the *production* `Room4DPipeline` use the trained
  renderer end-to-end — is its own headline item below.

### Productionize the trained renderer into Room4DPipeline — → F0 (design doc)

**Superseded by the full-pipeline design.** This is now **F0 (renderable core)** in
`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`: delete the
no-train `Gaussian4DTrainer`/`Gaussian4DScene` toy, wire the proven `Static/DynamicGaussianTrainer`
into the render stage (`SceneArtifact→seed` adapter, static/dynamic select, training-config home),
clean the scene↔render boundary, and emit in-pipeline masked-NVS. The open design forks (trainer
init, trainer selection, config home, resume/stage-window, the weak BenchmarkRunner metric) are
carried into the design doc's §5d. **F0 is the first build phase.**

### Follow-up: dynamic densification — DONE (v0.1.10.0, PR 6)

Standard 3DGS adaptive density control now runs on the dynamic fg + bg fields
(`DensificationController` generalized with `param_names`/`group_prefix`/telemetry
slicing; `DynamicGaussianTrainer` runs one controller per field). On
paper-windmill: Gaussians grow ~137k → ~970k, masked mLPIPS 0.178 → 0.164,
mSSIM 0.661 → 0.669, train-view PSNR ~25 → 27.2, sharper flythrough. The
remaining renderer-track follow-up is the native motion initializer (above).

## Open cleanup / parked

### 2D-tracks O(N²) files antipattern

**What:** the native tracker / `TrackArtifact` write one `.npy` per frame-pair
(`2d_tracks/{i}_{j}.npy`) — 181k files at 426 frames, ~1M for long sequences.
Replace with a single packed/memmapped array. Only the `--own_tracker` path emits
them (Path B / pose_refine uses a single VGGSfM npz), so not urgent.

**Effort:** S **Priority:** P3 **Depends on:** None (fix before own-tracker on a long seq)

### Benchmarker on real (in-pipeline) code — PARKED

**What:** a GT-aware benchmark runner with a composite input provider (per-slot
GT/native sourcing) that scores masked-NVS vs the oracle — the native-swap ladder's
permanent measuring instrument, replacing the pipeline's weak train-view render
metric. The masked-mPSNR machinery already exists (`eval/metrics/image.py`,
`render/eval/novel_view.py`, the `covisibility_masked_novel_view_synthesis` record
contract) but is only driven by `run_render_proof.py`.

**Why parked:** intentionally deferred until the trained renderer is in the pipeline
(see "Productionize the trained renderer") — there is no real in-pipeline render to
benchmark until then.

**Effort:** L **Priority:** P2 **Depends on:** trained renderer in Room4DPipeline
**→ Now folded into** the design doc's eval: in-pipeline masked-NVS lands in F0; the round-trip
graph-diff eval is P7. The "permanent measuring instrument" role is the design's eval stage.

### develop → main release merge

**What:** `main` is at v0.1.1.0; `develop` is at v0.1.16.0 (~45 commits). Cut a
`develop`→`main` release merge when a batch is ready — a per-batch call, not
per-feature.

**Effort:** S **Priority:** P2 **Depends on:** None

### Optional: confirm Path B render gain on spin

One render-proof run scoring the refined poses on the dynamic spin scene (like
windmill 17.01→19.58). Nice-to-have, not blocking.

**Effort:** S **Priority:** P3 **Depends on:** None

## Completed

### Proper resume handling in run_reference_benchmark.py

**What:** Make resume of an interrupted run recall the original schedule and
refuse zero-epoch completion stamps.

**How it landed:** Resume reads cfg.yaml's `num_epochs` and passes it back
explicitly (SoM never re-reads it); a missing schedule fails closed; the
checkpoint's epoch is read via the reference env (`weights_only=True`,
120s timeout) and a resume with zero epochs left aborts before training.
`--num-epochs 500` and plain runs share the canonical `default` label; the
sentinel is written atomically; resumed runs record
`resumed`/`resumed_from_epoch` in the result JSON. GPU-verified (cfg-bump
resume trained 2→4 and stamped "4").

**Completed:** v0.1.3.1 (2026-06-12)

### Lockfile guard against concurrent runs sharing a run dir

**What:** Exclusive per-run-dir lock for the duration of training + harvest.

**How it landed:** `fcntl.flock` held on an fd for the wrapper's lifetime —
kernel-owned, releases on any death (kill -9 included), no PID/reclaim
protocol to race (the hand-rolled O_EXCL+PID design was rejected in review
for TOCTOU reclaim races). The training child inherits the fd via
`pass_fds`, so an orphaned trainer keeps the run dir locked until it exits.
Live-verified: held lock refuses a second invocation; released lock
proceeds. Residual same-data-dir cache race split into its own P3 entry.

**Completed:** v0.1.3.1 (2026-06-12)

### Download NVIDIA Dynamic Scenes dataset + extend benchmark wrapper

**What:** Download the processed NVIDIA Dynamic Scenes data (Drive link in
`scripts/verify_reference_data.sh`) and extend `scripts/run_reference_benchmark.py`
beyond iphone-format (`data:nvidia` plumbing).

**How it landed:** Balloon1 + Jumping downloaded manually and verified
(layout, per-frame consistency, full N² pairwise track sets). The wrapper
gained `--dataset {iphone,nvidia}`; nvidia metrics are harvested from the
trainer's final-epoch TensorBoard `val/*` scalars (SoM has no
`evaluate_nvidia.py`). Proven on GPU with a 5-epoch Jumping smoke run.
Note: the processed download ships 4 of the rig's 12 cameras with a fixed
1-train/3-val split — the N=2/4/8 input sweep from the original "Why" needs
custom splits later (Track 2).

**Completed:** v0.1.3.0 (2026-06-11)

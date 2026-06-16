# TODOS

## Reference Comparison Tool

### Post-spike: calibration M0 hardening + M1 baseline batch

**What:** Conditional on S1's CONTINUE verdict: fingerprinted SfM reuse +
copied flock for `score_calibration_gt.py` (design D8/D12 — copy the ~20
lines, do NOT import from the wrapper, which must stay stdlib-anywhere);
per-camera GT-intrinsics injection for Panoptic (per-folder imports or
COLMAP DB writes — hloc's `image_options` is global, D12); batch runner
over certified wide-baseline configurations with the overlap-certification
report (M1).

**Why:** The deferred half of the eng-reviewed M0/M1 scope; deliberately
gated so a kill verdict strands nothing.

**Context:** Design doc M0-HARDENING + M1 sections. **If S1 returns KILL,
delete this entry** (that condition is the point of it).

**Effort:** M
**Priority:** P2
**Depends on:** S1 spike verdict (see the M0-minimal + S1 entry below)

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

### Calibration arc: M0-minimal GT scorer + S1 kyotovision spike

**What:** (a) M0-minimal — `room4d/calibration/gt_scoring.py` +
`scripts/score_calibration_gt.py`: score recovered extrinsics against
dataset GT with pairwise-relative errors (primary) and guarded Sim(3) APE
(secondary). (b) S1 — standalone spike: run the kyotovision IROS'22
moving-person calibration code on CMU Panoptic wide-baseline subsets,
score with M0-minimal, deliver an explicit kill/continue verdict for the
calibration arc.

**Why:** M0 turns the calibration stage's correctness into a number; S1 is
the kill gate for the pipeline-as-rig-calibrator vision (entity bridges vs
static SfM) — cheap to falsify before any deeper investment.

**Context:** Full decision record (D2–D12, incl. outside-voice round) in
`a private design note`.
S1 needs the Panoptic toolbox download + kyotovision checkout. A Panoptic
PASS validates wide-baseline only, NOT disjoint-FOV (no public GT exists
for that regime — own-rig captures later).

**Effort:** M0-minimal S (built 2026-06-12); S1 M (downloads + runs + memo)
**Priority:** P1
**Depends on:** None

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
a bundle). Implement a native room4d `MotionInitializer` backend (lift borrowed
2D tracks, cluster without cuML, Procrustes) behind the existing
`render/gaussian4d/motion_init.py` seam.
**Why:** the borrow is a stopgap; the dynamic init should be room4d's own, like
every other stage. **Priority:** P2.

### Follow-up: dynamic densification — DONE (v0.1.10.0, PR 6)

Standard 3DGS adaptive density control now runs on the dynamic fg + bg fields
(`DensificationController` generalized with `param_names`/`group_prefix`/telemetry
slicing; `DynamicGaussianTrainer` runs one controller per field). On
paper-windmill: Gaussians grow ~137k → ~970k, masked mLPIPS 0.178 → 0.164,
mSSIM 0.661 → 0.669, train-view PSNR ~25 → 27.2, sharper flythrough. The
remaining renderer-track follow-up is the native motion initializer (above).

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

# End-to-end pipeline run — integration findings (spin, geometry-backend matrix)

**Date:** 2026-07-02 (runs 2026-07-01→02) · **Data:** `shape_of_motion/iphone/spin` (426 frames, 720×960)
· **Driver:** `scripts/run_single_video_room4d.py`, incremental stage windows, GPU-guarded (87 °C watchdog).

## Purpose

First full end-to-end run of the pipeline on real data after FP6b completed the geometry backend
(design §7: "the end-to-end run is what finds these" — the F4/G1 discovery mechanism). Goal: prove
the chain runs and produces a rendered novel view, and surface integration gaps — **not** quality.
Varied the **geometry backend** (the axis FP6a/FP6b changed), holding perception/render at defaults.

Spin is a frame directory, so it was encoded to a **lossy mp4** (no ffmpeg → cv2 `mp4v`) because the
pipeline ingests videos. That encoding choice turned out to be load-bearing (see combo B).

## Result matrix (`ingest → … → render`, per geometry backend)

| Combo | Geometry backend | Outcome |
|---|---|---|
| **A** | `vggt` (joint real) | **Full end-to-end success** → `render/flythrough.mp4` (60-frame novel view), `trained_gaussians.npz`. Every stage ran; resume-from-prior worked. |
| **B** | fused `colmap`+`moge` (FP6b) | **Runs but not pipeline-robust.** MoGe (depth) reliable on every frame; COLMAP (pose) **erratic** on the lossy frames (0–71 registered across configs) → composer correctly degrades to `depth_only` / a 3-frame fused set. |
| **C** | `dataset` (GT oracle) | **Incompatible with video ingest** → graceful-empty geometry (no crash). Ingest keyframe `stream_id` is `None`, so the backend looks for `None_00060.png` not `0_00060.png`. |

## Gap ledger

Severity: **P1** blocks a real end-to-end result · **P2** limits fidelity/robustness · **P3** polish.

| # | Gap | Sev | Root cause | Proposed fix | Status |
|---|---|---|---|---|---|
| 1 | Runner `stage_labels` KeyError mid-run | P1 | Hardcoded display map drifted from `STAGE_ORDER` (missing `dyn_mask`/`segment`/`reid`/`semantics`; stale "/11") | Derive labels from `STAGE_ORDER` | **FIXED** (this PR — trivial, display-only) |
| 2 | Fused (COLMAP) geometry not pipeline-robust | P1 | COLMAP unreliable on **lossy** video frames (validators used lossless PNGs + fixed intrinsics) + the intermittent COLMAP instability from FP6b steps 2–3 | Lossless/high-quality frame path for SfM; make intrinsics-prior + density first-class fused config; carry the step-3 health-check gate | OPEN (needs sign-off — behavior/design) |
| 3 | No dataset / frame-dir ingest path | P1 | Pipeline ingests videos (`cv2.VideoCapture`); datasets are frame dirs; `stream_id=None` from video ingest breaks GT-name mapping | A dataset/frame-sequence ingest adapter (stream_id + frame-name alignment) | OPEN (needs sign-off — design) |
| 4 | Runner CLI can't select fused/`moge`/`colmap` geometry | P2 | `--geometry-backend-name` choices = `vggt`/`vggt-long`/`dataset` only; FP6b path is config-only | Add pose/depth-backend CLI flags (or a `fused` choice) | OPEN (needs sign-off — interface) |
| 5 | `optimization` (F2) is a stub | P2 | Factors return no-op; "connect to a real optimizer backend" | F2 = calibration-arc M2/M3 (already sequenced) | KNOWN (roadmap F2) |
| 6 | `graph` is the passive builder, not queryable | P2 | "TODO: replace per-stream track nodes with cross-view associated entities" | P3 queryable graph (gated on G1) | KNOWN (roadmap P3) |
| 7 | Scene is static-only on few keyframes | P3 | Default keyframe sampling gave 4 frames → no dynamic reconstruction of the moving subject | Denser/motion-aware keyframes for a *meaningful* (vs "it runs") result | OPEN (tuning) |

## What's proven

- **The chain runs end-to-end on real data and renders a novel view** (combo A) — the observable-
  degradation machinery, the new `dyn_mask` stage (all-static default), the FP6b manifests, resume,
  and the gaussian4d render all integrate.
- **The ladder's graceful degradation holds** — combos B and C degraded (`depth_only`, empty) instead
  of crashing, exactly as designed (§4A invariants).
- **MoGe depth is pipeline-ready.** The FP6b weak point in the real pipeline is **COLMAP robustness**,
  which is a *known* theme (steps 2–3) now confirmed to bite through the ingest path.

## Recommended next (a focused FP6b-hardening pass, post this PR)

Gaps 2 + 3 are the same root story (SfM needs clean frames the video-ingest path doesn't give) and are
the highest-leverage next work: a **lossless frame-sequence ingest adapter** + **intrinsics-prior/density
as first-class fused config** would make both combo B (fused) and combo C (dataset) viable end-to-end.
Gap 4 (CLI) rides along. F2/P3 (gaps 5–6) remain their own roadmap items.

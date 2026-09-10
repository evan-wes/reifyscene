"""GT-free health-check for SfM camera trajectories (FP6b step 3).

Step 2 found monocular COLMAP is **intermittently degenerate** on low-parallax dynamic
clips: incremental BA occasionally lands a geometrically-wrong reconstruction that is still
100% registered + low reproj error, so neither reproj-error nor the scale-reconciliation
residual catches it — it fuses **silently**. This makes the failure **observable**.

Signal: **consecutive pairwise-relative camera rotation**, computed **per stream** (a
handheld camera moves continuously, so adjacent sampled frames rotate little; a degenerate
draw injects an implausible jump — the step-2 bad spin draw hit 39° between adjacent frames).
Two design points from the step-2 discussion:
- **Per-stream** — never compare across streams, where large inter-camera angles are genuine
  (a global consecutive-pairwise check would false-positive on any multi-cam rig).
- **Self-calibrating** — flag against the stream's *own* median + k·MAD (plus an absolute
  physical cap), so it adapts to fast vs slow motion instead of a brittle fixed threshold.

Caveat (documented): this catches the **outlier-jump** degenerate type, not a
globally-bent-but-locally-smooth one — no cheap pose-only or reconciliation-residual signal
does. It is a partial, observable guard, not a proof of correctness.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def assess_pose_health(
    camera_states_by_stream: dict[str, list],
    *,
    abs_cap_deg: float = 30.0,
    mad_k: float = 6.0,
    min_frames: int = 4,
) -> dict[str, Any]:
    """Assess per-stream trajectory plausibility from estimated poses (GT-free).

    Returns a JSON-safe dict: ``healthy`` (all streams pass), ``per_stream`` stats
    (median/max/MAD/bound of consecutive pairwise-relative rotation + ``flagged``), and a
    human-readable ``reason``. A stream is flagged when its **max** consecutive pairwise
    rotation exceeds ``max(abs_cap_deg, median + mad_k·MAD)`` — an implausible jump relative
    to both physics and that stream's own motion.
    """
    per_stream: dict[str, Any] = {}
    reasons: list[str] = []
    healthy = True
    for stream_id, states in (camera_states_by_stream or {}).items():
        ordered = sorted(states, key=lambda s: float(s.time_sec))
        if len(ordered) < int(min_frames):
            per_stream[stream_id] = {"n": len(ordered), "flagged": False, "reason": "too-few-frames"}
            continue
        rots = [np.asarray(s.pose.T_world_cam, dtype=np.float64)[:3, :3] for s in ordered]
        angles = np.asarray([_geodesic_deg(rots[i], rots[i + 1]) for i in range(len(rots) - 1)])
        med = float(np.median(angles))
        mx = float(np.max(angles))
        mad = float(np.median(np.abs(angles - med)))
        bound = float(max(abs_cap_deg, med + mad_k * max(mad, 1e-6)))
        flagged = mx > bound
        per_stream[stream_id] = {
            "n": len(ordered), "median_deg": med, "max_deg": mx, "mad_deg": mad,
            "bound_deg": bound, "flagged": flagged,
        }
        if flagged:
            healthy = False
            reasons.append(f"{stream_id}: max pairwise rot {mx:.1f}° > bound {bound:.1f}° (median {med:.2f}°)")
    return {
        "healthy": healthy,
        "per_stream": per_stream,
        "reason": "; ".join(reasons),
        "note": "GT-free per-stream outlier-jump check; does not catch globally-bent draws.",
    }

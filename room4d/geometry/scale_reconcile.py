"""COLMAP ↔ MoGe metric-scale reconciliation (FP6b step 2, design §4A.1).

Monocular COLMAP egomotion is **scale-free** (a 7-DOF Sim(3) gauge); MoGe-2 depth is
**metric**. When the pose rung is COLMAP and the depth rung is MoGe, the two channels
live at different scales and cannot be fused until a **single global scale** ties them
together. This module solves that scalar.

**The solve.** COLMAP's sparse reconstruction gives, per registered frame, a set of 3D
points with 2D reprojections. Each such point has a COLMAP camera-space depth ``z_colmap``
(scale-free) and, at its reprojection pixel, a MoGe **metric** depth ``z_moge``. Their
ratio ``z_moge / z_colmap`` estimates the COLMAP→metric scale; a robust median over **all**
sparse-point observations across all frames gives one global scale ``s``. The per-frame
spread of that ratio is the **residual** the design hands to F2 (person-anchored per-frame
scale), *not* fixed here.

**Why this is gauge-free (and can run before ground-plane, design §4A.1).** The ratio is a
per-frame *camera-space* depth comparison, independent of the world rotation / floor height
the ground-plane canonicalizer sets. So reconciling scale first — then canonicalizing on a
now scale-consistent reconstruction — recovers the same ``s`` while sparing the ground-plane
estimator a mix of metric (MoGe) and scale-free (COLMAP) inputs. Scale is exactly the one
gauge DOF the ground plane does not pin; this owns it.

The pure solver (:func:`solve_global_scale`) is COLMAP-free and unit-tested on synthetic
observations; the thin :func:`extract_colmap_observations` reads a persisted sparse model
via ``pycolmap`` (probe-gated by the ladder, like the COLMAP rung itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScaleReconcileResult:
    """Outcome of a COLMAP↔MoGe scale solve.

    ``scale`` multiplies COLMAP world coordinates (and camera centers) to bring them
    into MoGe's metric units. ``status`` is ``'reconciled'`` when solved from real
    observations, or ``'identity_fallback'`` (scale ``1.0``) when there was nothing to
    solve from (no sparse model / too few valid depth samples) — never a crash.
    """

    scale: float = 1.0
    residual_median: float = 1.0
    residual_p10: float = 1.0
    residual_p90: float = 1.0
    n_obs: int = 0
    n_frames: int = 0
    per_frame_residual: dict[str, float] = field(default_factory=dict)
    status: str = "identity_fallback"
    reason: str = ""

    def summary(self) -> dict[str, Any]:
        """JSON-safe dict for ``GeometryArtifact.metadata`` / manifest notes."""
        return {
            "scale": float(self.scale),
            "residual_median": float(self.residual_median),
            "residual_p10": float(self.residual_p10),
            "residual_p90": float(self.residual_p90),
            "n_obs": int(self.n_obs),
            "n_frames": int(self.n_frames),
            "status": self.status,
            "reason": self.reason,
        }

    def note(self) -> str:
        """One-line human-readable summary for artifact notes."""
        if self.status != "reconciled":
            return f"scale reconciliation: {self.status} (s=1.0) — {self.reason}"
        spread = f"[{self.residual_p10:.3f}, {self.residual_p90:.3f}]"
        return (
            f"COLMAP↔MoGe scale reconciled: s={self.scale:.4g} from {self.n_obs} obs over "
            f"{self.n_frames} frames; per-frame residual {spread} (→F2)."
        )


def solve_global_scale(
    obs_by_key: dict[str, list[tuple[float, float, float]]],
    depth_by_key: dict[str, np.ndarray],
    *,
    min_obs: int = 32,
    mask_by_key: dict[str, np.ndarray] | None = None,
    max_ratio: float = 1e4,
) -> ScaleReconcileResult:
    """Solve the single global COLMAP→metric scale from sparse reprojections + MoGe depth.

    Parameters
    ----------
    obs_by_key
        ``{frame_key: [(u, v, z_colmap), ...]}`` — per-frame sparse-point observations,
        ``(u, v)`` in pixels, ``z_colmap`` the point's COLMAP camera-space depth (>0).
    depth_by_key
        ``{frame_key: HxW metric depth}`` — MoGe's metric depth per frame.
    min_obs
        Minimum number of *valid* (in-bounds, finite, positive) ratio samples required
        to trust a solve; below this the result is an identity fallback (``s=1.0``).
    mask_by_key
        Optional ``{frame_key: HxW validity}`` (MoGe's confidence/validity mask); a pixel
        is used only where the mask is truthy (``>0.5``).
    max_ratio
        Guard: ratios outside ``[1/max_ratio, max_ratio]`` are discarded as degenerate.

    Returns
    -------
    ScaleReconcileResult with the global ``scale`` (median of all valid ratios), the
    per-frame residual distribution, and provenance counts.
    """
    all_ratios: list[float] = []
    per_frame_ratios: dict[str, list[float]] = {}
    for key, obs in obs_by_key.items():
        depth = depth_by_key.get(key)
        if depth is None or not obs:
            continue
        depth = np.asarray(depth)
        h, w = depth.shape[:2]
        mask = None if mask_by_key is None else mask_by_key.get(key)
        for u, v, z_colmap in obs:
            if not (z_colmap > 0.0):
                continue
            iu, iv = int(round(u)), int(round(v))
            if iu < 0 or iv < 0 or iu >= w or iv >= h:
                continue
            z_moge = float(depth[iv, iu])
            if not np.isfinite(z_moge) or z_moge <= 0.0:
                continue
            if mask is not None and not (float(np.asarray(mask)[iv, iu]) > 0.5):
                continue
            ratio = z_moge / float(z_colmap)
            if not np.isfinite(ratio) or ratio <= 0.0 or ratio > max_ratio or ratio < 1.0 / max_ratio:
                continue
            all_ratios.append(ratio)
            per_frame_ratios.setdefault(key, []).append(ratio)

    n_obs = len(all_ratios)
    if n_obs < int(min_obs):
        return ScaleReconcileResult(
            scale=1.0, n_obs=n_obs, n_frames=len(per_frame_ratios),
            status="identity_fallback",
            reason=f"only {n_obs} valid depth-ratio samples (< min_obs={int(min_obs)})",
        )

    ratios = np.asarray(all_ratios, dtype=np.float64)
    scale = float(np.median(ratios))
    per_frame_residual = {
        key: float(np.median(r) / scale) for key, r in per_frame_ratios.items() if r
    }
    resid = np.asarray(list(per_frame_residual.values()), dtype=np.float64)
    return ScaleReconcileResult(
        scale=scale,
        residual_median=float(np.median(resid)) if resid.size else 1.0,
        residual_p10=float(np.percentile(resid, 10)) if resid.size else 1.0,
        residual_p90=float(np.percentile(resid, 90)) if resid.size else 1.0,
        n_obs=n_obs,
        n_frames=len(per_frame_ratios),
        per_frame_residual=per_frame_residual,
        status="reconciled",
        reason="",
    )


def extract_colmap_observations(
    model_dir: str | Path,
    index_to_key: dict[int, str],
) -> dict[str, list[tuple[float, float, float]]]:
    """Read sparse-point reprojections from a persisted COLMAP model via ``pycolmap``.

    Each registered image is named ``{index:06d}.png`` by the COLMAP rung, so its name
    stem maps back through ``index_to_key`` to the shared ``FrameRef`` key MoGe depth is
    stored under. For every 2D point with a triangulated 3D point, returns the pixel
    ``(u, v)`` and the point's COLMAP camera-space depth ``z``.

    Returns ``{}`` on any failure (missing ``pycolmap``, unreadable model) — the pure
    solver then produces an identity fallback. Kept thin + probe-gated: no numpy-heavy
    work, just the read.
    """
    try:
        import pycolmap  # noqa: PLC0415 — optional dep, probe-gated by the ladder
    except Exception:
        return {}
    model_dir = Path(model_dir)
    try:
        rec = pycolmap.Reconstruction(str(model_dir))
    except Exception:
        return {}

    out: dict[str, list[tuple[float, float, float]]] = {}
    points3d = rec.points3D
    for im in rec.images.values():
        try:
            idx = int(Path(im.name).stem)
        except (ValueError, TypeError):
            continue
        key = index_to_key.get(idx)
        if key is None:
            continue
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        r = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64).reshape(3)
        samples: list[tuple[float, float, float]] = []
        for p2d in im.points2D:
            has = p2d.has_point3D
            if not (has() if callable(has) else has):
                continue
            pt = points3d.get(int(p2d.point3D_id)) if hasattr(points3d, "get") else points3d[int(p2d.point3D_id)]
            if pt is None:
                continue
            xyz = np.asarray(pt.xyz, dtype=np.float64).reshape(3)
            z = float((r @ xyz + t)[2])
            xy = np.asarray(p2d.xy, dtype=np.float64).reshape(2)
            samples.append((float(xy[0]), float(xy[1]), z))
        if samples:
            out[key] = samples
    return out

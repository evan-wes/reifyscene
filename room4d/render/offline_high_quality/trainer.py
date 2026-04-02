"""Offline high-quality renderer with a NeRF-refinement hook."""

from __future__ import annotations

from room4d.render.gaussian4d.trainer import Gaussian4DTrainer


class OfflineHighQualityTrainer(Gaussian4DTrainer):
    """Higher-quality layered renderer intended for slower offline passes."""

    backend_name = "offline_high_quality"

    def __init__(self, cfg: dict | None = None) -> None:
        merged = dict(cfg or {})
        merged.setdefault("quality_mode", "offline_high_quality")
        merged.setdefault("debug_triptych", True)
        merged.setdefault("offline_high_quality", dict(merged.get("offline_high_quality", {}) or {}))
        merged["offline_high_quality"].setdefault("render_scale", 2)
        nerf_cfg = dict(merged["offline_high_quality"].get("nerf_refinement", {}) or {})
        nerf_cfg.setdefault("enabled", True)
        merged["offline_high_quality"]["nerf_refinement"] = nerf_cfg
        super().__init__(merged)

"""Renderable layered-scene model for Gaussian backends."""

from __future__ import annotations

from room4d.render.base import RenderSceneModel


class Gaussian4DScene(RenderSceneModel):
    """Container for a layered static/dynamic scene contract."""

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg=cfg or {}, contract_name="layered_scene_v1")

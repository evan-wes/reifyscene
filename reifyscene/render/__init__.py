"""Rendering modules for the 4D scene representation."""

from reifyscene.render.base import RenderBackend
from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend
from reifyscene.render.registry import build_renderer

__all__ = ["RenderBackend", "GaussianRenderBackend", "build_renderer"]

"""Rendering modules for the 4D scene representation."""

from room4d.render.base import RenderBackend
from room4d.render.gaussian4d.render_backend import GaussianRenderBackend
from room4d.render.registry import build_renderer

__all__ = ["RenderBackend", "GaussianRenderBackend", "build_renderer"]

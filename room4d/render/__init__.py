"""Rendering modules for the 4D scene representation."""

from room4d.render.base import RenderBackend, RenderSceneModel
from room4d.render.gaussian4d.trainer import Gaussian4DTrainer
from room4d.render.registry import build_renderer

__all__ = ["RenderBackend", "RenderSceneModel", "Gaussian4DTrainer", "build_renderer"]

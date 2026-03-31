"""Lightweight scene-model container for simple 4D scene rendering."""

from __future__ import annotations

from room4d.scene.scene_artifact import SceneArtifact


class Gaussian4DScene:
    """Container for a lightweight dynamic scene model."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.scene_artifact: SceneArtifact | None = None

    def initialize_from_scene(self, scene_artifact: SceneArtifact) -> None:
        self.scene_artifact = scene_artifact

"""Trainer for the dynamic Gaussian scene model."""
from pathlib import Path

from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.gaussian4d.render_artifact import RenderArtifact


class Gaussian4DTrainer:
    """Fit a dynamic Gaussian scene representation from optimized state."""

    def fit(self, scene: Gaussian4DScene, optimization_artifact) -> RenderArtifact:
        """Return a placeholder render artifact."""
        return RenderArtifact(run_id="dev", stage_name="render", save_dir=Path("."))

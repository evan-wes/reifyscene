"""Wrapper module for `dust3r_wrapper`."""
from pathlib import Path

from reifyscene.geometry.base import GeometryInitializer
from reifyscene.geometry.geometry_artifact import GeometryArtifact


class DUSt3RInitializer(GeometryInitializer):
    """Foundation-model wrapper for coarse geometry initialization."""

    def __init__(self, cfg: dict | None = None) -> None:
        """Store backend configuration."""
        self.cfg = cfg or {}

    def initialize(self, frames):
        """Return a placeholder geometry artifact."""
        return GeometryArtifact(run_id="dev", stage_name="geometry", save_dir=Path("."))

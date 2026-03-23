"""Wrapper module for `tapir_wrapper`."""
from pathlib import Path

from room4d.tracking.base import PointTracker
from room4d.tracking.track_artifact import TrackArtifact


class TAPIRWrapper(PointTracker):
    """Foundation-model wrapper for point tracking."""

    def __init__(self, cfg: dict | None = None) -> None:
        """Store backend configuration."""
        self.cfg = cfg or {}

    def track(self, stream, seed_points=None):
        """Return a placeholder track artifact."""
        return TrackArtifact(run_id="dev", stage_name="tracking", save_dir=Path("."))

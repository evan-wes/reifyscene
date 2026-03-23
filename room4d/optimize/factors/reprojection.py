"""Reprojection factor scaffold."""

from __future__ import annotations

from room4d.common.types import TrackObservation
from room4d.optimize.factors.base import Factor


class ReprojectionFactor(Factor):
    """Measure consistency between 2D observations and camera/point variables."""

    name = "reprojection"

    def __init__(self, observations: list[TrackObservation] | None = None, weight: float = 1.0) -> None:
        self.observations = observations or []
        self.weight = weight

    def residual(self, state: dict):
        """Return a placeholder residual proportional to observation count."""
        residual = float(len(self.observations)) * self.weight
        return residual, {"num_observations": len(self.observations), "weight": self.weight}

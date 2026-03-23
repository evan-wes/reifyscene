"""Track-consistency factor scaffold."""

from __future__ import annotations

from room4d.optimize.factors.base import Factor


class TrackConsistencyFactor(Factor):
    """Encourage track observations to agree with one latent 3D explanation."""

    name = "track_consistency"

    def __init__(self, per_stream_counts: dict[str, int] | None = None, weight: float = 1.0) -> None:
        self.per_stream_counts = per_stream_counts or {}
        self.weight = weight

    def residual(self, state: dict):
        """Return a placeholder residual based on observation density."""
        count = sum(self.per_stream_counts.values())
        residual = float(count) * self.weight
        return residual, {"total_track_observations": count, "weight": self.weight}

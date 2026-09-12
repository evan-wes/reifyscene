"""Time-alignment factor scaffold."""

from __future__ import annotations

from reifyscene.optimize.factors.base import Factor


class TimeAlignmentFactor(Factor):
    """Regularize per-stream time offsets against sync priors."""

    name = "time_alignment"

    def __init__(self, offsets_by_stream: dict[str, float] | None = None, weight: float = 1.0) -> None:
        self.offsets_by_stream = offsets_by_stream or {}
        self.weight = weight

    def residual(self, state: dict):
        """Return a placeholder residual proportional to active stream count."""
        residual = float(len(self.offsets_by_stream)) * self.weight
        return residual, {"num_streams": len(self.offsets_by_stream), "weight": self.weight}

"""Prior factor scaffold."""

from __future__ import annotations

from room4d.optimize.factors.base import Factor


class PriorFactor(Factor):
    """Keep optimization variables near geometry and sync priors."""

    name = "priors"

    def __init__(self, num_priors: int = 0, weight: float = 1.0) -> None:
        self.num_priors = num_priors
        self.weight = weight

    def residual(self, state: dict):
        """Return a placeholder residual proportional to prior count."""
        residual = float(self.num_priors) * self.weight
        return residual, {"num_priors": self.num_priors, "weight": self.weight}

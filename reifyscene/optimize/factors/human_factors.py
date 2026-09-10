"""Optimization factor module `human_factors`."""

from reifyscene.optimize.factors.base import Factor


class HumanFactors(Factor):
    """Placeholder factor implementation."""

    name = "human_factors"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

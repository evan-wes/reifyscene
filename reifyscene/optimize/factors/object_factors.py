"""Optimization factor module `object_factors`."""

from reifyscene.optimize.factors.base import Factor


class ObjectFactors(Factor):
    """Placeholder factor implementation."""

    name = "object_factors"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

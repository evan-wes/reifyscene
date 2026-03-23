"""Optimization factor module `object_factors`."""

from room4d.optimize.factors.base import Factor


class ObjectFactors(Factor):
    """Placeholder factor implementation."""

    name = "object_factors"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

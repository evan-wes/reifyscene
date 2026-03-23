"""Optimization factor module `smoothness`."""

from room4d.optimize.factors.base import Factor


class SmoothnessFactor(Factor):
    """Placeholder factor implementation."""

    name = "smoothness"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

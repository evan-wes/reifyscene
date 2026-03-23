"""Optimization factor module `depth_consistency`."""

from room4d.optimize.factors.base import Factor


class DepthConsistencyFactor(Factor):
    """Placeholder factor implementation."""

    name = "depth_consistency"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

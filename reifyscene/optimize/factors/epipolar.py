"""Optimization factor module `epipolar`."""

from reifyscene.optimize.factors.base import Factor


class EpipolarFactor(Factor):
    """Placeholder factor implementation."""

    name = "epipolar"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

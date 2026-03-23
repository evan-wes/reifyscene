"""Optimization factor module `graph_factors`."""

from room4d.optimize.factors.base import Factor


class GraphFactors(Factor):
    """Placeholder factor implementation."""

    name = "graph_factors"

    def residual(self, state: dict):
        """Return placeholder residuals and diagnostics."""
        return 0.0, {}

"""Base factor interface."""

from abc import ABC, abstractmethod


class Factor(ABC):
    """Abstract factor class for residual computations."""

    name: str = "factor"

    @abstractmethod
    def residual(self, state: dict):
        """Return residuals and diagnostics for the given state."""
        raise NotImplementedError

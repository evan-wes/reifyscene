"""Dynamic Gaussian scene representation."""

class Gaussian4DScene:
    """Container for a dynamic Gaussian scene model."""

    def __init__(self, cfg: dict | None = None) -> None:
        """Store renderer configuration."""
        self.cfg = cfg or {}

    def initialize_from_optimization(self, opt_artifact) -> None:
        """Initialize scene state from an optimization artifact."""
        return None

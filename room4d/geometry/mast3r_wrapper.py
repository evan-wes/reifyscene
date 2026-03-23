"""Wrapper for MASt3R matching and refinement."""

class MASt3RMatcher:
    """Run pairwise image matching and confidence estimation."""

    def __init__(self, cfg: dict | None = None) -> None:
        """Store backend configuration."""
        self.cfg = cfg or {}

    def match_pairs(self, frame_pairs):
        """Return placeholder pairwise matches."""
        return {}

"""Frame index utilities."""

from reifyscene.common.types import FrameRef


class FrameIndex:
    """In-memory frame index for fast lookup."""

    def __init__(self) -> None:
        """Initialize an empty frame index."""
        self.frames: list[FrameRef] = []

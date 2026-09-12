"""Clip sampling helpers."""

from reifyscene.common.types import FrameRef


class ClipSampler:
    """Sample sparse or dense clips from a session."""

    def sample(self, frames: list[FrameRef]) -> list[FrameRef]:
        """Return a placeholder subset of frame references."""
        return frames

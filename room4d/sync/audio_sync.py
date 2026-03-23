"""Audio-based synchronization estimators."""

from room4d.common.types import VideoStreamRef


class AudioSyncEstimator:
    """Estimate pairwise temporal offsets from audio."""

    def estimate_offset(self, a: VideoStreamRef, b: VideoStreamRef) -> tuple[float, float]:
        """Return a placeholder offset and confidence."""
        return 0.0, 0.0

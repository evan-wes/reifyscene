"""Motion-based synchronization estimators."""

from reifyscene.common.types import VideoStreamRef


class MotionSyncEstimator:
    """Estimate pairwise temporal offsets from motion cues."""

    def estimate_offset(
        self,
        stream_a: VideoStreamRef,
        stream_b: VideoStreamRef,
        track_artifact=None,
    ) -> tuple[float, float]:
        """Return a placeholder offset and confidence."""
        return 0.0, 0.0

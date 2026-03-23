"""Base interfaces for point trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from room4d.common.types import FrameRef, VideoStreamRef

if TYPE_CHECKING:
    from room4d.tracking.track_artifact import TrackArtifact


class PointTracker(ABC):
    """Abstract point-tracking interface."""

    @abstractmethod
    def track(
        self,
        stream: VideoStreamRef,
        *,
        frame_refs: list[FrameRef] | None = None,
        seed_points: np.ndarray | None = None,
    ) -> TrackArtifact:
        """Track points through a stream and return a track artifact."""
        raise NotImplementedError

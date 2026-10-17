"""Clip quality scoring utilities for preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

from reifyscene.common.types import VideoStreamRef


class ClipQualityAssessor:
    """Score stream usability before expensive geometry and tracking stages.

    TODO implementation order:
    1. Add blur, exposure, and motion scores from sampled thumbnails.
    2. Add audio-presence and variable-frame-rate diagnostics.
    3. Add a learned ranking model once real data is available.
    """

    def score_frame(self, frame: np.ndarray) -> dict[str, float]:
        """Return placeholder per-frame quality statistics."""
        mean_intensity = float(frame.mean()) if frame.size else 0.0
        return {
            "usable": 1.0,
            "mean_intensity": mean_intensity,
            "blur_score": 0.0,
        }

    def score_stream(self, stream: VideoStreamRef) -> dict[str, float]:
        """Return placeholder stream-level quality statistics."""
        return {
            "usable": 1.0,
            "fps": stream.fps,
            "has_audio": 1.0 if stream.has_audio else 0.0,
            "resolution_known": 1.0 if stream.width > 0 and stream.height > 0 else 0.0,
        }

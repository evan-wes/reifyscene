"""Artifact for 2D and merged track results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact, TrackObservation


@dataclass
class TrackArtifact(BaseArtifact):
    """Store per-stream and merged point tracks.

    The first vertical slice keeps only sparse bookkeeping here. The eventual
    tracker integration can add feature descriptors, track scores, and lifted 3D
    associations without changing the basic contract.
    """

    per_stream_observations: dict[str, list[TrackObservation]] = field(default_factory=dict)
    seed_points_by_stream: dict[str, Any] = field(default_factory=dict)
    merged_track_groups: dict[str, list[str]] = field(default_factory=dict)
    visibility_masks: dict[str, Any] = field(default_factory=dict)
    static_dynamic_hints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stream_ids(self) -> list[str]:
        """Return all streams that contributed observations."""
        return sorted(self.per_stream_observations.keys())

    def total_observations(self) -> int:
        """Return the total number of 2D point observations."""
        return sum(len(obs) for obs in self.per_stream_observations.values())

    def summary(self) -> dict[str, Any]:
        """Return a compact tracking summary."""
        return {
            "run_id": self.run_id,
            "num_streams": len(self.per_stream_observations),
            "total_observations": self.total_observations(),
            "notes": self.notes,
        }

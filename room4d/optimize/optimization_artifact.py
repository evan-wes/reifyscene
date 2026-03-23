"""Artifact for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class OptimizationArtifact(BaseArtifact):
    """Store refined state after joint optimization.

    The initial scaffold stores optimizer *state and diagnostics* rather than a
    final solved geometry. That keeps the vertical slice useful before a true
    nonlinear optimizer is integrated.
    """

    refined_cameras: dict[str, Any] = field(default_factory=dict)
    refined_offsets: dict[str, Any] = field(default_factory=dict)
    tracks_3d: dict[str, Any] = field(default_factory=dict)
    object_trajectories: dict[str, Any] = field(default_factory=dict)
    human_trajectories: dict[str, Any] = field(default_factory=dict)
    stage_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    schedule_used: list[str] = field(default_factory=list)
    pending_todos: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a compact optimization summary."""
        return {
            "run_id": self.run_id,
            "num_refined_cameras": len(self.refined_cameras),
            "num_stages": len(self.stage_diagnostics),
            "schedule_used": self.schedule_used,
            "pending_todos": self.pending_todos,
        }

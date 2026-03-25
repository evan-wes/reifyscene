"""Artifact for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import BaseArtifact, CameraPose, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact


@dataclass
class OptimizationArtifact(BaseArtifact):
    """Store refined state after joint optimization."""

    refined_cameras: dict[str, list[TimedCameraState]] = field(default_factory=dict)
    refined_offsets: dict[str, Any] = field(default_factory=dict)
    tracks_3d: dict[str, Any] = field(default_factory=dict)
    object_trajectories: dict[str, Any] = field(default_factory=dict)
    human_trajectories: dict[str, Any] = field(default_factory=dict)
    stage_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    schedule_used: list[str] = field(default_factory=list)
    pending_todos: list[str] = field(default_factory=list)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): OptimizationArtifact._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OptimizationArtifact._to_serializable(v) for v in value]
        return value

    def save(self) -> None:
        ensure_dir(self.save_dir)
        raw = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": self.metadata,
            "refined_cameras": {
                stream_id: [GeometryArtifact._camera_state_to_dict(state) for state in states]
                for stream_id, states in self.refined_cameras.items()
            },
            "refined_offsets": self._to_serializable(self.refined_offsets),
            "tracks_3d": self._to_serializable(self.tracks_3d),
            "object_trajectories": self._to_serializable(self.object_trajectories),
            "human_trajectories": self._to_serializable(self.human_trajectories),
            "stage_diagnostics": self._to_serializable(self.stage_diagnostics),
            "schedule_used": list(self.schedule_used),
            "pending_todos": list(self.pending_todos),
        }
        write_json(self.artifact_path(), raw)

    @classmethod
    def load(cls, save_dir: Path) -> "OptimizationArtifact":
        raw = read_json(save_dir / "artifact.json")
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
            refined_cameras={
                stream_id: [GeometryArtifact._camera_state_from_dict(state) for state in states]
                for stream_id, states in raw.get("refined_cameras", {}).items()
            },
            refined_offsets=raw.get("refined_offsets", {}),
            tracks_3d=raw.get("tracks_3d", {}),
            object_trajectories=raw.get("object_trajectories", {}),
            human_trajectories=raw.get("human_trajectories", {}),
            stage_diagnostics=raw.get("stage_diagnostics", []),
            schedule_used=raw.get("schedule_used", []),
            pending_todos=raw.get("pending_todos", []),
        )

    def summary(self) -> dict[str, Any]:
        track_summary = self.tracks_3d.get("streams", {}) if isinstance(self.tracks_3d, dict) else {}
        improved_streams = [
            stream_id for stream_id, info in track_summary.items() if isinstance(info, dict) and info.get("applied")
        ]
        return {
            "run_id": self.run_id,
            "num_refined_cameras": sum(len(states) for states in self.refined_cameras.values()),
            "num_stages": len(self.stage_diagnostics),
            "schedule_used": self.schedule_used,
            "static_track_refined_streams": improved_streams,
            "pending_todos": self.pending_todos,
        }

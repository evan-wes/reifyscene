"""Artifact for rendered novel-view outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact


@dataclass
class RenderArtifact(BaseArtifact):
    """Store rendered novel-view outputs and playback metadata."""

    scene_state: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    path_index: dict[str, list[str]] = field(default_factory=dict)
    layer_path_index: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    player_html: str | None = None
    debug_player_html: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "num_paths": len(self.path_index),
            "frames_per_path": {k: len(v) for k, v in self.path_index.items()},
            "layer_frames_per_path": {
                path: {layer: len(frames) for layer, frames in layers.items()}
                for path, layers in self.layer_path_index.items()
            },
            "player_html": self.player_html,
            "debug_player_html": self.debug_player_html,
            "coverage": self.coverage,
        }

    def save(self) -> None:
        ensure_dir(self.save_dir)
        write_json(self.artifact_path(), {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": self.metadata,
            "scene_state": self.scene_state,
            "coverage": self.coverage,
            "path_index": self.path_index,
            "layer_path_index": self.layer_path_index,
            "player_html": self.player_html,
            "debug_player_html": self.debug_player_html,
        })

    @classmethod
    def load(cls, save_dir: Path) -> "RenderArtifact":
        raw = read_json(save_dir / "artifact.json")
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
            scene_state=raw.get("scene_state", {}),
            coverage=raw.get("coverage", {}),
            path_index=raw.get("path_index", {}),
            layer_path_index=raw.get("layer_path_index", {}),
            player_html=raw.get("player_html"),
            debug_player_html=raw.get("debug_player_html"),
        )

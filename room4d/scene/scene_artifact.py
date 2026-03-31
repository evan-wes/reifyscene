"""Artifact for assembled single-video 4D scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import BaseArtifact, FrameRef


@dataclass
class SceneArtifact(BaseArtifact):
    """Persist a lightweight static/dynamic single-video 4D scene."""

    static_scene: dict[str, Any] = field(default_factory=dict)
    dynamic_entities: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    frame_refs: list[FrameRef] = field(default_factory=list)
    timeline: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        counts = {kind: len(items) for kind, items in self.dynamic_entities.items()}
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "num_static_points": int(self.static_scene.get("num_points", 0)),
            "dynamic_counts": counts,
            "num_frames": len(self.frame_refs),
            "notes": list(self.notes),
        }

    def save(self) -> None:
        ensure_dir(self.save_dir)
        static_dir = ensure_dir(self.save_dir / "static")
        points = np.asarray(self.static_scene.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(self.static_scene.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        np.savez_compressed(static_dir / "background_points.npz", points_xyz=points, colors_rgb=colors)

        serializable = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": self.metadata,
            "static_scene": {
                key: value
                for key, value in self.static_scene.items()
                if key not in {"points_xyz", "colors_rgb"}
            },
            "dynamic_entities": self.dynamic_entities,
            "frame_refs": [
                {"stream_id": fr.stream_id, "frame_idx": int(fr.frame_idx), "timestamp_sec": float(fr.timestamp_sec)}
                for fr in self.frame_refs
            ],
            "timeline": self.timeline,
            "notes": self.notes,
        }
        write_json(self.artifact_path(), serializable)

    @classmethod
    def load(cls, save_dir: Path) -> "SceneArtifact":
        raw = read_json(save_dir / "artifact.json")
        static_npz = np.load(save_dir / "static" / "background_points.npz")
        static_scene = dict(raw.get("static_scene", {}))
        static_scene["points_xyz"] = np.asarray(static_npz["points_xyz"], dtype=np.float32)
        static_scene["colors_rgb"] = np.asarray(static_npz["colors_rgb"], dtype=np.uint8)
        frame_refs = [
            FrameRef(stream_id=str(fr["stream_id"]), frame_idx=int(fr["frame_idx"]), timestamp_sec=float(fr["timestamp_sec"]))
            for fr in raw.get("frame_refs", [])
        ]
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
            static_scene=static_scene,
            dynamic_entities=raw.get("dynamic_entities", {}),
            frame_refs=frame_refs,
            timeline=raw.get("timeline", {}),
            notes=raw.get("notes", []),
        )

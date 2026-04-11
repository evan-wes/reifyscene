"""Artifact for assembled single-video 4D scenes."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
    render_contract: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        counts = {kind: len(items) for kind, items in self.dynamic_entities.items()}
        static_layer = (self.render_contract or {}).get("static_layer", {}) or {}
        dynamic_layers = (self.render_contract or {}).get("dynamic_layers", {}) or {}
        appearance = (self.render_contract or {}).get("appearance", {}) or {}
        dynamic_layer_counts = {kind: len(items) for kind, items in dynamic_layers.items()} if isinstance(dynamic_layers, dict) else {}
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "num_static_points": int(self.static_scene.get("num_points", 0)),
            "dynamic_counts": counts,
            "num_frames": len(self.frame_refs),
            "render_contract_schema": str((self.render_contract or {}).get("schema", "")),
            "num_static_gaussians": int(static_layer.get("num_gaussians", 0)),
            "dynamic_layer_counts": dynamic_layer_counts,
            "static_appearance_sources": int(appearance.get("static_source_count", 0)),
            "notes": list(self.notes),
        }

    @staticmethod
    def _strip_static_layer_arrays(render_contract: dict[str, Any]) -> dict[str, Any]:
        serializable = dict(render_contract)
        static_layer = dict(serializable.get("static_layer", {}) or {})
        for key in ("means_xyz", "scales_xyz", "colors_rgb", "opacities", "source_codes"):
            static_layer.pop(key, None)
        if static_layer:
            serializable["static_layer"] = static_layer
        else:
            serializable.pop("static_layer", None)
        return serializable

    def _save_static_gaussians(self, contract: dict[str, Any], layer_dir: Path) -> dict[str, Any]:
        static_layer = dict(contract.get("static_layer", {}) or {})
        means = np.asarray(static_layer.get("means_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        scales = np.asarray(static_layer.get("scales_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(static_layer.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        opacities = np.asarray(static_layer.get("opacities", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        source_codes = np.asarray(static_layer.get("source_codes", np.zeros((0,), dtype=np.int16)), dtype=np.int16)
        np.savez_compressed(
            layer_dir / "static_gaussians.npz",
            means_xyz=means,
            scales_xyz=scales,
            colors_rgb=colors,
            opacities=opacities,
            source_codes=source_codes,
        )
        static_meta = dict(static_layer)
        static_meta["num_gaussians"] = int(means.shape[0])
        static_meta["npz_file"] = str((Path("render_contract") / "static_gaussians.npz").as_posix())
        contract["static_layer"] = static_meta
        return contract

    def _save_appearance(self, contract: dict[str, Any], layer_dir: Path) -> dict[str, Any]:
        appearance = dict(contract.get("appearance", {}) or {})
        static_sources = list(appearance.get("static_sources", []) or [])
        saved_static_sources: list[dict[str, Any]] = []
        static_dir = ensure_dir(layer_dir / "static_sources")
        for idx, source in enumerate(static_sources):
            src = dict(source)
            stem = f"{idx:03d}_{str(src.get('frame_key', idx)).replace(':', '__').replace('/', '_')}"
            image_path = src.get("image_path") or src.get("image_file")
            if image_path is not None:
                src_path = Path(image_path)
                dst_path = static_dir / f"{stem}{src_path.suffix or '.png'}"
                if src_path.exists() and src_path.resolve() != dst_path.resolve():
                    shutil.copy2(src_path, dst_path)
                src["image_file"] = str((Path("render_contract") / "static_sources" / dst_path.name).as_posix())
            points = src.pop("points_xyz", None)
            confidence = src.pop("confidence", None)
            existing_points_file = src.get("points_file")
            if points is not None:
                npz_path = static_dir / f"{stem}_points.npz"
                np.savez_compressed(
                    npz_path,
                    points_xyz=np.asarray(points, dtype=np.float32),
                    confidence=np.zeros((0,), dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32),
                )
                src["points_file"] = str((Path("render_contract") / "static_sources" / npz_path.name).as_posix())
            elif existing_points_file is not None:
                src["points_file"] = str((Path("render_contract") / "static_sources" / Path(existing_points_file).name).as_posix()) if Path(existing_points_file).exists() else existing_points_file
            saved_static_sources.append(src)
        appearance["static_sources"] = saved_static_sources
        appearance["static_source_count"] = int(len(saved_static_sources))
        contract["appearance"] = appearance

        dynamic_dir = ensure_dir(layer_dir / "dynamic_sprites")
        dynamic_layers = dict(contract.get("dynamic_layers", {}) or {})
        for kind, entities in list(dynamic_layers.items()):
            saved_entities = []
            for ent_idx, entity in enumerate(list(entities or [])):
                ent = dict(entity)
                saved_obs = []
                for obs_idx, obs in enumerate(list(ent.get("observations", []) or [])):
                    obs_dict = dict(obs)
                    sprite = obs_dict.pop("sprite_rgba", None)
                    if sprite is not None:
                        name = f"{kind}_{ent_idx:03d}_{obs_idx:03d}.png"
                        sprite_path = dynamic_dir / name
                        Image.fromarray(np.asarray(sprite, dtype=np.uint8), mode="RGBA").save(sprite_path)
                        obs_dict["sprite_file"] = str((Path("render_contract") / "dynamic_sprites" / name).as_posix())
                    saved_obs.append(obs_dict)
                ent["observations"] = saved_obs
                saved_entities.append(ent)
            dynamic_layers[kind] = saved_entities
        contract["dynamic_layers"] = dynamic_layers
        return contract

    def _save_render_contract(self) -> dict[str, Any]:
        contract = dict(self.render_contract or {})
        if not contract:
            return {}
        layer_dir = ensure_dir(self.save_dir / "render_contract")
        contract = self._save_static_gaussians(contract, layer_dir)
        contract = self._save_appearance(contract, layer_dir)
        return self._strip_static_layer_arrays(contract)

    @classmethod
    def _load_render_contract(cls, save_dir: Path, raw_contract: dict[str, Any]) -> dict[str, Any]:
        contract = cls._deserialize_untyped(raw_contract or {})
        if not contract:
            return {}
        static_layer = dict(contract.get("static_layer", {}) or {})
        npz_rel = static_layer.get("npz_file")
        if npz_rel:
            npz_path = save_dir / Path(npz_rel)
            if npz_path.exists():
                static_npz = np.load(npz_path)
                static_layer["means_xyz"] = np.asarray(static_npz["means_xyz"], dtype=np.float32)
                static_layer["scales_xyz"] = np.asarray(static_npz["scales_xyz"], dtype=np.float32)
                static_layer["colors_rgb"] = np.asarray(static_npz["colors_rgb"], dtype=np.uint8)
                static_layer["opacities"] = np.asarray(static_npz["opacities"], dtype=np.float32)
                static_layer["source_codes"] = np.asarray(static_npz["source_codes"], dtype=np.int16)
        if static_layer:
            contract["static_layer"] = static_layer
        appearance = dict(contract.get("appearance", {}) or {})
        static_sources = []
        for source in list(appearance.get("static_sources", []) or []):
            src = dict(source)
            image_file = src.get("image_file")
            if image_file is not None:
                src["image_file"] = str((save_dir / Path(image_file)).resolve())
            points_file = src.get("points_file")
            if points_file is not None:
                src["points_file"] = str((save_dir / Path(points_file)).resolve())
            static_sources.append(src)
        appearance["static_sources"] = static_sources
        contract["appearance"] = appearance
        dynamic_layers = dict(contract.get("dynamic_layers", {}) or {})
        for kind, entities in list(dynamic_layers.items()):
            loaded_entities = []
            for entity in list(entities or []):
                ent = dict(entity)
                obs_loaded = []
                for obs in list(ent.get("observations", []) or []):
                    obs_dict = dict(obs)
                    sprite_file = obs_dict.get("sprite_file")
                    if sprite_file is not None:
                        obs_dict["sprite_file"] = str((save_dir / Path(sprite_file)).resolve())
                    obs_loaded.append(obs_dict)
                ent["observations"] = obs_loaded
                loaded_entities.append(ent)
            dynamic_layers[kind] = loaded_entities
        contract["dynamic_layers"] = dynamic_layers
        return contract

    def save(self) -> None:
        ensure_dir(self.save_dir)
        static_dir = ensure_dir(self.save_dir / "static")
        points = np.asarray(self.static_scene.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(self.static_scene.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        np.savez_compressed(static_dir / "background_points.npz", points_xyz=points, colors_rgb=colors)

        entity_debug_summary = dict((self.metadata or {}).get('entity_debug_summary', {}) or {})
        if entity_debug_summary:
            write_json(self.save_dir / 'render_entity_summary.json', entity_debug_summary)

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
            "render_contract": self._serialize_value(self._save_render_contract()),
            "notes": self.notes,
            "entity_debug_summary_file": "render_entity_summary.json" if entity_debug_summary else None,
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
        render_contract = cls._load_render_contract(save_dir, raw.get("render_contract", {}))
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
            static_scene=static_scene,
            dynamic_entities=raw.get("dynamic_entities", {}),
            frame_refs=frame_refs,
            timeline=raw.get("timeline", {}),
            render_contract=render_contract,
            notes=raw.get("notes", []),
        )

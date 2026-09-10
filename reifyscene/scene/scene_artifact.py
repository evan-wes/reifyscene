"""Artifact for a self-contained, renderable 4D scene.

The scene stage turns a reconstruction (geometry + ingest + tracking) into a
**self-contained renderable scene** and the render stage consumes *only* this
artifact. It carries everything the render stage needs:

* ``render_seed`` -- a dense Gaussian seed cloud (means / colors / scales),
  backprojected from geometry with dynamic regions masked out.
* ``train_views`` -- the per-frame training supervision: camera pose +
  intrinsics + the GT frame itself (persisted into the artifact directory, so
  the artifact is portable and reproducible with no dependency on the upstream
  geometry/ingest artifacts at render time).
* ``render_trajectory`` -- the novel-view cameras to render the answer along.

The held-out GT views + covisibility masks used to *score* a benchmark dataset
are deliberately NOT here -- they are an eval-stage input (dataset-sourced),
not part of the scene.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef


def _sanitize_key(key: str) -> str:
    return str(key).replace(":", "__").replace("/", "_")


@dataclass
class SceneArtifact(BaseArtifact):
    """A self-contained renderable static/dynamic 4D scene."""

    # "static" | "dynamic" -- selects the trainer in the render stage.
    kind: str = "static"
    # Dense Gaussian seed: means_xyz (N,3) float32, colors_rgb (N,3) uint8,
    # scales_xyz (N,3) float32, num_gaussians int.
    render_seed: dict[str, Any] = field(default_factory=dict)
    # Per-frame training supervision. Each entry:
    #   frame_key, time_id, t_world_cam (4x4), intrinsics {fx,fy,cx,cy},
    #   width, height, image_file (relative to save_dir once persisted).
    # During assembly the entry carries an absolute ``image_path``; save()
    # copies it into the artifact and rewrites it to ``image_file``.
    train_views: list[dict[str, Any]] = field(default_factory=list)
    # Novel-view output trajectory: t_world_cam, intrinsics, width, height, time_id.
    render_trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Dynamic SE(3) motion bundle init (F0b: DynamicScene / motion bases). Empty for static.
    dynamic_bundle: dict[str, Any] = field(default_factory=dict)
    frame_refs: list[FrameRef] = field(default_factory=list)
    timeline: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        seed = self.render_seed or {}
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "kind": self.kind,
            "num_seed_gaussians": int(seed.get("num_gaussians", np.asarray(seed.get("means_xyz", np.zeros((0, 3)))).shape[0])),
            "num_train_views": len(self.train_views),
            "num_render_cameras": len(self.render_trajectory),
            "num_frames": len(self.frame_refs),
            "notes": list(self.notes),
        }

    # ------------------------------------------------------------------ save

    def _save_seed(self, seed_dir: Path) -> dict[str, Any]:
        seed = dict(self.render_seed or {})
        means = np.asarray(seed.get("means_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(seed.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        scales = np.asarray(seed.get("scales_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        np.savez_compressed(seed_dir / "seed_cloud.npz", means_xyz=means, colors_rgb=colors, scales_xyz=scales)
        return {"num_gaussians": int(means.shape[0]), "npz_file": "render_seed/seed_cloud.npz"}

    def _save_train_views(self, views_dir: Path) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for idx, view in enumerate(self.train_views):
            entry = {
                "frame_key": str(view.get("frame_key", idx)),
                "time_id": int(view.get("time_id", idx)),
                "t_world_cam": np.asarray(view["t_world_cam"], dtype=np.float32).tolist(),
                "intrinsics": {k: float(view["intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")},
                "width": int(view["width"]),
                "height": int(view["height"]),
            }
            image_file = view.get("image_file")
            image_path = view.get("image_path")
            if image_path is not None and Path(image_path).exists():
                stem = f"{idx:04d}_{_sanitize_key(entry['frame_key'])}"
                dst = views_dir / f"{stem}{Path(image_path).suffix or '.png'}"
                if Path(image_path).resolve() != dst.resolve():
                    shutil.copy2(image_path, dst)
                entry["image_file"] = str((Path("train_views") / dst.name).as_posix())
            elif image_file is not None:
                entry["image_file"] = str(image_file)
            saved.append(entry)
        return saved

    def _save_dynamic_bundle(self, save_dir: Path) -> dict[str, Any]:
        """Persist a dynamic ``DynamicScene`` bundle as its own npz (not inlined JSON).

        ``dynamic_bundle`` holds the borrowed-init numpy schema (fg/bg gaussians +
        motion bases + per-split cameras, from ``motion_init.dump_bundle``). Mirrors
        ``_save_seed``: write the arrays to ``dynamic/scene_bundle.npz`` and keep
        only a path + counts in artifact.json. Empty for static scenes.
        """
        bundle = self.dynamic_bundle or {}
        if not bundle:
            return {}
        arrays = {k: np.asarray(v) for k, v in bundle.items()}
        dyn_dir = ensure_dir(save_dir / "dynamic")
        np.savez_compressed(dyn_dir / "scene_bundle.npz", **arrays)
        fg = np.asarray(arrays.get("fg_means", np.zeros((0, 3))))
        bg = np.asarray(arrays.get("bg_means", np.zeros((0, 3))))
        return {
            "npz_file": "dynamic/scene_bundle.npz",
            "num_fg": int(fg.shape[0]),
            "num_bg": int(bg.shape[0]),
            "num_bases": int(np.asarray(arrays.get("num_bases", 0))),
        }

    def save(self) -> None:
        ensure_dir(self.save_dir)
        seed_meta = self._save_seed(ensure_dir(self.save_dir / "render_seed"))
        train_views = self._save_train_views(ensure_dir(self.save_dir / "train_views"))
        serializable = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": self.metadata,
            "kind": self.kind,
            "render_seed": seed_meta,
            "train_views": train_views,
            "render_trajectory": [
                {
                    "t_world_cam": np.asarray(cam["t_world_cam"], dtype=np.float32).tolist(),
                    "intrinsics": {k: float(cam["intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")},
                    "width": int(cam["width"]),
                    "height": int(cam["height"]),
                    "time_id": int(cam.get("time_id", 0)),
                }
                for cam in self.render_trajectory
            ],
            "dynamic_bundle": self._save_dynamic_bundle(self.save_dir),
            "frame_refs": [
                {"stream_id": fr.stream_id, "frame_idx": int(fr.frame_idx), "timestamp_sec": float(fr.timestamp_sec)}
                for fr in self.frame_refs
            ],
            "timeline": self.timeline,
            "notes": self.notes,
        }
        write_json(self.artifact_path(), serializable)

    # ------------------------------------------------------------------ load

    @classmethod
    def _load_seed(cls, save_dir: Path, raw_seed: dict[str, Any]) -> dict[str, Any]:
        npz_rel = (raw_seed or {}).get("npz_file")
        if not npz_rel:
            return {"means_xyz": np.zeros((0, 3), dtype=np.float32), "colors_rgb": np.zeros((0, 3), dtype=np.uint8),
                    "scales_xyz": np.zeros((0, 3), dtype=np.float32), "num_gaussians": 0}
        npz = np.load(save_dir / Path(npz_rel))
        means = np.asarray(npz["means_xyz"], dtype=np.float32)
        return {
            "means_xyz": means,
            "colors_rgb": np.asarray(npz["colors_rgb"], dtype=np.uint8),
            "scales_xyz": np.asarray(npz["scales_xyz"], dtype=np.float32),
            "num_gaussians": int(means.shape[0]),
        }

    @classmethod
    def _load_dynamic_bundle(cls, save_dir: Path, raw: dict[str, Any]) -> dict[str, Any]:
        """Reload the dynamic npz back into the in-memory numpy-bundle shape.

        Symmetric with ``_save_dynamic_bundle`` / ``_load_seed``: a reloaded
        artifact carries the same numpy dict the assembler produced, so the
        render stage feeds it straight into ``motion_init.bundle_to_scene``.
        """
        npz_rel = (raw or {}).get("npz_file")
        if not npz_rel:
            return {}
        data = np.load(save_dir / Path(npz_rel), allow_pickle=True)
        return {k: data[k] for k in data.files}

    @classmethod
    def load(cls, save_dir: Path) -> "SceneArtifact":
        raw = read_json(save_dir / "artifact.json")
        render_seed = cls._load_seed(save_dir, raw.get("render_seed", {}))
        train_views = []
        for view in raw.get("train_views", []):
            entry = dict(view)
            image_file = entry.get("image_file")
            if image_file is not None:
                entry["image_file"] = str((save_dir / Path(image_file)).resolve())
            entry["t_world_cam"] = np.asarray(entry["t_world_cam"], dtype=np.float32)
            train_views.append(entry)
        render_trajectory = []
        for cam in raw.get("render_trajectory", []):
            entry = dict(cam)
            entry["t_world_cam"] = np.asarray(entry["t_world_cam"], dtype=np.float32)
            render_trajectory.append(entry)
        frame_refs = [
            FrameRef(stream_id=str(fr["stream_id"]), frame_idx=int(fr["frame_idx"]), timestamp_sec=float(fr["timestamp_sec"]))
            for fr in raw.get("frame_refs", [])
        ]
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(raw["save_dir"]),
            metadata=raw.get("metadata", {}),
            kind=str(raw.get("kind", "static")),
            render_seed=render_seed,
            train_views=train_views,
            render_trajectory=render_trajectory,
            dynamic_bundle=cls._load_dynamic_bundle(save_dir, raw.get("dynamic_bundle", {})),
            frame_refs=frame_refs,
            timeline=raw.get("timeline", {}),
            notes=raw.get("notes", []),
        )

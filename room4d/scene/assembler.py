"""Single-video scene assembly: build a self-contained renderable SceneArtifact.

The scene stage turns a reconstruction (geometry + ingest GT frames) into a
self-contained renderable scene:

* a dense Gaussian **seed** (backprojected geometry depth),
* per-frame **training supervision** (camera + the GT frame itself), and
* a novel-view **render trajectory** (an orbit around the scene).

The render stage consumes ONLY the SceneArtifact this emits -- it trains the
seed against the supervision and renders the trajectory. Entity semantics
(world trajectories, SMPL grounding) are a graph-stage concern and live no
longer here; the dynamic SE(3) motion bundle is wired in F0b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room4d.common.types import FrameRef
from room4d.render.gaussian4d.rasterizer import look_at_viewmat
from room4d.render.gaussian4d.seed import seed_arrays_from_geometry
from room4d.scene.scene_artifact import SceneArtifact


class SingleVideoSceneAssembler:
    """Assemble a self-contained renderable 4D scene from pipeline artifacts.

    Tolerant ``**cfg`` constructor: unknown keys (e.g. legacy ``scene`` config
    fields from the retired toy renderer) are ignored.
    """

    def __init__(self, **cfg: Any) -> None:
        # Seed (mirrors run_render_proof's ``seed`` cfg).
        self.seed_pixel_stride = int(cfg.get("seed_pixel_stride", 1))
        self.seed_max_points = int(cfg.get("seed_max_points", 1_500_000))
        self.seed_depth_near = float(cfg.get("seed_depth_near", 1e-3))
        _df = cfg.get("seed_depth_far", None)
        self.seed_depth_far = None if _df is None else float(_df)
        self.seed_scale_mult = float(cfg.get("seed_scale_mult", 1.5))
        self.seed_scale_min = float(cfg.get("seed_scale_min", 1e-5))
        self.seed_scale_max = float(cfg.get("seed_scale_max", 0.05))
        # Training views: cap to keep artifacts bounded (None = all frames).
        _mtv = cfg.get("max_train_views", None)
        self.max_train_views = None if _mtv is None else int(_mtv)
        # Render trajectory (novel-view orbit).
        self.num_render_cameras = int(cfg.get("num_render_cameras", 60))
        self.orbit_up_axis = str(cfg.get("orbit_up_axis", "z")).lower()

    @staticmethod
    def _frame_key(stream_id: str, frame_idx: int) -> str:
        return f"{stream_id}:{int(frame_idx)}"

    def _gt_image_path(self, frame_key: str, geometry_artifact) -> str | None:
        src_map = (getattr(geometry_artifact, "metadata", {}) or {}).get("debug_image_paths", {}) or {}
        path = src_map.get(frame_key)
        return str(path) if path is not None else None

    def _camera_state_for_frame(self, geometry_artifact, frame_ref: FrameRef):
        states = list((geometry_artifact.camera_states_by_stream or {}).get(frame_ref.stream_id, []))
        if not states:
            return None
        return min(states, key=lambda s: abs(float(s.time_sec) - float(frame_ref.timestamp_sec)))

    def _build_seed(self, geometry_artifact) -> dict[str, Any]:
        rgb_by_frame: dict[str, np.ndarray] = {}
        for frame in geometry_artifact.selected_frames:
            key = self._frame_key(frame.stream_id, frame.frame_idx)
            path = self._gt_image_path(key, geometry_artifact)
            if path is None:
                continue
            try:
                rgb_by_frame[key] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
        means, colors, scales = seed_arrays_from_geometry(
            geometry_artifact,
            rgb_by_frame,
            pixel_stride=self.seed_pixel_stride,
            max_points=self.seed_max_points,
            depth_near=self.seed_depth_near,
            depth_far=self.seed_depth_far,
            scale_mult=self.seed_scale_mult,
            scale_min=self.seed_scale_min,
            scale_max=self.seed_scale_max,
        )
        return {
            "means_xyz": means,
            "colors_rgb": colors,
            "scales_xyz": scales,
            "num_gaussians": int(means.shape[0]),
        }

    def _build_train_views(self, geometry_artifact) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        frames = sorted(geometry_artifact.selected_frames, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        for frame in frames:
            key = self._frame_key(frame.stream_id, frame.frame_idx)
            state = self._camera_state_for_frame(geometry_artifact, frame)
            path = self._gt_image_path(key, geometry_artifact)
            if state is None or path is None or not Path(path).exists():
                continue
            try:
                with Image.open(path) as img:
                    width, height = int(img.width), int(img.height)
            except Exception:
                continue
            intr = state.intrinsics
            views.append({
                "frame_key": key,
                "time_id": int(frame.frame_idx),
                "t_world_cam": np.asarray(state.pose.T_world_cam, dtype=np.float32),
                "intrinsics": {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.cx), "cy": float(intr.cy)},
                "width": width,
                "height": height,
                "image_path": str(path),
            })
        if self.max_train_views is not None and len(views) > self.max_train_views:
            idx = np.linspace(0, len(views) - 1, int(self.max_train_views)).round().astype(int)
            views = [views[i] for i in idx]
        return views

    def _orbit_trajectory(self, seed: dict[str, Any], train_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not train_views:
            return []
        means = np.asarray(seed.get("means_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        target = np.median(means, axis=0).astype(np.float32) if means.shape[0] else np.zeros((3,), dtype=np.float32)
        cam_centers = np.stack([np.asarray(v["t_world_cam"], dtype=np.float32)[:3, 3] for v in train_views], axis=0)
        center0 = cam_centers.mean(axis=0).astype(np.float32)
        radius = float(np.median(np.linalg.norm(cam_centers - target[None, :], axis=1)))
        if not np.isfinite(radius) or radius <= 1e-3:
            radius = 1.0
        up = np.asarray([0.0, 0.0, 1.0] if self.orbit_up_axis == "z" else [0.0, 1.0, 0.0], dtype=np.float32)
        z_idx = 2 if self.orbit_up_axis == "z" else 1
        height_offset = float(center0[z_idx] - target[z_idx])
        ref = train_views[len(train_views) // 2]
        traj: list[dict[str, Any]] = []
        for i in range(max(1, self.num_render_cameras)):
            theta = 2.0 * np.pi * i / float(max(1, self.num_render_cameras))
            offset = np.zeros((3,), dtype=np.float32)
            if self.orbit_up_axis == "z":
                offset[0], offset[1], offset[2] = radius * np.cos(theta), radius * np.sin(theta), height_offset
            else:
                offset[0], offset[2], offset[1] = radius * np.cos(theta), radius * np.sin(theta), height_offset
            eye = (target + offset).astype(np.float32)
            viewmat = np.asarray(look_at_viewmat(eye, target, up), dtype=np.float32)
            t_world_cam = np.linalg.inv(viewmat).astype(np.float32)
            traj.append({
                "t_world_cam": t_world_cam,
                "intrinsics": dict(ref["intrinsics"]),
                "width": int(ref["width"]),
                "height": int(ref["height"]),
                "time_id": int(ref["time_id"]),
            })
        return traj

    def assemble(
        self,
        *,
        run_id: str,
        save_dir,
        geometry_artifact,
        track_artifact=None,
        human_artifact=None,
        object_artifact=None,
        metadata: dict[str, Any] | None = None,
    ) -> SceneArtifact:
        seed = self._build_seed(geometry_artifact)
        train_views = self._build_train_views(geometry_artifact)
        render_trajectory = self._orbit_trajectory(seed, train_views)
        frame_refs = sorted(list(geometry_artifact.selected_frames), key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        timeline = {
            "source": "geometry",
            "time_min_sec": float(min((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "time_max_sec": float(max((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "frame_keys": [self._frame_key(fr.stream_id, fr.frame_idx) for fr in frame_refs],
        }
        notes = [
            f"seed_gaussians={seed['num_gaussians']}",
            f"train_views={len(train_views)}",
            f"render_cameras={len(render_trajectory)}",
        ]
        return SceneArtifact(
            run_id=run_id,
            stage_name="scene",
            save_dir=save_dir,
            metadata=dict(metadata or {}),
            kind="static",
            render_seed=seed,
            train_views=train_views,
            render_trajectory=render_trajectory,
            dynamic_bundle={},
            frame_refs=frame_refs,
            timeline=timeline,
            notes=notes,
        )

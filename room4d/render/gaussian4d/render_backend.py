"""Trainer-driven render backend: train + render a self-contained SceneArtifact.

Consumes ONLY the SceneArtifact (dense seed + persisted train views + render
trajectory): builds a GaussianField from the seed, optimizes it against the
train views with the proven StaticGaussianTrainer, renders the trajectory, and
emits a RenderArtifact carrying the trained checkpoint. Dynamic scenes
(``kind == "dynamic"``) land in F0b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from room4d.common.paths import ensure_dir
from room4d.common.types import CameraIntrinsics
from room4d.render.base import RenderBackend
from room4d.render.export import write_video
from room4d.render.gaussian4d.densify import DensifyConfig
from room4d.render.gaussian4d.gaussian_field import GaussianField
from room4d.render.gaussian4d.gaussian_trainer import StaticGaussianTrainer, TrainConfig
from room4d.render.gaussian4d.rasterizer import GaussianCloud, RenderCamera, build_rasterizer
from room4d.render.gaussian4d.render_artifact import RenderArtifact


def _intrinsics_obj(intr: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(fx=float(intr["fx"]), fy=float(intr["fy"]), cx=float(intr["cx"]), cy=float(intr["cy"]))


class GaussianRenderBackend(RenderBackend):
    """gsplat-backed trainer + renderer over a self-contained SceneArtifact."""

    backend_name = "gaussian"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.device = str(self.cfg.get("device", "cuda"))
        self.seed_opacity = float(self.cfg.get("seed_opacity", 0.9))
        self.fps = int(self.cfg.get("fps", 24))

    # -- seed -> trainable field ------------------------------------------
    def _build_field(self, scene_artifact) -> GaussianField:
        seed = scene_artifact.render_seed or {}
        means = np.asarray(seed.get("means_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(seed.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        scales = np.asarray(seed.get("scales_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        opacities = np.full((means.shape[0],), self.seed_opacity, dtype=np.float32)
        cloud = GaussianCloud.from_numpy(means=means, scales=scales, colors=colors, opacities=opacities, device=self.device)
        color_model = str(self.cfg.get("trainer", {}).get("color_model", "rgb"))
        return GaussianField.from_cloud(cloud, color_model=color_model)

    # -- views ------------------------------------------------------------
    def _camera(self, spec: dict[str, Any]) -> RenderCamera:
        return RenderCamera.from_world_cam(
            np.asarray(spec["t_world_cam"], dtype=np.float32),
            _intrinsics_obj(spec["intrinsics"]),
            int(spec["width"]),
            int(spec["height"]),
        )

    @staticmethod
    def _view_image_path(view: dict[str, Any]) -> str | None:
        # A loaded SceneArtifact carries the persisted ``image_file``; a fresh
        # in-memory one (post-assemble, pre-reload) still carries ``image_path``.
        return view.get("image_file") or view.get("image_path")

    def _load_gt(self, path: str, width: int, height: int) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        w, h = int(width), int(height)
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # copy -> writable (no torch warning)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device) / 255.0

    def _train_views(self, scene_artifact) -> list[tuple[RenderCamera, torch.Tensor]]:
        out: list[tuple[RenderCamera, torch.Tensor]] = []
        for view in scene_artifact.train_views:
            path = self._view_image_path(view)
            if not path:
                continue
            out.append((self._camera(view), self._load_gt(path, view["width"], view["height"])))
        return out

    # -- rendering / persistence ------------------------------------------
    def _render_trajectory(self, scene_artifact, field: GaussianField, rasterizer, save_dir: Path) -> dict[str, list[str]]:
        traj = scene_artifact.render_trajectory or []
        if not traj:
            return {}
        cloud = field.to_cloud()
        frames_dir = ensure_dir(save_dir / "flythrough")
        frame_paths: list[str] = []
        frames: list[np.ndarray] = []
        for i, spec in enumerate(traj):
            cam = self._camera(spec)
            out = rasterizer.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, cam)
            rgb = (out.rgb.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            path = frames_dir / f"frame_{i:04d}.png"
            Image.fromarray(rgb).save(path)
            frame_paths.append(str(path))
            frames.append(rgb)
        try:
            write_video(frames, save_dir / "flythrough.mp4", fps=self.fps)
        except Exception:
            pass
        return {"flythrough": frame_paths}

    def _save_checkpoint(self, field: GaussianField, save_dir: Path) -> str:
        cloud = field.to_cloud()
        path = save_dir / "trained_gaussians.npz"
        np.savez_compressed(
            path,
            means=cloud.means.detach().cpu().numpy(),
            quats=cloud.quats.detach().cpu().numpy(),
            scales=cloud.scales.detach().cpu().numpy(),
            opacities=cloud.opacities.detach().cpu().numpy(),
            colors=cloud.colors.detach().cpu().numpy(),
        )
        return path.name

    # -- entry point ------------------------------------------------------
    def fit(self, scene_artifact, *, save_dir=None, **kwargs) -> RenderArtifact:
        save_dir = Path(save_dir) if save_dir is not None else Path(scene_artifact.save_dir).parent / "render"
        ensure_dir(save_dir)
        if str(getattr(scene_artifact, "kind", "static")) == "dynamic":
            raise NotImplementedError("dynamic render backend lands in F0b")

        field = self._build_field(scene_artifact)
        train_views = self._train_views(scene_artifact)
        if not train_views or int(field.num_gaussians) == 0:
            # Degenerate scene (e.g. heuristic/synthetic geometry with no depth or
            # GT frames): nothing to train. Emit an empty render rather than crash;
            # the empty flag keeps it visible without breaking the pipeline.
            return RenderArtifact(
                run_id=getattr(scene_artifact, "run_id", "dev"),
                stage_name="render",
                save_dir=save_dir,
                metadata={"backend": self.backend_name, "device": self.device},
                scene_state={
                    "kind": "static",
                    "empty": True,
                    "num_gaussians": int(field.num_gaussians),
                    "num_train_views": len(train_views),
                    "reason": "no train_views or empty seed; render skipped",
                },
                path_index={},
            )

        densify_cfg = DensifyConfig.from_cfg(self.cfg.get("densify", {}))
        rasterizer = build_rasterizer({"device": self.device, "absgrad": densify_cfg.absgrad})
        trainer_cfg = dict(self.cfg.get("trainer", {}))
        if "iters" in self.cfg:
            trainer_cfg["iters"] = int(self.cfg["iters"])
        train_cfg = TrainConfig.from_cfg(trainer_cfg)
        trainer = StaticGaussianTrainer(field, train_cfg, densify_cfg, device=self.device, rasterizer=rasterizer)

        logs: list[dict] = []
        trainer.fit(train_views, on_log=lambda record: logs.append(record))

        path_index = self._render_trajectory(scene_artifact, field, rasterizer, save_dir)
        checkpoint = self._save_checkpoint(field, save_dir)
        eval_n = min(len(train_views), 8)
        eval_res = trainer.evaluate(train_views[:eval_n]) if eval_n else []
        train_view_psnr = float(np.mean([r["psnr"] for r in eval_res])) if eval_res else 0.0

        return RenderArtifact(
            run_id=getattr(scene_artifact, "run_id", "dev"),
            stage_name="render",
            save_dir=save_dir,
            metadata={"backend": self.backend_name, "device": self.device},
            scene_state={
                "kind": "static",
                "num_gaussians": int(field.num_gaussians),
                "iters": int(train_cfg.iters),
                "train_view_psnr": train_view_psnr,
                "checkpoint": checkpoint,
                "num_train_views": len(train_views),
            },
            path_index=path_index,
        )

    def release_resources(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

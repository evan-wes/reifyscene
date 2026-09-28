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

from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import CameraIntrinsics
from reifyscene.render.base import RenderBackend
from reifyscene.render.export import write_video
from reifyscene.render.gaussian4d.densify import DensifyConfig
from reifyscene.render.gaussian4d.gaussian_field import GaussianField
from reifyscene.render.gaussian4d.gaussian_trainer import StaticGaussianTrainer, TrainConfig, image_psnr
from reifyscene.render.gaussian4d.rasterizer import GaussianCloud, RenderCamera, build_rasterizer
from reifyscene.render.gaussian4d.render_artifact import RenderArtifact


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
        """A training photo, which must already be its camera's size.

        The scene stage writes each photo in the geometry's pixels, matching its
        camera. A photo of any other size would be resized (squashed) into a
        different image than the camera sees -- the bug this refuses.
        """
        with Image.open(path) as img:
            img = img.convert("RGB")
        w, h = int(width), int(height)
        if img.size != (w, h):
            raise ValueError(
                f"training photo {path} is {img.size[0]}x{img.size[1]} but its camera is {w}x{h}; "
                "the scene stage must write it in the geometry's pixels (a scene made before image transforms "
                "were recorded has to be rebuilt)"
            )
        arr = np.array(img, dtype=np.uint8)  # copy -> writable (no torch warning)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device) / 255.0

    def _require_image_path(self, view: dict[str, Any]) -> str:
        path = self._view_image_path(view)
        if not path:
            raise FileNotFoundError(f"training view {view.get('frame_key')!r} has no photo (image_file / image_path)")
        return path

    def _train_views(self, scene_artifact) -> list[tuple[RenderCamera, torch.Tensor]]:
        out: list[tuple[RenderCamera, torch.Tensor]] = []
        for view in scene_artifact.train_views:
            path = self._require_image_path(view)
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
            return self._fit_dynamic(scene_artifact, save_dir)

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

    # -- dynamic (F0b) ----------------------------------------------------
    def _dynamic_train_views(self, scene_artifact) -> list[tuple[RenderCamera, "torch.Tensor", int]]:
        """Train views for the dynamic trainer: (camera, gt[H,W,3], time_id)."""
        out: list[tuple[RenderCamera, torch.Tensor, int]] = []
        self._train_frame_keys = []
        for view in scene_artifact.train_views:
            path = self._require_image_path(view)
            out.append((self._camera(view), self._load_gt(path, view["width"], view["height"]), int(view.get("time_id", 0))))
            self._train_frame_keys.append(view.get("frame_key"))
        return out

    def _attach_identity(
        self, scene, train_views: list[tuple], run_dir: Path, ident_cfg: dict[str, Any]
    ) -> tuple[list[tuple], dict[str, Any]]:
        """Export fresh id maps for the run, enable identity, and pair each view with its map.

        The maps are regenerated here rather than read from disk so they cannot be stale
        against the graph they encode. A view whose frame has no map, or whose map is not
        the camera's size, is refused: training on a misaligned target is silent damage.
        """
        from reifyscene.graph.id_maps import export_id_maps, load_id_maps

        manifest = export_id_maps(run_dir)
        _, maps = load_id_maps(run_dir / "identity")
        stream = manifest["stream_id"]
        node_ids = [c["node_id"] for c in sorted(manifest["classes"], key=lambda c: c["index"])]
        scene.enable_identity(int(ident_cfg.get("dim", 20)), node_ids,
                              seed=int(ident_cfg.get("seed", 0)))
        paired: list[tuple] = []
        for (camera, gt, time_id), key in zip(train_views, self._train_frame_keys, strict=True):
            sid, _, idx = str(key).rpartition(":")
            if sid != stream or int(idx) not in maps:
                raise ValueError(f"train view {key} has no id map in {run_dir / 'identity'}")
            arr = maps[int(idx)]
            if arr.shape != tuple(gt.shape[:2]):
                raise ValueError(
                    f"id map for {key} is {arr.shape}, its photo is {tuple(gt.shape[:2])}"
                )
            paired.append((camera, gt, time_id,
                           torch.as_tensor(arr.astype(np.int64), device=self.device)))
        summary = {"dim": scene.fg.identity_dim, "num_classes": manifest["num_classes"],
                   "ignore_fraction": manifest["fractions"]["ignore"]}
        return paired, summary

    def _render_dynamic_trajectory(self, scene_artifact, scene, rasterizer, save_dir: Path) -> dict[str, list[str]]:
        traj = scene_artifact.render_trajectory or []
        if not traj:
            return {}
        frames_dir = ensure_dir(save_dir / "flythrough")
        frame_paths: list[str] = []
        frames: list[np.ndarray] = []
        for i, spec in enumerate(traj):
            cam = self._camera(spec)
            cloud = scene.cloud_at(int(spec.get("time_id", 0)))
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

    def _save_dynamic_checkpoint(self, scene, cams, source_bundle: dict, save_dir: Path) -> str:
        """Persist the trained DynamicScene as a dump_bundle npz the scorer reloads.

        Carries forward the scene-norm (and any held-out val cameras) from the
        source bundle so masked-NVS can place dataset cameras in the scene frame.
        """
        from reifyscene.render.gaussian4d.motion_init import dump_bundle

        ckpt = dump_bundle(scene, cams, factor=1)
        for key in ("scene_norm_transfm", "scene_norm_scale"):
            if key in (source_bundle or {}):
                ckpt[key] = np.asarray(source_bundle[key])
        path = save_dir / "trained_dynamic.npz"
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in ckpt.items()})
        return path.name

    def _fit_dynamic(self, scene_artifact, save_dir: Path) -> RenderArtifact:
        from reifyscene.render.gaussian4d.dynamic_trainer import DynamicGaussianTrainer, DynamicTrainConfig
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        bundle = scene_artifact.dynamic_bundle or {}
        train_views = self._dynamic_train_views(scene_artifact)
        if not bundle or not train_views:
            return RenderArtifact(
                run_id=getattr(scene_artifact, "run_id", "dev"),
                stage_name="render",
                save_dir=save_dir,
                metadata={"backend": self.backend_name, "device": self.device},
                scene_state={
                    "kind": "dynamic", "empty": True,
                    "num_train_views": len(train_views),
                    "reason": "empty dynamic bundle or no train_views; render skipped",
                },
                path_index={},
            )

        scene, cams = bundle_to_scene(bundle, device=self.device)

        dyn_cfg = dict(self.cfg.get("dynamic", self.cfg.get("trainer", {})) or {})
        iters = int(self.cfg.get("iters", dyn_cfg.get("iters", 5000)))
        train_cfg = DynamicTrainConfig.from_cfg({**dyn_cfg, "iters": iters})
        densify_cfg = DensifyConfig.from_cfg(self.cfg.get("densify"), max_iters=iters, num_frames=scene.num_frames)
        rasterizer = build_rasterizer({"device": self.device, "absgrad": densify_cfg.absgrad})
        ident_cfg = dict(dyn_cfg.get("identity") or {})
        identity_summary = None
        if ident_cfg.get("enabled"):
            # A run's stages are siblings; the render stage's parent is the run.
            train_views, identity_summary = self._attach_identity(
                scene, train_views, Path(save_dir).parent, ident_cfg
            )
            if "lambda" in ident_cfg:
                train_cfg.lambda_identity = float(ident_cfg["lambda"])
            if "lambda_3d" in ident_cfg:
                train_cfg.lambda_identity_3d = float(ident_cfg["lambda_3d"])
            if "detach_geometry" in ident_cfg:
                train_cfg.identity_detach_geometry = bool(ident_cfg["detach_geometry"])
            identity_summary["detach_geometry"] = train_cfg.identity_detach_geometry
            identity_summary["lambda_3d"] = train_cfg.lambda_identity_3d
        trainer = DynamicGaussianTrainer(scene, train_cfg, densify_cfg, device=self.device, rasterizer=rasterizer)

        logs: list[dict] = []
        trained = trainer.fit(train_views, on_log=lambda record: logs.append(record))

        path_index = self._render_dynamic_trajectory(scene_artifact, trained, rasterizer, save_dir)
        checkpoint = self._save_dynamic_checkpoint(trained, cams, bundle, save_dir)
        eval_n = min(len(train_views), 8)
        with torch.no_grad():
            psnrs = [
                float(image_psnr(trainer.render_at(v[0], v[2]), v[1]))
                for v in train_views[:eval_n]
            ]
        train_view_psnr = float(np.mean(psnrs)) if psnrs else 0.0

        return RenderArtifact(
            run_id=getattr(scene_artifact, "run_id", "dev"),
            stage_name="render",
            save_dir=save_dir,
            metadata={"backend": self.backend_name, "device": self.device},
            scene_state={
                "kind": "dynamic",
                "num_gaussians": int(trained.num_gaussians),
                "num_fg": int(trained.fg.num_gaussians),
                "num_bg": int(trained.bg.num_gaussians) if trained.bg is not None else 0,
                "num_frames": int(trained.num_frames),
                "iters": int(train_cfg.iters),
                "train_view_psnr": train_view_psnr,
                "checkpoint": checkpoint,
                "num_train_views": len(train_views),
                **({"identity": {**identity_summary, "final_identity_loss": next(
                    (r["identity_loss"] for r in reversed(logs) if "identity_loss" in r), None
                )}} if identity_summary else {}),
            },
            path_index=path_index,
        )

    def release_resources(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

from reifyscene.common.frame_keys import frame_key, parse_key
from reifyscene.common.types import FrameRef
from reifyscene.geometry.image_transform import image_transform_for
from reifyscene.render.gaussian4d.rasterizer import look_at_viewmat
from reifyscene.render.gaussian4d.seed import seed_arrays_from_geometry
from reifyscene.scene.scene_artifact import TRAIN_VIEWS_DIR, SceneArtifact, train_view_filename


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
        # Dynamic scene (F0b): ``kind="dynamic"`` swaps the static seed path for the
        # SE(3) motion-bundle path (NativeMotionInitializer over the pipeline's own
        # geometry + in-memory all-pairs tracks). Default static -> F0a unchanged.
        self.kind = str(cfg.get("kind", "static")).lower()
        self.dynamic_num_fg = int(cfg.get("dynamic_num_fg", 40_000))
        self.dynamic_num_bg = int(cfg.get("dynamic_num_bg", 100_000))
        self.dynamic_num_bases = int(cfg.get("dynamic_num_bases", 10))
        _ds = cfg.get("dynamic_seed", None)
        self.dynamic_seed = None if _ds is None else int(_ds)
        self.device = str(cfg.get("device", "cuda"))
        # Tracker cfg for the in-assembler all-pairs generation (decision: tracks
        # are produced inside the dynamic init, not the tracking stage, for F0b).
        # All-pairs seeds a dense grid in every frame and tracks all-to-all, so
        # query work is O(grid^-2 . T^2). The lift subsamples foreground to
        # ``dynamic_num_fg`` anyway, so a coarser grid (stride 8 vs SoM's 4) cuts
        # query points ~4x with negligible init quality impact. SoM's stride-4
        # default is kept on the standalone tracker (shipped-track parity);
        # override here via ``scene.tracker.grid_size``.
        self.tracker_cfg = dict(cfg.get("tracker", {}) or {})
        self.tracker_cfg.setdefault("grid_size", 8)
        # Cap tracked query points to a few multiples of the foreground budget the
        # lift keeps (~dynamic_num_fg total, ~num_fg/T per frame). 3x leaves a
        # comfortable margin for the lift's own random keep + visibility filtering
        # while cutting the dominant O(T^2) tracking work. Override via
        # ``scene.tracker.max_total_queries``.
        self.tracker_cfg.setdefault("max_total_queries", 3 * self.dynamic_num_fg)
        # FP2c motion gate ({"enabled","percentile","threshold"}): keep only
        # ever-moved tracks as deformable fg; static (e.g. SAM-2 background) -> bg.
        self.dynamic_motion_gate = dict(cfg.get("dynamic_motion_gate", {}) or {})

    @staticmethod
    def _frame_key(stream_id: str, frame_idx: int) -> str:
        return frame_key(stream_id, frame_idx)

    @staticmethod
    def _scene_metadata(metadata: dict[str, Any] | None, geometry_artifact) -> dict[str, Any]:
        """The caller's metadata plus the image transforms the training photos went through."""
        out = dict(metadata or {})
        transforms = (getattr(geometry_artifact, "metadata", {}) or {}).get("image_transforms")
        if transforms:
            out["image_transforms"] = {str(k): dict(v) for k, v in transforms.items()}
        return out

    @staticmethod
    def _photo_map(geometry_artifact) -> dict[str, str]:
        return dict((getattr(geometry_artifact, "metadata", {}) or {}).get("debug_image_paths", {}) or {})

    def _gt_image_path(self, frame_key: str, geometry_artifact) -> str | None:
        """The photo of a frame the geometry lists, or ``None`` when the geometry has no photos at all.

        Heuristic geometry carries no photos by design (an empty map), and the scene
        is then built without training views. But a frame missing from a non-empty
        map, or whose file is gone, raises: it would silently drop that frame.
        """
        src_map = self._photo_map(geometry_artifact)
        if not src_map:
            return None
        path = src_map.get(frame_key)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"geometry frame {frame_key} has no photo on disk (debug_image_paths entry: {path!r})"
            )
        return str(path)

    @staticmethod
    def _geometry_photo(path: str, geometry_artifact, stream_id: str) -> np.ndarray:
        """A frame's photo in the geometry's pixels, through its recorded image transform.

        VGGT's cameras and depth describe a resized, centre-cropped square; the photo
        is cropped and resized the same way (never squashed) so it matches them.
        """
        with Image.open(path) as img:
            photo = img.convert("RGB")
        transform = image_transform_for(geometry_artifact, str(stream_id), source_size=photo.size)
        return transform.apply_to_image(photo)

    def _camera_state_for_frame(self, geometry_artifact, frame_ref: FrameRef):
        states = list((geometry_artifact.camera_states_by_stream or {}).get(frame_ref.stream_id, []))
        if not states:
            return None
        return min(states, key=lambda s: abs(float(s.time_sec) - float(frame_ref.timestamp_sec)))

    def _build_seed(self, geometry_artifact) -> dict[str, Any]:
        rgb_by_frame: dict[str, np.ndarray] = {}
        for frame in geometry_artifact.selected_frames:
            key = self._frame_key(frame.stream_id, frame.frame_idx)
            if key not in (geometry_artifact.depthmaps_by_frame or {}):
                continue  # no depth to seed from: the frame contributes no seed points
            path = self._gt_image_path(key, geometry_artifact)
            if path is None:
                continue
            rgb_by_frame[key] = self._geometry_photo(path, geometry_artifact, frame.stream_id)
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

    def _write_train_photos(self, views: list[dict[str, Any]], geometry_artifact, save_dir) -> list[dict[str, Any]]:
        """Write each view's photo, in the geometry's pixels, to its final place in the scene.

        The file lands exactly where :meth:`SceneArtifact.save` keeps it, and the
        view's ``image_path`` points there -- so the render stage trains on the same
        pixels whether it gets this in-memory scene or reloads the saved one. The
        photo must come out at the view's camera size, or it would not match it.
        """
        views_dir = Path(save_dir) / TRAIN_VIEWS_DIR
        views_dir.mkdir(parents=True, exist_ok=True)
        out: list[dict[str, Any]] = []
        for idx, view in enumerate(views):
            view = dict(view)
            source = view.pop("source_image_path")
            rgb = self._geometry_photo(source, geometry_artifact, parse_key(view["frame_key"])[0])
            if rgb.shape[:2] != (int(view["height"]), int(view["width"])):
                raise ValueError(
                    f"training view {view['frame_key']}: photo comes out {rgb.shape[1]}x{rgb.shape[0]} in the geometry's "
                    f"pixels but its camera is {view['width']}x{view['height']}"
                )
            dst = views_dir / train_view_filename(idx, str(view["frame_key"]))
            Image.fromarray(rgb).save(dst)
            view["image_path"] = str(dst.resolve())
            out.append(view)
        return out

    def _build_train_views(self, geometry_artifact, save_dir) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        frames = sorted(geometry_artifact.selected_frames, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        for frame in frames:
            key = self._frame_key(frame.stream_id, frame.frame_idx)
            state = self._camera_state_for_frame(geometry_artifact, frame)
            if state is None:
                continue  # no camera for this frame: it cannot be a training view
            path = self._gt_image_path(key, geometry_artifact)
            if path is None:
                continue  # the geometry carries no photos at all (heuristic)
            with Image.open(path) as img:
                photo_size = (int(img.width), int(img.height))
            transform = image_transform_for(geometry_artifact, str(frame.stream_id), source_size=photo_size)
            width, height = transform.output_size
            intr = state.intrinsics
            views.append({
                "frame_key": key,
                "time_id": int(frame.frame_idx),
                "t_world_cam": np.asarray(state.pose.T_world_cam, dtype=np.float32),
                "intrinsics": {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.cx), "cy": float(intr.cy)},
                "width": int(width),
                "height": int(height),
                "source_image_path": str(path),
            })
        if self.max_train_views is not None and len(views) > self.max_train_views:
            idx = np.linspace(0, len(views) - 1, int(self.max_train_views)).round().astype(int)
            views = [views[i] for i in idx]
        return self._write_train_photos(views, geometry_artifact, save_dir)

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

    # ------------------------------------------------------------ dynamic (F0b)
    @staticmethod
    def _fold_ground_plane_into_norm(transfm, geometry_artifact):
        """Compose the geometry's ground-plane transform into a scene_norm ``transfm``.

        The dynamic init computes ``scene_norm`` on the *already ground-plane-canonicalized*
        geometry (the provider serves canonicalized poses/points), so ``scene_norm.transfm``
        only maps canonicalized → scene. Held-out val cameras are placed at scoring time from
        **raw GT** poses via this transfm; without the Z-up rotation they land ~45° off and
        render the wrong place (masked-NVS ~5 vs ~15). Convention (``world_frame.apply`` →
        ``_transform_pose = T_gp @ T_world_cam`` ⇒ ``w2c_new = w2c_old @ inv(T_gp)``): the full
        raw-world → scene placement transfm is ``scene_norm.transfm @ T_gp``. No-op when no
        ground plane ran (``world_frame`` absent).
        """
        gp = ((geometry_artifact.metadata or {}).get("world_frame") or {}).get("transform_world_old_to_new")
        if gp is None:
            return transfm
        import torch  # noqa: PLC0415 — heavy dep, only reached on the dynamic path
        t_gp = torch.as_tensor(np.asarray(gp, dtype=np.float32), dtype=transfm.dtype, device=transfm.device)
        return transfm @ t_gp

    def _fg_mask_by_frame(self, instance_artifact) -> dict[str, np.ndarray]:
        """Per-frame union of all segment instance masks -> the dynamic foreground.

        Keyed ``"{stream_id}:{frame_idx}"`` to match ArtifactGeometryProvider. The
        union of every instance's mask in a frame is the foreground the native lift
        seeds tracks in. Empty when there is no segment artifact, which lifts no
        foreground -> a degenerate (effectively static) scene.
        """
        if instance_artifact is None:
            return {}
        mask_arrays = getattr(instance_artifact, "mask_arrays", {}) or {}
        frame_index = getattr(instance_artifact, "frame_index", {}) or {}
        if not frame_index:
            # frame_index is populated at save-time (rebuild_indices); an in-memory
            # (pre-save) segment artifact has it empty — rebuild from observations so the
            # live dynamic scene sees the foreground (mirrors InstanceArtifact.save()).
            frame_index = instance_artifact._build_frame_index(getattr(instance_artifact, "observations", []) or [])
        out: dict[str, np.ndarray] = {}
        for key, instance_ids in frame_index.items():
            union: np.ndarray | None = None
            for iid in instance_ids:
                mask = mask_arrays.get(iid)
                if mask is None:
                    continue
                m = np.asarray(mask, dtype=bool)
                if union is None:
                    union = m.copy()
                elif union.shape != m.shape:
                    raise ValueError(f"{key}: instance masks of one frame have different "
                                     f"rasters {union.shape} and {m.shape}")
                else:
                    union |= m
            if union is not None:
                out[key] = union.astype(np.float32)
        return out

    def _dynamic_train_views(self, cams_train, height: int, width: int, geometry_artifact, save_dir) -> list[dict[str, Any]]:
        """Training views from the (scene-normalized) dynamic cameras + GT frames.

        Cameras are the normalized train cameras the init produced (so they sit in
        the gaussians' frame); intrinsics/size are the lift resolution, i.e. the
        geometry's pixels; each GT photo is put into those pixels through the
        geometry's recorded image transform and written into the scene.
        """
        views: list[dict[str, Any]] = []
        for cam in cams_train:
            key = str(cam.frame_name)
            path = self._gt_image_path(key, geometry_artifact)
            if path is None:
                raise FileNotFoundError(f"dynamic training camera {key} has no photo (the geometry lists none)")
            t_world_cam = np.linalg.inv(np.asarray(cam.w2c, dtype=np.float32)).astype(np.float32)
            K = np.asarray(cam.K, dtype=np.float32)
            views.append({
                "frame_key": key,
                "time_id": int(cam.time_id),
                "t_world_cam": t_world_cam,
                "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
                "width": int(width),
                "height": int(height),
                "source_image_path": str(path),
            })
        return self._write_train_photos(views, geometry_artifact, save_dir)

    def _dynamic_orbit_trajectory(self, scene, train_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Novel-view orbit for a dynamic scene: the static orbit + a time sweep."""
        if not train_views:
            return []
        means = scene.cloud_at(0).means.detach().cpu().numpy().astype(np.float32)
        traj = self._orbit_trajectory({"means_xyz": means}, train_views)
        num_frames = int(scene.num_frames)
        n = max(1, len(traj))
        for i, cam in enumerate(traj):
            cam["time_id"] = int(round(i / max(n - 1, 1) * max(num_frames - 1, 0)))
        return traj

    def _assemble_dynamic(
        self, *, run_id: str, save_dir, geometry_artifact, instance_artifact,
        metadata: dict[str, Any] | None, frame_refs, timeline,
    ) -> SceneArtifact:
        """Build a dynamic SE(3) scene from the pipeline's own geometry + tracks.

        Geometry comes from an ``ArtifactGeometryProvider`` over the (pose-refined)
        geometry artifact; foreground from the segment masks; the all-pairs 2D
        tracks are generated in-memory here (decision: tracks live in the dynamic
        init for F0b, not the tracking stage). The built ``DynamicScene`` + cameras
        are serialized into ``dynamic_bundle`` for the render stage.
        """
        from reifyscene.lifting.geometry_provider import ArtifactGeometryProvider
        from reifyscene.render.gaussian4d.motion_init import NativeMotionInitializer, dump_bundle
        from reifyscene.tracking.tapir_wrapper import TAPIRWrapper

        fg_by_frame = self._fg_mask_by_frame(instance_artifact)
        provider = ArtifactGeometryProvider(geometry_artifact, fg_mask_by_frame=fg_by_frame)
        # NOTE: build_native_init_inputs re-reads frame_stack internally; we read it
        # once more here for the all-pairs frames + the lift resolution (acceptable
        # for v1 -- it reads the in-memory artifact, not disk-heavy dataset loaders).
        stack = provider.frame_stack("train", device="cpu")
        if stack.num_frames == 0:
            return SceneArtifact(
                run_id=run_id, stage_name="scene", save_dir=save_dir, metadata=dict(metadata or {}),
                kind="dynamic", render_seed={}, train_views=[], render_trajectory=[],
                dynamic_bundle={}, frame_refs=frame_refs, timeline=timeline,
                notes=["dynamic: no usable geometry frames"],
            )
        height, width = int(stack.depths.shape[1]), int(stack.depths.shape[2])
        frames = stack.rgbs.cpu().numpy()
        masks = stack.fg.cpu().numpy()
        frame_names = list(stack.cameras.frame_names)

        # Robustness: the dynamic init lifts tracks only inside the foreground. If the fg
        # masks don't cover these (geometry-timeline) frames, the lifted foreground is empty
        # and scene-normalization's quantile crashes on an empty tensor. Gracefully fall back
        # to a static scene (the documented "no foreground → effectively static" intent).
        frames_with_fg = int((masks.reshape(len(masks), -1) > 0.5).any(axis=1).sum())
        if frames_with_fg < 2:
            return self._assemble_static(
                run_id=run_id, save_dir=save_dir, geometry_artifact=geometry_artifact,
                metadata=metadata, frame_refs=frame_refs, timeline=timeline,
                extra_notes=[f"dynamic→static fallback: foreground in only {frames_with_fg}/{len(masks)} "
                             "frames (segment masks don't cover the geometry timeline)."],
            )

        try:
            all_pairs = TAPIRWrapper(self.tracker_cfg).compute_all_pairs(frames, masks, frame_names, factor=1)
            scene, cams, norm = NativeMotionInitializer(
                dataset_root="",  # unused once provider + pair_loader supply geometry + tracks
                factor=1, num_fg=self.dynamic_num_fg, num_bg=self.dynamic_num_bg,
                num_bases=self.dynamic_num_bases, seed=self.dynamic_seed,
                provider=provider, pair_loader=all_pairs.pair,
                motion_gate=self.dynamic_motion_gate,
            ).build(device=self.device, return_norm=True)
        except (RuntimeError, ValueError) as exc:
            return self._assemble_static(
                run_id=run_id, save_dir=save_dir, geometry_artifact=geometry_artifact,
                metadata=metadata, frame_refs=frame_refs, timeline=timeline,
                extra_notes=[f"dynamic→static fallback: dynamic init failed ({type(exc).__name__}: {exc})."],
            )

        norm.transfm = self._fold_ground_plane_into_norm(norm.transfm, geometry_artifact)
        bundle = dump_bundle(scene, cams, factor=1, scene_norm=norm)
        train_views = self._dynamic_train_views(cams.get("train", []), height, width, geometry_artifact, save_dir)
        render_trajectory = self._dynamic_orbit_trajectory(scene, train_views)
        notes = [
            f"dynamic_fg={scene.fg.num_gaussians}",
            f"dynamic_bg={scene.bg.num_gaussians if scene.bg is not None else 0}",
            f"motion_bases={scene.motion_bases.num_bases}",
            f"train_views={len(train_views)}",
            f"fg_frames={len(fg_by_frame)}",
            f"render_cameras={len(render_trajectory)}",
        ]
        return SceneArtifact(
            run_id=run_id, stage_name="scene", save_dir=save_dir,
            metadata=self._scene_metadata(metadata, geometry_artifact),
            kind="dynamic", render_seed={}, train_views=train_views,
            render_trajectory=render_trajectory, dynamic_bundle=bundle,
            frame_refs=frame_refs, timeline=timeline, notes=notes,
        )

    def assemble(
        self,
        *,
        run_id: str,
        save_dir,
        geometry_artifact,
        track_artifact=None,
        human_artifact=None,
        object_artifact=None,
        instance_artifact=None,
        metadata: dict[str, Any] | None = None,
    ) -> SceneArtifact:
        frame_refs = sorted(list(geometry_artifact.selected_frames), key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        timeline = {
            "source": "geometry",
            "time_min_sec": float(min((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "time_max_sec": float(max((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "frame_keys": [self._frame_key(fr.stream_id, fr.frame_idx) for fr in frame_refs],
        }
        if self.kind == "dynamic":
            return self._assemble_dynamic(
                run_id=run_id, save_dir=save_dir, geometry_artifact=geometry_artifact,
                instance_artifact=instance_artifact, metadata=metadata,
                frame_refs=frame_refs, timeline=timeline,
            )

        return self._assemble_static(
            run_id=run_id, save_dir=save_dir, geometry_artifact=geometry_artifact,
            metadata=metadata, frame_refs=frame_refs, timeline=timeline,
        )

    def _assemble_static(self, *, run_id, save_dir, geometry_artifact, metadata, frame_refs, timeline,
                         extra_notes: list[str] | None = None) -> SceneArtifact:
        """Build a static scene (depth-backprojected seed). Also the graceful fallback for
        a degenerate dynamic scene (no usable foreground)."""
        seed = self._build_seed(geometry_artifact)
        train_views = self._build_train_views(geometry_artifact, save_dir)
        render_trajectory = self._orbit_trajectory(seed, train_views)
        notes = list(extra_notes or []) + [
            f"seed_gaussians={seed['num_gaussians']}",
            f"train_views={len(train_views)}",
            f"render_cameras={len(render_trajectory)}",
        ]
        return SceneArtifact(
            run_id=run_id, stage_name="scene", save_dir=save_dir,
            metadata=self._scene_metadata(metadata, geometry_artifact),
            kind="static", render_seed=seed, train_views=train_views,
            render_trajectory=render_trajectory, dynamic_bundle={},
            frame_refs=frame_refs, timeline=timeline, notes=notes,
        )

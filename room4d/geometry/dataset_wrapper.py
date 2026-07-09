"""Dataset-backed geometry initializer.

Reads pre-computed DyCheck/Nerfies-format camera poses, intrinsics, and depth
from disk and emits a standard :class:`GeometryArtifact`, so borrowed dataset
geometry flows through room4d's real contracts (bypassing on-the-fly SfM/VGGT).
One backend handles both NVIDIA Dynamic Scenes (e.g. Balloon1) and DyCheck
iPhone (e.g. paper-windmill): both ship the same on-disk layout::

    <root>/camera/{cam}_{time:05d}.json   # focal_length, principal_point,
                                          # image_size, orientation (world->cam R),
                                          # position (camera center in world)
    <root>/depth/{factor}x/{name}.npy     # metric depth, HxW(x1), meters
    <root>/rgb/{factor}x/{name}.png
    <root>/splits/{train,val}.json        # camera_ids / frame_names / time_ids

Camera convention (matches Shape-of-Motion's iPhone loader): the JSON stores a
world-to-camera rotation ``R`` (``orientation``) and the camera center ``p``
(``position``), so ``w2c = [R | -R p]`` and the room4d pose is
``T_world_cam = inv(w2c)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room4d.common.depth import unproject_depth_to_world
from room4d.common.io import read_json
from room4d.common.types import (
    CameraIntrinsics,
    CameraPose,
    FrameRef,
    TimedCameraState,
)
from room4d.geometry.base import GeometryInitializer
from room4d.geometry.geometry_artifact import GeometryArtifact


class DatasetGeometryInitializer(GeometryInitializer):
    """Emit a GeometryArtifact from on-disk DyCheck-format dataset poses/depth."""

    backend_name = "dataset"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg
        if not cfg.get("dataset_root"):
            raise ValueError(
                "DatasetGeometryInitializer requires cfg['dataset_root'] "
                "(path to a DyCheck-format sequence directory)."
            )
        self.dataset_root = Path(cfg["dataset_root"])
        self.factor = int(cfg.get("factor", 1))
        self.split = str(cfg.get("split", "train"))
        # time_sec = time_id / fps; fps=1.0 makes time_sec == time_id (a stable
        # monotonic key, which is all downstream ordering needs here).
        self.fps = float(cfg.get("fps", 1.0))
        self.name_pad = int(cfg.get("name_pad", 5))
        self.run_id = str(cfg.get("run_id", f"dataset:{self.dataset_root.name}"))
        self.save_dir = Path(cfg.get("geometry_save_dir", "artifacts/geometry_dataset"))

    # -- path helpers ------------------------------------------------------

    @property
    def camera_dir(self) -> Path:
        return self.dataset_root / "camera"

    @property
    def depth_dir(self) -> Path:
        return self.dataset_root / "depth" / f"{self.factor}x"

    @property
    def rgb_dir(self) -> Path:
        return self.dataset_root / "rgb" / f"{self.factor}x"

    def frame_name(self, stream_id: Any, frame_idx: int) -> str:
        """Dataset frame name ``{camera}_{time:0Nd}`` for a (stream, frame)."""
        return f"{stream_id}_{int(frame_idx):0{self.name_pad}d}"

    # -- dataset readers ---------------------------------------------------

    def _distortion(self, cam: dict[str, Any]) -> np.ndarray | None:
        radial = list(cam.get("radial_distortion", []) or [])
        tangential = list(cam.get("tangential_distortion", []) or [])
        k1 = radial[0] if len(radial) > 0 else 0.0
        k2 = radial[1] if len(radial) > 1 else 0.0
        k3 = radial[2] if len(radial) > 2 else 0.0
        p1 = tangential[0] if len(tangential) > 0 else 0.0
        p2 = tangential[1] if len(tangential) > 1 else 0.0
        params = np.asarray([k1, k2, p1, p2, k3], dtype=float)
        if np.allclose(params, 0.0):
            return None
        return params

    def load_camera(self, frame_name: str) -> tuple[CameraIntrinsics, np.ndarray, tuple[int, int]]:
        """Return (intrinsics, T_world_cam 4x4, (width, height)) for a frame.

        Intrinsics and image size are scaled by ``factor`` to match the
        ``{factor}x`` rgb/depth resolution actually used downstream.
        """
        cam = read_json(self.camera_dir / f"{frame_name}.json")
        full_w, full_h = (int(round(float(v))) for v in cam["image_size"])
        factor = self.factor
        focal = float(cam["focal_length"])
        aspect = float(cam.get("pixel_aspect_ratio", 1.0))
        cx, cy = (float(v) for v in cam["principal_point"])
        intrinsics = CameraIntrinsics(
            fx=focal / factor,
            fy=(focal * aspect) / factor,
            cx=cx / factor,
            cy=cy / factor,
            distortion_model="brown",
            distortion_params=self._distortion(cam),
        )
        R = np.asarray(cam["orientation"], dtype=np.float64).reshape(3, 3)
        p = np.asarray(cam["position"], dtype=np.float64).reshape(3)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ p
        t_world_cam = np.linalg.inv(w2c).astype(np.float32)
        return intrinsics, t_world_cam, (full_w // factor, full_h // factor)

    def load_depth(self, frame_name: str) -> np.ndarray:
        """Metric depth (HxW float32) for a frame; squeezes a trailing channel."""
        arr = np.load(self.depth_dir / f"{frame_name}.npy")
        return np.squeeze(np.asarray(arr, dtype=np.float32))

    def load_rgb(self, frame_name: str) -> np.ndarray:
        """RGB image (HxWx3 uint8) for a frame."""
        with Image.open(self.rgb_dir / f"{frame_name}.png") as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def load_fg_mask(self, frame_name: str) -> np.ndarray:
        """Foreground (dynamic-object) mask (HxW float in [0, 1]) for a frame.

        DyCheck/SoM sequences ship Track-Anything masks under
        ``flow3d_preprocessed/track_anything/{factor}x/``; 1 = foreground. Used
        by the native motion initializer to split fg trajectories from bg points.
        """
        path = self.dataset_root / "flow3d_preprocessed" / "track_anything" / f"{self.factor}x" / f"{frame_name}.png"
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def load_covisible(self, frame_name: str, split: str = "val") -> np.ndarray:
        """Co-visibility mask (HxW bool) for a held-out frame, True = covisible.

        DyCheck/NVIDIA sequences ship these under covisible/{factor}x/{split}/.
        """
        path = self.dataset_root / "covisible" / f"{self.factor}x" / split / f"{frame_name}.png"
        with Image.open(path) as img:
            return np.asarray(img.convert("L")) > 0

    def split_frames(self, split: str | None = None) -> list[tuple[str, int]]:
        """Return (camera_id, time_id) pairs from a split json."""
        split = split or self.split
        meta = read_json(self.dataset_root / "splits" / f"{split}.json")
        cameras = [str(c) for c in meta["camera_ids"]]
        times = [int(t) for t in meta["time_ids"]]
        return list(zip(cameras, times))

    # -- artifact construction --------------------------------------------

    def _resolve_specs(self, frames, ingest_artifact) -> list[tuple[str, int, int]]:
        """Resolve the (stream_id, frame_idx, time_id) list to load."""
        if frames:
            source = frames
        elif ingest_artifact is not None and getattr(ingest_artifact, "keyframes", None):
            source = ingest_artifact.keyframes
        else:
            # Standalone (render-proof) mode: enumerate the configured split.
            specs = [(cam, time_id, time_id) for cam, time_id in self.split_frames()]
            cameras = self.cfg.get("cameras")
            if cameras is not None:
                allow = {str(c) for c in cameras}
                specs = [s for s in specs if s[0] in allow]
            stride = max(1, int(self.cfg.get("frame_stride", 1)))
            specs = specs[::stride]
            max_frames = self.cfg.get("max_frames")
            if max_frames is not None:
                specs = specs[: int(max_frames)]
            return specs
        return [(str(fr.stream_id), int(fr.frame_idx), int(fr.frame_idx)) for fr in source]

    def initialize(
        self,
        frames=None,
        *,
        ingest_artifact=None,
        sync_artifact=None,
    ) -> GeometryArtifact:
        specs = self._resolve_specs(frames, ingest_artifact)
        if not specs:
            raise ValueError(
                f"DatasetGeometryInitializer found no frames for split '{self.split}' "
                f"under {self.dataset_root}."
            )

        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}
        debug_image_paths: dict[str, str] = {}
        selected_frames: list[FrameRef] = []

        for stream_id, frame_idx, time_id in specs:
            stream_id = str(stream_id)
            frame_idx = int(frame_idx)
            name = self.frame_name(stream_id, frame_idx)
            intrinsics, t_world_cam, _wh = self.load_camera(name)
            depth = np.squeeze(np.asarray(self.load_depth(name), dtype=np.float32))
            time_sec = float(time_id) / self.fps

            frame_ref = FrameRef(stream_id=stream_id, frame_idx=frame_idx, timestamp_sec=time_sec)
            selected_frames.append(frame_ref)
            intrinsics_by_stream[stream_id] = intrinsics
            camera_states_by_stream.setdefault(stream_id, []).append(
                TimedCameraState(
                    stream_id=stream_id,
                    time_sec=time_sec,
                    intrinsics=intrinsics,
                    pose=CameraPose(T_world_cam=t_world_cam),
                )
            )
            # Complete oracle geometry: back-project GT depth to WORLD points (VGGT
            # convention), NaN at depth holes with a confidence = validity mask, so
            # ground-plane + every point-map consumer treat it like any other backend.
            world_pts, valid = unproject_depth_to_world(depth, intrinsics, t_world_cam)
            conf = valid.astype(np.float32)
            key = f"{stream_id}:{frame_idx}"
            depthmaps_by_frame[key] = {"depth": depth, "confidence": conf}
            pointmaps_by_frame[key] = {"points": world_pts, "confidence": conf}
            # Wire the GT RGB frame so scene/render seeding + debug export can find it
            # (same metadata['debug_image_paths'] contract VGGT populates).
            rgb_path = self.rgb_dir / f"{name}.png"
            if rgb_path.exists():
                debug_image_paths[key] = str(rgb_path.resolve())

        return GeometryArtifact(
            run_id=self.run_id,
            stage_name="geometry",
            save_dir=self.save_dir,
            backend_name=self.backend_name,
            selected_frames=selected_frames,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            depthmaps_by_frame=depthmaps_by_frame,
            pointmaps_by_frame=pointmaps_by_frame,
            metadata={"debug_image_paths": debug_image_paths},
            notes=[
                f"dataset_root={self.dataset_root}",
                f"factor={self.factor}",
                f"split={self.split}",
                f"frames={len(selected_frames)}",
                "GT-oracle geometry: poses + depth + world point maps (depth holes → NaN, "
                "confidence = validity); matches the VGGT/fused point-map contract.",
            ],
        )

"""Wrapper module for the VGGT coarse geometry initializer.

This implementation keeps the existing ``GeometryArtifact`` contract unchanged,
while providing a real adapter path for the official VGGT package when it is
installed locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from room4d.geometry.base import GeometryInitializer
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.ingest.decoder import VideoDecoder


class VGGTInitializer(GeometryInitializer):
    """Foundation-model wrapper for coarse geometry initialization.

    Backend strategy
    ----------------
    - ``backend_mode='real'``: require the official VGGT package and fail if it
      is unavailable.
    - ``backend_mode='auto'``: use VGGT when available, otherwise fall back to a
      deterministic heuristic initializer so the prototype remains runnable.
    - ``backend_mode='heuristic'``: skip VGGT and emit simple geometric priors.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """Store backend configuration.

        Supported config keys
        ---------------------
        backend_mode:
            One of ``auto``, ``real``, or ``heuristic``.
        hf_model_id:
            Hugging Face model identifier to pass to ``VGGT.from_pretrained``.
        checkpoint_path:
            Optional local checkpoint path for manual loading.
        device:
            Torch device string. Defaults to ``cuda`` when available.
        use_autocast:
            Enable autocast during inference.
        frame_cache_dir:
            Optional directory where selected frames are materialized before
            being passed into the official VGGT preprocessing helper.
        """
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get("backend_mode", "auto"))
        self.hf_model_id = str(self.cfg.get("hf_model_id", "facebook/VGGT-1B"))
        self.checkpoint_path = self.cfg.get("checkpoint_path")
        self.device = str(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.use_autocast = bool(self.cfg.get("use_autocast", torch.cuda.is_available()))
        self.frame_cache_dir = self.cfg.get("frame_cache_dir")
        self.debug_shapes = bool(self.cfg.get("debug_shapes", False))
        self.save_depth_pngs = bool(self.cfg.get("save_depth_pngs", False))
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get("default_fps", 30.0)))
        resize_max_dim = self.cfg.get("resize_max_dim")
        self.resize_max_dim = None if resize_max_dim in (None, 0) else int(resize_max_dim)
        self.cpu_threads = self.cfg.get("cpu_threads")

        self._is_loaded = False
        self._model: Any | None = None
        self._load_and_preprocess_images = None
        self._pose_encoding_to_extri_intri = None
        self._unproject_depth_map_to_point_map = None
        self._backend_error: Exception | None = None

    def _load_backend_components(self) -> None:
        """Import the official VGGT modules lazily."""
        from vggt.models.vggt import VGGT  # type: ignore[import-not-found]
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore[import-not-found]
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[import-not-found]
        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore[import-not-found]

        self._VGGT = VGGT
        self._load_and_preprocess_images = load_and_preprocess_images
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self._unproject_depth_map_to_point_map = unproject_depth_map_to_point_map

    def load_checkpoint(self) -> None:
        """Load the VGGT backend if available.

        Notes
        -----
        The official quick-start shows loading via
        ``VGGT.from_pretrained('facebook/VGGT-1B')`` and then using
        ``load_and_preprocess_images`` before calling ``model(images)``. The
        same path is used here when the package is installed locally.
        """
        if self.backend_mode == "heuristic":
            self._is_loaded = False
            return

        try:
            if self.cpu_threads is not None and not self.device.startswith("cuda"):
                try:
                    torch.set_num_threads(max(1, int(self.cpu_threads)))
                except Exception:
                    pass
                try:
                    torch.set_num_interop_threads(max(1, int(self.cpu_threads) // 2))
                except Exception:
                    pass
            self._load_backend_components()
            if self.checkpoint_path:
                model = self._VGGT()
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                model = self._VGGT.from_pretrained(self.hf_model_id)
            model = model.to(self.device)
            model.eval()
            self._model = model
            self._is_loaded = True
        except Exception as exc:  # pragma: no cover - exercised in integration usage
            self._backend_error = exc
            self._is_loaded = False
            if self.backend_mode == "real":
                raise RuntimeError(
                    "Failed to initialize the real VGGT backend. Install the official "
                    "facebookresearch/vggt package and required dependencies, or switch "
                    "backend_mode to 'auto' or 'heuristic'."
                ) from exc

    def prepare_inputs(self, frames: list[FrameRef]) -> list[dict[str, Any]]:
        """Convert frame references into backend-ready request records."""
        return [
            {
                "stream_id": frame.stream_id,
                "frame_idx": frame.frame_idx,
                "timestamp_sec": frame.timestamp_sec,
            }
            for frame in frames
        ]

    def _frame_key(self, frame: FrameRef) -> str:
        """Return a stable key for per-frame artifact maps."""
        return f"{frame.stream_id}:{frame.frame_idx}"

    def _select_frames(self, frames: list[FrameRef], ingest_artifact) -> list[FrameRef]:
        """Resolve explicit or ingest-provided frame selections."""
        if frames:
            return list(frames)
        if ingest_artifact is not None:
            return list(ingest_artifact.keyframes)
        return []

    def _materialize_images(
        self,
        frames: list[FrameRef],
        ingest_artifact,
        target_root: Path,
        *,
        resize_max_dim: int | None = None,
    ) -> tuple[list[Path], list[str]]:
        """Decode and save selected frames to disk for VGGT preprocessing."""
        if ingest_artifact is None:
            raise ValueError("ingest_artifact is required for the real VGGT backend.")

        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        image_paths: list[Path] = []
        notes: list[str] = []
        applied_resize = self.resize_max_dim if resize_max_dim is None else resize_max_dim
        applied_resize = None if applied_resize in (None, 0) else int(applied_resize)
        resized_any = False

        target_root.mkdir(parents=True, exist_ok=True)
        for old_png in target_root.glob("*.png"):
            old_png.unlink()
        frame_iter = enumerate(frames)
        if self.debug_shapes:
            frame_iter = track(frame_iter, total=len(frames), description="[cyan]VGGT · materializing frames")
        for idx, frame in frame_iter:
            stream = stream_lookup[frame.stream_id]
            image = self.decoder.get_frame(stream, frame.frame_idx)
            pil_image = Image.fromarray(image)
            if applied_resize is not None:
                width, height = pil_image.size
                max_dim = max(width, height)
                if max_dim > applied_resize:
                    scale = float(applied_resize) / float(max_dim)
                    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                    pil_image = pil_image.resize(new_size, resample=Image.BILINEAR)
                    resized_any = True
            out_path = target_root / f"{idx:04d}_{frame.stream_id}_{frame.frame_idx:06d}.png"
            pil_image.save(out_path)
            image_paths.append(out_path)

        notes.append(f"Materialized VGGT inputs under cache: {target_root}")
        if applied_resize is not None:
            if resized_any:
                notes.append(f"Materialized images were resized to max_dim={applied_resize} before backend execution.")
            else:
                notes.append(f"Materialized images fit within resize_max_dim={applied_resize}; no resize was needed.")
        return image_paths, notes

    def _numpyify(self, value: Any) -> Any:
        """Recursively convert tensors to CPU numpy arrays."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        if isinstance(value, dict):
            return {k: self._numpyify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._numpyify(v) for v in value)
        return value

    def _debug_describe_value(
        self,
        name: str,
        value: Any,
        *,
        prefix: str = "[VGGT DEBUG]",
        max_depth: int = 3,
        _depth: int = 0,
    ) -> None:
        """Recursively print shape/type/device information for model outputs."""
        indent = "  " * _depth

        if _depth > max_depth:
            print(f"{prefix} {indent}{name}: <max depth reached>")
            return

        if value is None:
            print(f"{prefix} {indent}{name}: None")
            return

        try:
            if isinstance(value, torch.Tensor):
                print(
                    f"{prefix} {indent}{name}: "
                    f"torch.Tensor shape={tuple(value.shape)} "
                    f"dtype={value.dtype} device={value.device} "
                    f"requires_grad={value.requires_grad}"
                )
                return
        except Exception:
            pass

        if isinstance(value, np.ndarray):
            print(
                f"{prefix} {indent}{name}: "
                f"np.ndarray shape={value.shape} dtype={value.dtype}"
            )
            return

        if isinstance(value, dict):
            print(f"{prefix} {indent}{name}: dict keys={list(value.keys())}")
            for k, v in value.items():
                self._debug_describe_value(
                    str(k),
                    v,
                    prefix=prefix,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            return

        if isinstance(value, (list, tuple)):
            print(f"{prefix} {indent}{name}: {type(value).__name__} len={len(value)}")
            for i, item in enumerate(value):
                self._debug_describe_value(
                    f"[{i}]",
                    item,
                    prefix=prefix,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            return

        print(f"{prefix} {indent}{name}: type={type(value).__name__} value={value!r}")

    def _debug_dump_predictions(self, predictions: Any) -> None:
        """Print a structured summary of the raw VGGT predictions."""
        print("[VGGT DEBUG] ---- raw prediction summary ----")
        self._debug_describe_value("predictions", predictions)
        print("[VGGT DEBUG] --------------------------------")

    def _debug_dump_selected_outputs(
        self,
        *,
        extrinsic: Any = None,
        intrinsic: Any = None,
        depth: Any = None,
        point_map: Any = None,
        depth_conf: Any = None,
        point_conf: Any = None,
    ) -> None:
        """Print a focused summary of extracted outputs before normalization."""
        print("[VGGT DEBUG] ---- selected outputs before normalization ----")
        self._debug_describe_value("extrinsic", extrinsic)
        self._debug_describe_value("intrinsic", intrinsic)
        self._debug_describe_value("depth", depth)
        self._debug_describe_value("point_map", point_map)
        self._debug_describe_value("depth_conf", depth_conf)
        self._debug_describe_value("point_conf", point_conf)
        print("[VGGT DEBUG] ----------------------------------------------")

    def _intrinsics_from_matrix(self, matrix: np.ndarray) -> CameraIntrinsics:
        """Convert a 3x3 intrinsic matrix into the internal dataclass."""
        return CameraIntrinsics(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            distortion_model="brown",
            distortion_params=np.zeros(4, dtype=float),
        )


    def release_resources(self) -> None:
        self._model = None
        self._torch = None
        self._VGGT = None
        self._load_and_preprocess_images = None
        self._cache_dir = None
        self._is_loaded = False
        self._backend_error = None

    def _normalize_per_frame_array(self, value: Any, *, item_ndim: int, num_frames: int) -> Any:
        """Normalize model outputs to an explicit per-frame leading dimension.

        Parameters
        ----------
        value:
            Tensor/array-like model output or ``None``.
        item_ndim:
            Dimensionality of a single per-frame item, e.g. 2 for a 3x3
            intrinsic matrix and 3 for an ``H x W x 1`` depth map.
        num_frames:
            Number of selected frames supplied to the model.

        Notes
        -----
        Some VGGT outputs arrive batched as ``(1, F, ...)`` while others are
        already per-frame as ``(F, ...)``. This helper uses ``num_frames`` to
        disambiguate those cases instead of assuming every ``ndim=item_ndim+1``
        tensor is already indexed by frame.
        """
        if value is None:
            return None
        arr = np.asarray(value)

        if arr.ndim == item_ndim:
            if num_frames != 1:
                raise ValueError(
                    f"Received a single-item output with shape {arr.shape} for {num_frames} frames."
                )
            return arr[None, ...]

        if arr.ndim == item_ndim + 1:
            if arr.shape[0] == num_frames:
                return arr
            raise ValueError(
                f"Ambiguous per-frame output shape {arr.shape}; expected leading dimension {num_frames}."
            )

        if arr.ndim == item_ndim + 2 and arr.shape[0] == 1:
            squeezed = arr[0]
            if squeezed.shape[0] == num_frames:
                return squeezed
            raise ValueError(
                f"Batched output shape {arr.shape} did not contain {num_frames} frames after squeezing."
            )

        raise ValueError(
            f"Could not normalize output with shape {arr.shape} to per-frame items of ndim {item_ndim}."
        )

    def _world_from_cam(self, extrinsic_cam_from_world: np.ndarray) -> np.ndarray:
        """Convert an OpenCV camera-from-world matrix to world-from-camera."""
        extrinsic = np.asarray(extrinsic_cam_from_world, dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic_h = np.eye(4, dtype=float)
            extrinsic_h[:3, :4] = extrinsic
            extrinsic = extrinsic_h
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Expected extrinsic shape (4, 4) or (3, 4), got {extrinsic.shape}.")
        return np.linalg.inv(extrinsic)

    def _initialize_from_predictions(
        self,
        frames: list[FrameRef],
        predictions: dict[str, Any],
        ingest_artifact,
        *,
        notes: list[str],
        image_paths: list[Path] | None = None,
    ) -> GeometryArtifact:
        """Convert raw VGGT predictions into a ``GeometryArtifact``."""
        numpy_predictions = self._numpyify(predictions)
        extrinsic = numpy_predictions.get("extrinsic")
        intrinsic = numpy_predictions.get("intrinsic")
        depth = numpy_predictions.get("depth")
        if depth is None:
            depth = numpy_predictions.get("depth_map")
        point_map = numpy_predictions.get("point_map")
        if point_map is None:
            point_map = numpy_predictions.get("world_points_from_depth")
        point_conf = numpy_predictions.get("point_conf")
        depth_conf = numpy_predictions.get("depth_conf")

        if extrinsic is None or intrinsic is None:
            raise RuntimeError(
                "VGGT predictions did not include extrinsic/intrinsic outputs in a parsable form."
            )

        if self.debug_shapes:
            self._debug_dump_selected_outputs(
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                depth=depth,
                point_map=point_map,
                depth_conf=depth_conf,
                point_conf=point_conf,
            )

        extrinsic = self._normalize_per_frame_array(extrinsic, item_ndim=2, num_frames=len(frames))
        intrinsic = self._normalize_per_frame_array(intrinsic, item_ndim=2, num_frames=len(frames))
        depth = self._normalize_per_frame_array(depth, item_ndim=3, num_frames=len(frames))
        point_map = self._normalize_per_frame_array(point_map, item_ndim=3, num_frames=len(frames))
        depth_conf = self._normalize_per_frame_array(depth_conf, item_ndim=2, num_frames=len(frames))
        point_conf = self._normalize_per_frame_array(point_conf, item_ndim=2, num_frames=len(frames))

        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}
        pairwise_matches: dict[str, Any] = {}

        for idx, frame in enumerate(frames):
            frame_intrinsic = np.asarray(intrinsic[idx], dtype=float)
            frame_extrinsic = np.asarray(extrinsic[idx], dtype=float)
            intrinsics = intrinsics_by_stream.setdefault(frame.stream_id, self._intrinsics_from_matrix(frame_intrinsic))
            camera_states_by_stream.setdefault(frame.stream_id, []).append(
                TimedCameraState(
                    stream_id=frame.stream_id,
                    time_sec=frame.timestamp_sec,
                    intrinsics=intrinsics,
                    pose=CameraPose(T_world_cam=self._world_from_cam(frame_extrinsic)),
                )
            )
            frame_key = self._frame_key(frame)
            if depth is not None:
                depthmaps_by_frame[frame_key] = {
                    "depth": np.asarray(depth[idx]),
                    "confidence": None if depth_conf is None else np.asarray(depth_conf[idx]),
                }
            if point_map is not None:
                pointmaps_by_frame[frame_key] = {
                    "points": np.asarray(point_map[idx]),
                    "confidence": None if point_conf is None else np.asarray(point_conf[idx]),
                }

        stream_ids = sorted({frame.stream_id for frame in frames})
        for idx, stream_a in enumerate(stream_ids):
            for stream_b in stream_ids[idx + 1 :]:
                pair_key = f"{stream_a}|{stream_b}"
                pairwise_matches[pair_key] = {
                    "status": "vggt_global_init",
                    "num_matches": None,
                    "confidence": None,
                }

        notes = list(notes)
        notes.append(
            "Initialized geometry with the real VGGT backend; distortion remains an optimizer-stage variable."
        )
        notes.append(f"Saved depth maps for {len(depthmaps_by_frame)} selected frames.")
        notes.append(f"Saved point maps for {len(pointmaps_by_frame)} selected frames.")
        return GeometryArtifact(
            run_id=(ingest_artifact.run_id if ingest_artifact is not None else "dev"),
            stage_name="geometry",
            save_dir=Path("."),
            backend_name="vggt",
            selected_frames=frames,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            pointmaps_by_frame=pointmaps_by_frame,
            depthmaps_by_frame=depthmaps_by_frame,
            pairwise_matches=pairwise_matches,
            view_graph_edges={},
            notes=notes,
            metadata={
                "debug_shapes": self.debug_shapes,
                "save_depth_pngs": False,
                "save_debug_assets": bool(self.debug_shapes),
                "save_camera_json": bool(self.debug_shapes),
                "save_pointcloud_plys": bool(self.debug_shapes),
                "save_debug_html": bool(self.debug_shapes),
                "debug_max_points": int(self.cfg.get('debug_max_points', 5000)),
                "debug_image_paths": {}
                if not image_paths else {
                    self._frame_key(frame): str(path)
                    for frame, path in zip(frames, image_paths)
                },
            },
        )

    def _initialize_heuristic(self, frames: list[FrameRef], ingest_artifact, sync_artifact) -> GeometryArtifact:
        """Fallback initializer used when the VGGT backend is unavailable."""
        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}

        stream_lookup = {}
        if ingest_artifact is not None:
            stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}

        for frame in frames:
            stream = stream_lookup.get(frame.stream_id)
            width = float(stream.width if stream is not None and stream.width > 0 else 1920)
            height = float(stream.height if stream is not None and stream.height > 0 else 1080)
            intrinsics = intrinsics_by_stream.setdefault(
                frame.stream_id,
                CameraIntrinsics(
                    fx=0.9 * width,
                    fy=0.9 * height,
                    cx=0.5 * width,
                    cy=0.5 * height,
                    distortion_model="brown",
                    distortion_params=np.zeros(4, dtype=float),
                ),
            )
            camera_states_by_stream.setdefault(frame.stream_id, []).append(
                TimedCameraState(
                    stream_id=frame.stream_id,
                    time_sec=frame.timestamp_sec,
                    intrinsics=intrinsics,
                    pose=CameraPose(T_world_cam=np.eye(4, dtype=float)),
                )
            )

        pairwise_matches: dict[str, Any] = {}
        stream_ids = sorted({frame.stream_id for frame in frames})
        for idx, stream_a in enumerate(stream_ids):
            for stream_b in stream_ids[idx + 1 :]:
                pair_key = f"{stream_a}|{stream_b}"
                pairwise_matches[pair_key] = {
                    "status": "heuristic_fallback",
                    "num_matches": 0,
                    "confidence": 0.0,
                }

        notes = [
            "Heuristic fallback path used because the real VGGT backend was unavailable.",
            "Install facebookresearch/vggt and set backend_mode='real' to force actual model inference.",
        ]
        if self._backend_error is not None:
            notes.append(f"Backend import/load error: {type(self._backend_error).__name__}: {self._backend_error}")
        if sync_artifact is not None and sync_artifact.offsets_by_stream:
            notes.append("Sync priors were available when geometry initialization ran.")

        return GeometryArtifact(
            run_id=(ingest_artifact.run_id if ingest_artifact is not None else "dev"),
            stage_name="geometry",
            save_dir=Path("."),
            backend_name="vggt",
            selected_frames=frames,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            pointmaps_by_frame={},
            depthmaps_by_frame={},
            pairwise_matches=pairwise_matches,
            view_graph_edges={},
            notes=notes,
            metadata={
                "debug_shapes": self.debug_shapes,
                "save_depth_pngs": False,
                "save_debug_assets": bool(self.debug_shapes),
                "save_camera_json": bool(self.debug_shapes),
                "save_pointcloud_plys": bool(self.debug_shapes),
                "save_debug_html": bool(self.debug_shapes),
                "debug_max_points": int(self.cfg.get('debug_max_points', 5000)),
                "debug_image_paths": {},
            },
        )

    def initialize(
        self,
        frames: list[FrameRef],
        *,
        ingest_artifact=None,
        sync_artifact=None,
    ) -> GeometryArtifact:
        """Produce coarse camera and geometry priors for the selected frames."""
        selected_frames = self._select_frames(frames, ingest_artifact)
        if not selected_frames:
            raise ValueError("VGGTInitializer.initialize() received no frames to process.")

        if not self._is_loaded and self._model is None and self.backend_mode != "heuristic":
            self.load_checkpoint()

        if self._model is None:
            return self._initialize_heuristic(selected_frames, ingest_artifact, sync_artifact)

        cache_root = Path(self.frame_cache_dir) if self.frame_cache_dir else Path(".room4d_cache") / "vggt_inputs"
        image_paths, notes = self._materialize_images(selected_frames, ingest_artifact, cache_root)
        images = self._load_and_preprocess_images([str(path) for path in image_paths]).to(self.device)

        dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_enabled = self.use_autocast and self.device.startswith("cuda") and torch.cuda.is_available()

        with torch.no_grad():
            if autocast_enabled:
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self._model(images)
            else:
                predictions = self._model(images)

        if self.debug_shapes:
            self._debug_dump_predictions(predictions)

        if isinstance(predictions, tuple):
            raise RuntimeError(
                "VGGT returned a tuple instead of the expected prediction dictionary; "
                "update the adapter if the upstream API changed."
            )

        if "pose_enc" in predictions and ("extrinsic" not in predictions or "intrinsic" not in predictions):
            extrinsic, intrinsic = self._pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions = dict(predictions)
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

        if (
            ("point_map" not in predictions or predictions.get("point_map") is None)
            and predictions.get("depth") is not None
            and predictions.get("extrinsic") is not None
            and predictions.get("intrinsic") is not None
            and self._unproject_depth_map_to_point_map is not None
        ):
            point_map = self._unproject_depth_map_to_point_map(
                predictions["depth"].squeeze(0),
                predictions["extrinsic"].squeeze(0),
                predictions["intrinsic"].squeeze(0),
            )
            predictions = dict(predictions)
            predictions["world_points_from_depth"] = point_map
            notes.append("Point maps were derived by unprojecting the VGGT depth branch.")

        return self._initialize_from_predictions(
            selected_frames,
            predictions,
            ingest_artifact,
            notes=notes,
            image_paths=image_paths,
        )

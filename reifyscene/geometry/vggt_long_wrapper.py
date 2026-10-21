"""Wrapper module for VGGT-Long coarse geometry initialization.

This adapter preserves the existing ``GeometryArtifact`` contract while
allowing the long-sequence backend to be selected next to the standard VGGT
initializer.
"""

from __future__ import annotations

import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reifyscene.common import data_dir
from reifyscene.common.memory import format_memory_snapshot, process_memory_snapshot, release_memory
from reifyscene.common.system_report import collect_system_report, recommend_vggt_long_profile
from reifyscene.common.types import FrameRef
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import vggt_input_transform
from reifyscene.geometry.vggt_derived import (
    forward_mixed_precision,
    load_and_preprocess_images_without_upscale,
)
from reifyscene.geometry.vggt_wrapper import VGGTInitializer, input_cache_root


@contextmanager
def _torch_cuda_visibility(device: str):
    """Temporarily shape torch.cuda visibility for backend construction/run.

    VGGT-Long probes CUDA during ``__init__`` and may do so again during ``run``.
    For CPU execution we need those probes to behave as if CUDA is unavailable,
    otherwise upstream can crash before our wrapper gets to override the runner
    device. For CUDA execution this context is a no-op.
    """
    normalized = str(device).lower()
    if normalized != "cpu":
        yield
        return

    original = {
        "is_available": torch.cuda.is_available,
        "device_count": torch.cuda.device_count,
        "get_device_capability": torch.cuda.get_device_capability,
        "current_device": getattr(torch.cuda, "current_device", None),
    }

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.get_device_capability = lambda *args, **kwargs: (0, 0)
    if original["current_device"] is not None:
        torch.cuda.current_device = lambda: 0

    try:
        yield
    finally:
        torch.cuda.is_available = original["is_available"]
        torch.cuda.device_count = original["device_count"]
        torch.cuda.get_device_capability = original["get_device_capability"]
        if original["current_device"] is not None:
            torch.cuda.current_device = original["current_device"]


class VGGTLongInitializer(VGGTInitializer):
    """Adapter for the VGGT-Long offline long-sequence geometry backend.

    The backend is expected to expose a ``VGGT_Long`` class taking
    ``(image_dir, save_dir, config)`` and to emit chunk-level ``.npy`` files
    containing arrays with the same semantic fields as VGGT predictions.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.backend_name = "vggt-long"
        # VGGT-Long's own chunking (configs/base_config.yaml). 16/4 gave each clip several
        # chunk boundaries, each aligned with some error: on ten DAVIS clips, the share of
        # static depth agreeing within 5% 15 frames apart stayed within 7 points of one
        # VGGT pass at 60/30 and fell up to 37.5 points below it at 16/4 (2026-10-21).
        self.chunk_size = int(self.cfg.get("chunk_size", 60))
        self.overlap = int(self.cfg.get("overlap", 30))
        self.loop_enable = bool(self.cfg.get("loop_enable", False))
        self.use_dbow = bool(self.cfg.get("use_dbow", False))
        self.delete_temp_files = bool(self.cfg.get("delete_temp_files", False))
        self.output_cache_dir = self.cfg.get("output_cache_dir")
        self.long_config = dict(self.cfg.get("long_config", {}))
        self.auto_tune = bool(self.cfg.get("auto_tune", False))
        self.auto_tune_max_attempts = max(int(self.cfg.get("auto_tune_max_attempts", 12)), 1)
        self.min_chunk_size = max(int(self.cfg.get("min_chunk_size", 2)), 1)
        min_resize_max_dim = self.cfg.get("min_resize_max_dim")
        self.min_resize_max_dim = None if min_resize_max_dim in (None, 0) else int(min_resize_max_dim)
        self.system_report = self.cfg.get("system_report")
        self.enable_true_mixed_precision = bool(
            self.cfg.get("enable_true_mixed_precision", str(self.device).lower().startswith("cuda"))
        )
        self.prevent_input_upscale = bool(self.cfg.get("prevent_input_upscale", True))
        self._VGGT_Long: Any | None = None

    def _log_auto_tune(self, message: str) -> None:
        print(f"[VGGT-Long auto-tune] {message}", flush=True)

    @staticmethod
    def _format_resize_max_dim(resize_max_dim: int | None) -> str:
        return "original" if resize_max_dim is None else str(int(resize_max_dim))

    @staticmethod
    def _effective_vggt_input_target_size(resize_max_dim: int | None, source_max_dim: int | None = None) -> int:
        candidate = 518 if resize_max_dim in (None, 0) else int(resize_max_dim)
        if source_max_dim not in (None, 0):
            candidate = min(candidate, int(source_max_dim))
        candidate = min(518, max(14, candidate))
        return max(14, (candidate // 14) * 14)

    def _format_attempt_summary(self, attempt: dict[str, Any]) -> str:
        summary = (
            f"attempt {attempt.get('attempt_index', '?')}: "
            f"chunk_size={attempt.get('chunk_size')}, "
            f"overlap={attempt.get('overlap')}, "
            f"resize_max_dim={self._format_resize_max_dim(attempt.get('resize_max_dim'))}, "
            f"effective_vggt_input_max_dim={attempt.get('effective_vggt_input_max_dim')}, "
            f"status={attempt.get('status', 'unknown')}"
        )
        error = attempt.get("error")
        if error:
            summary += f", error={error}"
        chunk_files = attempt.get("num_chunk_files")
        if chunk_files is not None:
            summary += f", chunk_files={chunk_files}"
        return summary

    @staticmethod
    def _clear_exception_traceback(exc: Exception | None) -> None:
        if exc is None:
            return
        tb = exc.__traceback__
        if tb is not None:
            try:
                traceback.clear_frames(tb)
            except Exception:
                pass
        try:
            exc.__traceback__ = None
        except Exception:
            pass
        try:
            exc.__cause__ = None
        except Exception:
            pass
        try:
            exc.__context__ = None
        except Exception:
            pass

    @staticmethod
    def _release_runner_references(runner: Any | None) -> None:
        if runner is None:
            return
        try:
            if hasattr(runner, "img_list"):
                runner.img_list = []
        except Exception:
            pass
        model_wrapper = getattr(runner, "model", None)
        if model_wrapper is not None:
            inner_model = getattr(model_wrapper, "model", None)
            if inner_model is not None and hasattr(inner_model, "to"):
                try:
                    inner_model.to("cpu")
                except Exception:
                    pass
            try:
                if hasattr(model_wrapper, "model"):
                    model_wrapper.model = None
            except Exception:
                pass
        try:
            if hasattr(runner, "model"):
                runner.model = None
        except Exception:
            pass

    def _cleanup_attempt_resources(self, runner: Any | None, exc: Exception | None, *, context: str) -> None:
        before_snapshot = process_memory_snapshot()
        self._log_auto_tune(
            f"{context}: cleanup starting; before cleanup {format_memory_snapshot(before_snapshot)}."
        )
        self._release_runner_references(runner)
        self._clear_exception_traceback(exc)
        release_memory()
        after_snapshot = process_memory_snapshot()
        self._log_auto_tune(
            f"{context}: cleanup finished; after cleanup {format_memory_snapshot(after_snapshot)}."
        )


    def _load_backend_components(self) -> None:  # type: ignore[override]
        """Import the VGGT-Long module lazily.

        We first try a normal import. If that fails, we fall back to importing
        directly from a local third_party/VGGT-Long checkout.
        """
        import importlib
        import sys
        from pathlib import Path

        module = None
        errors: list[str] = []

        # 1) Normal environment import
        try:
            module = importlib.import_module("vggt_long")
        except Exception as exc:
            errors.append(f"normal import failed: {exc!r}")

        # 2) Fallback: local checkout under third_party/VGGT-Long
        if module is None:
            repo_root = Path(__file__).resolve().parents[2]
            local_checkout = repo_root / "third_party" / "VGGT-Long"
            if local_checkout.exists():
                sys.path.insert(0, str(local_checkout))
                try:
                    module = importlib.import_module("vggt_long")
                except Exception as exc:
                    errors.append(f"local checkout import failed: {exc!r}")

        if module is None:
            raise ImportError(
                "Could not import VGGT-Long module 'vggt_long'. "
                + " | ".join(errors)
            )

        self._VGGT_Long = getattr(module, "VGGT_Long")

    def load_checkpoint(self) -> None:  # type: ignore[override]
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
            self._is_loaded = True
        except Exception as exc:  # pragma: no cover - exercised in integration usage
            self._backend_error = exc
            self._is_loaded = False
            if self.backend_mode == "real":
                raise RuntimeError(
                    "Failed to initialize the real VGGT-Long backend. Install the official "
                    "VGGT-Long package and required dependencies, or switch "
                    "backend_mode to 'auto' or 'heuristic'."
                ) from exc


    def release_resources(self) -> None:
        self._VGGT_Long = None
        self._backend_error = None

    def _default_long_config(self) -> dict[str, Any]:
        """Build a VGGT-Long config compatible with upstream expectations.

        This mirrors the structure of upstream `configs/base_config.yaml`
        closely enough for the Python pipeline path, while still allowing
        reifyscene-specific overrides.
        """
        # Optional explicit weights path. If the user doesn't provide one, leave the
        # upstream default-style relative path so local VGGT-Long setups that have
        # downloaded weights into their own repo can still work.
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        weights_root = repo_root / "third_party" / "VGGT-Long" / "weights"

        vggt_weight = self.cfg.get("vggt_weight", str(weights_root / "model.pt"))
        dbow_weight = self.cfg.get("dbow_weight", str(weights_root / "ORBvoc.txt"))
        salad_weight = self.cfg.get("salad_weight", str(weights_root / "dino_salad.ckpt"))
        dnio_weight = self.cfg.get("dnio_weight", str(weights_root / "dinov2_vitb14_pretrain.pth"))
        map_weight = self.cfg.get("map_weight", str(weights_root / "map_anything"))
        pi3_weight = self.cfg.get("pi3_weight", str(weights_root / "Pi3" / "model.safetensors"))

        cfg: dict[str, Any] = {
            "Weights": {
                "model": "VGGT",
                "VGGT": vggt_weight,
                "Pi3": pi3_weight,
                "Map": map_weight,
                "DBoW": dbow_weight,
                "SALAD": salad_weight,
                "DNIO": dnio_weight,
            },
            "Model": {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "loop_chunk_size": max(int(self.cfg.get("loop_chunk_size", max(self.chunk_size, 8))), 1),
                "loop_enable": self.loop_enable,
                "useDBoW": self.use_dbow,
                "delete_temp_files": self.delete_temp_files,
                "align_method": self.cfg.get("align_method", "numba"),
                "using_sim3": bool(self.cfg.get("using_sim3", True)),
                "reference_frame_mid": bool(self.cfg.get("reference_frame_mid", False)),
                "calib": bool(self.cfg.get("calib", False)),
                "h": int(self.cfg.get("input_h", 376)),
                "w": int(self.cfg.get("input_w", 1241)),
                "Pointcloud_Save": {
                    "sample_ratio": float(self.cfg.get("pc_sample_ratio", 0.015)),
                    "conf_threshold_coef": float(self.cfg.get("pc_conf_threshold_coef", 0.75)),
                    # Keep BOTH spellings because upstream code appears inconsistent
                    # across versions/paths.
                    "use_conf_filter": bool(self.cfg.get("pc_use_conf_filter", True)),
                    "use_confidence_filtering": bool(self.cfg.get("pc_use_conf_filter", True)),
                },
                "IRLS": {
                    "delta": float(self.cfg.get("irls_delta", 0.1)),
                    "max_iters": int(self.cfg.get("irls_max_iters", 5)),
                    # upstream sim3utils.py does eval(config['Model']['IRLS']['tol'])
                    # so this must be a string, not a float
                    "tol": str(self.cfg.get("irls_tol", "1e-9")),
                },
            },
            "Loop": {
                "DBoW": {
                    "thresh": float(self.cfg.get("dbow_thresh", 0.034)),
                    "num_repeat": int(self.cfg.get("dbow_num_repeat", 3)),
                    "NMS": int(self.cfg.get("dbow_nms", 25)),
                },
                "SALAD": {
                    "image_size": list(self.cfg.get("salad_image_size", [336, 336])),
                    "batch_size": int(self.cfg.get("salad_batch_size", 32)),
                    "similarity_threshold": float(self.cfg.get("salad_similarity_threshold", 0.85)),
                    "top_k": int(self.cfg.get("salad_top_k", 5)),
                    "use_nms": bool(self.cfg.get("salad_use_nms", True)),
                    "nms_threshold": int(self.cfg.get("salad_nms_threshold", 25)),
                },
                "SIM3_Optimizer": {
                    # Default to python so the pure-Python path works even when the C++
                    # extension has not been built.
                    "lang_version": self.cfg.get("sim3_lang_version", "python"),
                    "max_iterations": int(self.cfg.get("sim3_max_iterations", 30)),
                    "lambda_init": str(self.cfg.get("sim3_lambda_init", "1e-6")),
                },
            },
        }

        # Deep-ish merge with user overrides.
        for key, value in self.long_config.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and isinstance(cfg[key].get(subkey), dict):
                        cfg[key][subkey].update(subvalue)
                    else:
                        cfg[key][subkey] = subvalue
            else:
                cfg[key] = value

        return cfg

    def _select_chunk_prediction_dir(self, runner: Any) -> Path:
        candidates = []
        for attr in ("result_aligned_dir", "result_unaligned_dir"):
            value = getattr(runner, attr, None)
            if value:
                path = Path(value)
                if path.exists() and list(path.glob("chunk_*.npy")):
                    candidates.append(path)
        if candidates:
            # Prefer aligned chunks when available.
            for path in candidates:
                if "aligned" in path.name:
                    return path
            return candidates[0]
        raise RuntimeError("VGGT-Long did not produce any chunk_*.npy prediction files.")

    def _load_chunk_predictions(self, runner: Any) -> list[dict[str, Any]]:
        chunk_dir = self._select_chunk_prediction_dir(runner)
        chunks: list[dict[str, Any]] = []
        chunk_indices = list(getattr(runner, "chunk_indices", []))
        for chunk_idx, chunk_range in enumerate(chunk_indices):
            path = chunk_dir / f"chunk_{chunk_idx}.npy"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.shape == ():
                payload = data.item()
            elif isinstance(data, dict):
                payload = data
            else:
                # np.save(dict) returns 0-d object arrays on load, but keep a
                # defensive fallback for odd formats.
                payload = dict(data)
            payload = self._numpyify(payload)
            payload["_chunk_range"] = tuple(int(v) for v in chunk_range)
            payload["_chunk_index"] = int(chunk_idx)
            chunks.append(payload)
        if not chunks:
            raise RuntimeError("VGGT-Long produced no readable chunk predictions.")
        return chunks

    @staticmethod
    def _chunk_scales(runner: Any, num_chunks: int) -> list[float]:
        """Each chunk's scale in chunk 0's frame, by chunk index.

        After its run, VGGT-Long's ``sim3_list`` holds the accumulated Sim(3) ``(s, R, t)``
        taking chunk ``k + 1`` into chunk 0's frame -- the transform ``camera_poses.txt``
        and the aligned ``world_points`` went through. Chunk 0 is the reference (scale 1).
        """
        sim3 = list(getattr(runner, "sim3_list", None) or [])
        if len(sim3) != num_chunks - 1:
            raise RuntimeError(
                f"VGGT-Long recorded {len(sim3)} chunk alignment(s) for {num_chunks} chunk(s); "
                "its depth cannot be put on one scale."
            )
        return [1.0] + [float(np.asarray(s).reshape(-1)[0]) for s, _rot, _t in sim3]

    def _load_global_camera_to_world(self, runner: Any, num_frames: int) -> np.ndarray:
        """Read VGGT-Long's globally-aligned camera poses from ``camera_poses.txt``.

        VGGT-Long estimates per-chunk Sim(3) transforms (overlap registration +
        optional loop closure), accumulates them, and applies the result only
        when writing ``camera_poses.txt`` — the per-chunk ``_tmp_results_aligned``
        ``.npy`` files keep their **chunk-local** extrinsics (only ``world_points``
        are globalized there). Merging those local extrinsics yields 23
        independently-anchored mini-trajectories. So we read the authoritative
        global trajectory VGGT-Long published instead.

        Returns an ``(N, 4, 4)`` stack of camera-to-world (world-from-camera)
        matrices in global-frame order. A missing file, or one whose frame count
        does not match the selected frames, raises: chunk-local poses are not one
        scene.
        """
        candidates = []
        output_dir = getattr(runner, "output_dir", None)
        if output_dir:
            candidates.append(Path(output_dir) / "camera_poses.txt")
        aligned_dir = getattr(runner, "result_aligned_dir", None)
        if aligned_dir:
            candidates.append(Path(aligned_dir).parent / "camera_poses.txt")
        for path in candidates:
            if not path.exists():
                continue
            rows = [
                [float(x) for x in line.split()]
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if len(rows) != num_frames or any(len(r) != 16 for r in rows):
                raise RuntimeError(
                    f"camera_poses.txt at {path} has {len(rows)} pose(s) for {num_frames} frame(s)."
                )
            return np.asarray(rows, dtype=float).reshape(-1, 4, 4)
        raise RuntimeError("VGGT-Long wrote no camera_poses.txt; it has no globally aligned poses.")

    def _combined_predictions_from_chunks(
        self,
        frames: list[FrameRef],
        chunks: list[dict[str, Any]],
        global_c2w: np.ndarray,
        chunk_scales: list[float],
    ) -> dict[str, Any]:
        """One prediction per frame, in VGGT-Long's global frame.

        Cameras come from ``global_c2w`` (``camera_poses.txt``). A frame in an overlap
        takes the later chunk's predictions, as ``camera_poses.txt`` takes that chunk's
        pose. Each chunk's depth is multiplied by its Sim(3) scale (``chunk_scales``, by
        chunk index): depth is camera-frame, so the rotation and translation leave it
        alone, but the scale does not -- left out, every chunk's depth keeps its own
        scale against cameras on chunk 0's, and frames from different chunks disagree.
        """
        num_frames = len(frames)
        extrinsic: list[Any] = [None] * num_frames
        intrinsic: list[Any] = [None] * num_frames
        depth: list[Any] = [None] * num_frames
        depth_conf: list[Any] = [None] * num_frames
        point_map: list[Any] = [None] * num_frames
        point_conf: list[Any] = [None] * num_frames

        for payload in chunks:
            chunk_start, chunk_end = payload["_chunk_range"]
            chunk_len = chunk_end - chunk_start
            chunk_extr = np.asarray(payload.get("extrinsic"))
            chunk_intr = np.asarray(payload.get("intrinsic"))
            chunk_depth = payload.get("depth")
            chunk_depth_conf = payload.get("depth_conf")
            chunk_points = payload.get("world_points")
            if chunk_points is None:
                chunk_points = payload.get("point_map")
            chunk_point_conf = payload.get("world_points_conf")
            if chunk_point_conf is None:
                chunk_point_conf = payload.get("point_conf")

            if chunk_depth is not None:
                chunk_depth = np.asarray(chunk_depth) * chunk_scales[payload["_chunk_index"]]
                if chunk_depth.ndim == 3:
                    chunk_depth = chunk_depth[..., None]
            if chunk_depth_conf is not None:
                chunk_depth_conf = np.asarray(chunk_depth_conf)
            if chunk_points is not None:
                chunk_points = np.asarray(chunk_points)
            if chunk_point_conf is not None:
                chunk_point_conf = np.asarray(chunk_point_conf)

            for local_idx in range(min(chunk_len, chunk_extr.shape[0])):
                global_idx = chunk_start + local_idx
                if global_idx >= num_frames:
                    break
                # A later chunk overwrites an overlap frame, as camera_poses.txt does.
                extrinsic[global_idx] = chunk_extr[local_idx]
                intrinsic[global_idx] = chunk_intr[local_idx]
                if chunk_depth is not None and local_idx < chunk_depth.shape[0]:
                    depth[global_idx] = chunk_depth[local_idx]
                if chunk_depth_conf is not None and local_idx < chunk_depth_conf.shape[0]:
                    depth_conf[global_idx] = chunk_depth_conf[local_idx]
                if chunk_points is not None and local_idx < chunk_points.shape[0]:
                    point_map[global_idx] = chunk_points[local_idx]
                if chunk_point_conf is not None and local_idx < chunk_point_conf.shape[0]:
                    point_conf[global_idx] = chunk_point_conf[local_idx]

        missing = [idx for idx, item in enumerate(extrinsic) if item is None or intrinsic[idx] is None]
        if missing:
            raise RuntimeError(f"VGGT-Long did not produce geometry for frame indices: {missing}")

        # Allow depth/points to be missing for some frames; downstream already handles absent maps.
        preds: dict[str, Any] = {
            "extrinsic": np.stack([np.asarray(v) for v in extrinsic], axis=0),
            "intrinsic": np.stack([np.asarray(v) for v in intrinsic], axis=0),
        }
        if all(v is not None for v in depth):
            preds["depth"] = np.stack([np.asarray(v) for v in depth], axis=0)
        if all(v is not None for v in depth_conf):
            preds["depth_conf"] = np.stack([np.asarray(v) for v in depth_conf], axis=0)
        if all(v is not None for v in point_map):
            preds["point_map"] = np.stack([np.asarray(v) for v in point_map], axis=0)
        if all(v is not None for v in point_conf):
            preds["point_conf"] = np.stack([np.asarray(v) for v in point_conf], axis=0)

        # VGGT-Long's per-chunk extrinsics are chunk-local; replace them with the
        # globally-aligned poses from camera_poses.txt. Those are camera-to-world, so
        # invert to camera-from-world (the convention the shared
        # `_initialize_from_predictions` / `_world_from_cam` path expects, matching plain
        # VGGT) -- it inverts back to recover a correct global `T_world_cam`. world_points
        # are already globalized; depth was scaled above.
        if global_c2w.shape[0] != len(frames):
            raise RuntimeError(
                f"global camera poses ({global_c2w.shape[0]}) do not match frame count ({len(frames)})."
            )
        preds["extrinsic"] = np.stack([np.linalg.inv(np.asarray(p, dtype=float)) for p in global_c2w], axis=0)
        return preds

    def _requested_device(self) -> str:
        return str(self.device).lower()

    def _configure_runner_device(self, runner: Any) -> None:
        requested_device = self._requested_device()
        if requested_device == "cpu":
            runner.device = "cpu"
            runner.dtype = torch.float32
            if getattr(runner, "model", None) is not None:
                runner.model.device = "cpu"
                if hasattr(runner.model, "dtype"):
                    runner.model.dtype = torch.float32
        else:
            runner.device = requested_device
            if getattr(runner, "model", None) is not None:
                runner.model.device = requested_device

    @contextmanager
    def _patched_torch_load_for_device(self):
        """Temporarily patch torch.load so checkpoints always deserialize on CPU first.

        Upstream VGGT-Long hardcodes `map_location='cuda'` when loading VGGT
        weights. Rewriting those calls to CPU reduces peak GPU memory during
        model initialization; upstream then moves the fully loaded model to the
        requested device via ``model.to(self.device)``.
        """
        original_torch_load = torch.load

        def _cpu_safe_torch_load(*args, **kwargs):
            map_location = kwargs.get("map_location", None)
            if map_location is None and len(args) >= 2:
                map_location = args[1]

            if map_location == "cuda":
                kwargs["map_location"] = torch.device("cpu")
            elif isinstance(map_location, str) and map_location.startswith("cuda"):
                kwargs["map_location"] = torch.device("cpu")
            elif isinstance(map_location, torch.device) and map_location.type == "cuda":
                kwargs["map_location"] = torch.device("cpu")

            return original_torch_load(*args, **kwargs)

        torch.load = _cpu_safe_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load

    @contextmanager
    def _patched_vggt_long_preprocess(self):
        if not self.prevent_input_upscale:
            yield
            return

        target_size = self._effective_vggt_input_target_size(self.resize_max_dim)
        if target_size >= 518:
            yield
            return

        try:
            from base_models.vggt.utils import load_fn as load_fn_module
        except Exception:
            yield
            return

        original_loader = load_fn_module.load_and_preprocess_images

        def _bounded_loader(image_path_list, mode="crop"):
            return load_and_preprocess_images_without_upscale(
                [str(path) for path in image_path_list],
                mode=mode,
                target_size=target_size,
            )

        load_fn_module.load_and_preprocess_images = _bounded_loader
        try:
            yield
        finally:
            load_fn_module.load_and_preprocess_images = original_loader

    @contextmanager
    def _patched_vggt_true_mixed_precision(self):
        if not self.enable_true_mixed_precision or not self._requested_device().startswith("cuda"):
            yield
            return

        try:
            from base_models.vggt.models.vggt import VGGT
        except Exception:
            yield
            return

        original_forward = VGGT.forward

        VGGT.forward = forward_mixed_precision
        try:
            yield
        finally:
            VGGT.forward = original_forward

    def _create_runner(self, image_dir: Path, output_dir: Path) -> Any:
        with _torch_cuda_visibility(self._requested_device()):
            runner = self._VGGT_Long(str(image_dir), str(output_dir), self._default_long_config())
        self._configure_runner_device(runner)
        return runner

    def _run_runner(self, runner: Any) -> None:
        with _torch_cuda_visibility(self._requested_device()):
            with ExitStack() as stack:
                stack.enter_context(self._patched_torch_load_for_device())
                stack.enter_context(self._patched_vggt_long_preprocess())
                stack.enter_context(self._patched_vggt_true_mixed_precision())
                runner.run()

    def _frame_source_max_dim(self, frames: list[FrameRef], ingest_artifact) -> int | None:
        if ingest_artifact is None:
            return None
        requested_streams = {str(frame.stream_id) for frame in frames}
        max_dim = 0
        for stream in ingest_artifact.streams:
            if requested_streams and stream.stream_id not in requested_streams:
                continue
            max_dim = max(max_dim, int(stream.width), int(stream.height))
        return max_dim or None

    def _current_system_report(self) -> dict[str, Any]:
        report = self.system_report or self.cfg.get("system_report")
        if isinstance(report, dict) and report:
            return report
        return collect_system_report()

    @staticmethod
    def _chunk_candidates(initial_chunk_size: int, min_chunk_size: int, num_frames: int) -> list[int]:
        if num_frames <= 0:
            return []
        floor = max(1, min(min_chunk_size, num_frames))
        initial = max(floor, min(int(initial_chunk_size), num_frames))
        candidates: list[int] = []
        for value in (
            initial,
            max(floor, int(round(initial * 0.75))),
            max(floor, int(round(initial * 0.5))),
            floor,
        ):
            value = max(floor, min(int(value), num_frames))
            if value not in candidates:
                candidates.append(value)
        return candidates

    @staticmethod
    def _resize_candidates(
        initial_resize_max_dim: int | None,
        min_resize_max_dim: int | None,
        source_max_dim: int | None,
        recommendation_candidates: list[int | None] | None = None,
    ) -> list[int | None]:
        if source_max_dim is not None and source_max_dim > 0 and initial_resize_max_dim is None:
            initial_resize_max_dim = source_max_dim
        if initial_resize_max_dim is None and not recommendation_candidates:
            return [None]

        floor = int(min_resize_max_dim) if min_resize_max_dim not in (None, 0) else None
        if floor is None and initial_resize_max_dim not in (None, 0):
            floor = max(256, int(round(int(initial_resize_max_dim) * 0.5)))
        if source_max_dim is not None and source_max_dim > 0 and floor is not None:
            floor = min(floor, int(source_max_dim))

        values: list[int | None] = []

        def _append(value: int | None) -> None:
            if value in (None, 0):
                return
            value = int(value)
            if source_max_dim is not None and source_max_dim > 0:
                value = min(value, int(source_max_dim))
            if floor is not None:
                value = max(floor, value)
            if value not in values:
                values.append(value)

        initial_value = None if initial_resize_max_dim in (None, 0) else int(initial_resize_max_dim)
        _append(initial_value)

        smaller_or_equal: list[int | None] = []
        larger: list[int | None] = []
        for value in recommendation_candidates or []:
            if value in (None, 0):
                continue
            value = int(value)
            if initial_value is not None and value > initial_value:
                larger.append(value)
            else:
                smaller_or_equal.append(value)
        for value in smaller_or_equal:
            _append(value)
        if floor is not None:
            _append(floor)
        for value in larger:
            _append(value)
        if not values and initial_resize_max_dim not in (None, 0):
            initial = int(initial_resize_max_dim)
            for value in (
                initial,
                int(round(initial * 0.9)),
                int(round(initial * 0.8)),
                int(round(initial * 0.7)),
                int(round(initial * 0.6)),
                int(round(initial * 0.5)),
                floor,
            ):
                _append(value)
        return values or [initial_resize_max_dim]

    def _derive_overlap(self, chunk_size: int, num_frames: int, overlap_ratio: float, overlap_hint: int | None = None) -> int:
        if num_frames <= chunk_size:
            return 0
        if overlap_hint is not None and overlap_hint > 0:
            overlap = int(round(float(overlap_hint) * (float(chunk_size) / float(max(self.chunk_size, 1)))))
        else:
            overlap = int(round(float(chunk_size) * overlap_ratio))
        return max(1, min(chunk_size - 1, overlap))

    @staticmethod
    def _is_probable_oom_error(exc: Exception) -> bool:
        oom_types = []
        torch_oom = getattr(torch, "OutOfMemoryError", None)
        if isinstance(torch_oom, type):
            oom_types.append(torch_oom)
        cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if isinstance(cuda_oom, type) and cuda_oom not in oom_types:
            oom_types.append(cuda_oom)
        if oom_types and isinstance(exc, tuple(oom_types)):
            return True
        msg = str(exc).lower()
        markers = [
            "out of memory",
            "cuda oom",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
            "failed to allocate memory",
        ]
        return any(marker in msg for marker in markers)

    def _attempt_plan_sequence(self, frames: list[FrameRef], ingest_artifact) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        report = self._current_system_report()
        source_max_dim = self._frame_source_max_dim(frames, ingest_artifact)
        recommendation = recommend_vggt_long_profile(
            report,
            device=self.device,
            num_frames=len(frames),
            source_max_dim=source_max_dim,
        )

        configured_chunk = int(self.cfg.get("chunk_size", recommendation["initial_chunk_size"]))
        configured_resize = self.resize_max_dim if self.resize_max_dim is not None else recommendation.get("initial_resize_max_dim")
        overlap_hint = self.cfg.get("overlap")
        overlap_ratio = 0.25
        if configured_chunk > 0 and overlap_hint not in (None, 0):
            overlap_ratio = max(0.1, min(0.45, float(overlap_hint) / float(configured_chunk)))

        min_chunk_size = max(1, int(self.cfg.get("min_chunk_size", recommendation.get("min_chunk_size", self.min_chunk_size))))
        min_resize_max_dim = self.min_resize_max_dim
        if min_resize_max_dim is None:
            min_resize_max_dim = recommendation.get("min_resize_max_dim")

        resize_candidates = self._resize_candidates(
            None if configured_resize in (None, 0) else int(configured_resize),
            None if min_resize_max_dim in (None, 0) else int(min_resize_max_dim),
            source_max_dim,
            recommendation_candidates=recommendation.get("resize_candidates"),
        )
        chunk_candidates = self._chunk_candidates(configured_chunk, min_chunk_size, len(frames))

        plans: list[dict[str, Any]] = []
        seen: set[tuple[int, int | None]] = set()
        for resize_max_dim in resize_candidates:
            for chunk_size in chunk_candidates:
                overlap = self._derive_overlap(chunk_size, len(frames), overlap_ratio, overlap_hint=overlap_hint)
                key = (int(chunk_size), None if resize_max_dim is None else int(resize_max_dim))
                if key in seen:
                    continue
                seen.add(key)
                plans.append({
                    "chunk_size": int(chunk_size),
                    "overlap": int(overlap),
                    "resize_max_dim": None if resize_max_dim is None else int(resize_max_dim),
                })
                if len(plans) >= self.auto_tune_max_attempts:
                    return plans, recommendation
        return plans, recommendation

    def _run_auto_tuned(
        self,
        selected_frames: list[FrameRef],
        *,
        ingest_artifact,
        cache_root: Path,
        backend_output_root: Path,
    ) -> tuple[dict[str, Any], list[Path], list[str], list[dict[str, Any]], dict[str, Any]]:
        plans, recommendation = self._attempt_plan_sequence(selected_frames, ingest_artifact)
        attempt_history: list[dict[str, Any]] = []
        image_cache: dict[int | None, tuple[list[Path], list[str]]] = {}
        input_caches: dict[int | None, dict[str, Any] | None] = {}
        last_error_summary: str | None = None

        self._log_auto_tune(
            f"starting auto-tune for {len(selected_frames)} frame(s) with {len(plans)} candidate setting(s); "
            f"recommended start chunk_size={recommendation.get('initial_chunk_size')}, "
            f"overlap={recommendation.get('initial_overlap')}, "
            f"resize_max_dim={self._format_resize_max_dim(recommendation.get('initial_resize_max_dim'))}."
        )

        for attempt_idx, plan in enumerate(plans, start=1):
            resize_max_dim = plan["resize_max_dim"]
            resize_label = self._format_resize_max_dim(resize_max_dim)
            effective_input_target = self._effective_vggt_input_target_size(resize_max_dim)
            self._log_auto_tune(
                f"attempt {attempt_idx}/{len(plans)}: trying chunk_size={plan['chunk_size']}, "
                f"overlap={plan['overlap']}, resize_max_dim={resize_label}, "
                f"effective_vggt_input_max_dim={effective_input_target}."
            )
            self.resize_max_dim = None if resize_max_dim is None else int(resize_max_dim)
            if resize_max_dim not in image_cache:
                # Each resize setting materializes into its own content-keyed directory.
                image_cache[resize_max_dim] = self._materialize_images(
                    selected_frames,
                    ingest_artifact,
                    cache_root,
                    resize_max_dim=resize_max_dim,
                )
                input_caches[resize_max_dim] = self._last_input_cache
                self._log_auto_tune(
                    f"attempt {attempt_idx}/{len(plans)}: prepared {len(image_cache[resize_max_dim][0])} image(s) at resize_max_dim={resize_label}."
                )
            else:
                self._log_auto_tune(
                    f"attempt {attempt_idx}/{len(plans)}: reusing cached images for resize_max_dim={resize_label}."
                )
            image_paths, resize_notes = image_cache[resize_max_dim]

            self.chunk_size = int(plan["chunk_size"])
            self.overlap = int(plan["overlap"])
            attempt_output_dir = backend_output_root / (
                f"attempt_{attempt_idx:02d}_chunk_{self.chunk_size:03d}_overlap_{self.overlap:02d}_"
                f"resize_{'orig' if resize_max_dim is None else int(resize_max_dim)}"
            )
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            runner = None
            try:
                if len(selected_frames) > self.chunk_size and self.overlap <= 0:
                    raise ValueError(
                        "VGGT-Long requires overlap > 0 when more than one chunk is needed. "
                        f"Got {len(selected_frames)} frames, chunk_size={self.chunk_size}, overlap={self.overlap}."
                    )
                runner = self._create_runner(Path(image_paths[0]).parent if image_paths else cache_root, attempt_output_dir)
                runner.img_list = [str(path) for path in image_paths]
                self._run_runner(runner)
                chunks = self._load_chunk_predictions(runner)
                global_c2w = self._load_global_camera_to_world(runner, len(selected_frames))
                scales = self._chunk_scales(runner, len(runner.chunk_indices))
                predictions = self._combined_predictions_from_chunks(
                    selected_frames, chunks, global_c2w=global_c2w, chunk_scales=scales
                )
                self._last_input_cache = input_caches.get(resize_max_dim)
                attempt_history.append({
                    **plan,
                    "attempt_index": attempt_idx,
                    "effective_vggt_input_max_dim": effective_input_target,
                    "status": "success",
                    "num_chunk_files": len(chunks),
                })
                self._log_auto_tune(
                    f"attempt {attempt_idx}/{len(plans)} succeeded with {len(chunks)} chunk file(s); "
                    f"selected chunk_size={self.chunk_size}, overlap={self.overlap}, resize_max_dim={resize_label}."
                )
                notes = list(resize_notes)
                notes.append(
                    f"VGGT-Long auto-tune selected chunk_size={self.chunk_size}, overlap={self.overlap}, "
                    f"resize_max_dim={'original' if resize_max_dim is None else resize_max_dim}, "
                    f"effective_vggt_input_max_dim={effective_input_target}."
                )
                notes.append(
                    f"VGGT-Long processed {len(selected_frames)} selected frames across {len(chunks)} chunk file(s)."
                )
                notes.append(
                    "Used VGGT-Long globally-aligned camera_poses.txt for camera poses; each chunk's "
                    "depth scaled by its alignment scale ("
                    + ", ".join(f"{v:.3f}" for v in scales) + ")."
                )
                return predictions, image_paths, notes, attempt_history, recommendation
            except Exception as exc:
                is_oom = self._is_probable_oom_error(exc)
                error_summary = f"{type(exc).__name__}: {exc}"
                attempt_record = {
                    **plan,
                    "attempt_index": attempt_idx,
                    "effective_vggt_input_max_dim": effective_input_target,
                    "status": "oom" if is_oom else "error",
                    "error": error_summary,
                }
                attempt_history.append(attempt_record)
                self._log_auto_tune(
                    f"attempt {attempt_idx}/{len(plans)} failed with {'OOM' if is_oom else 'error'}: "
                    f"{error_summary}"
                )
                last_error_summary = error_summary
                self._cleanup_attempt_resources(
                    runner,
                    exc,
                    context=f"attempt {attempt_idx}/{len(plans)}",
                )
                runner = None
                if not is_oom:
                    raise

        if last_error_summary is None:
            raise RuntimeError("VGGT-Long auto-tune did not produce any attempt plans.")
        summary_lines = [self._format_attempt_summary(item) for item in attempt_history]
        self._log_auto_tune(
            "exhausted all candidate settings. Summary: " + " | ".join(summary_lines)
        )
        raise RuntimeError(
            "VGGT-Long auto-tune exhausted all candidate chunk/resize settings without finding a fit. "
            + "Attempts: "
            + " | ".join(summary_lines)
            + f". Last error: {last_error_summary}"
        ) from None


    def initialize(
        self,
        frames: list[FrameRef],
        *,
        ingest_artifact=None,
        sync_artifact=None,
    ) -> GeometryArtifact:
        selected_frames = self._select_frames(frames, ingest_artifact)
        if not selected_frames:
            raise ValueError("VGGTLongInitializer.initialize() received no frames to process.")

        if not self._is_loaded and self.backend_mode != "heuristic":
            self.load_checkpoint()

        if self._VGGT_Long is None:
            artifact = self._initialize_heuristic(selected_frames, ingest_artifact, sync_artifact)
            artifact.backend_name = self.backend_name
            artifact.notes.append("VGGT-Long backend unavailable; emitted heuristic fallback geometry.")
            return artifact

        # Content-keyed input cache shared with VGGT (never an ingest-folder-named
        # directory another run would empty and rewrite).
        cache_root = input_cache_root(self.frame_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        # VGGT-Long pads each chunk's mixed-size images up to that chunk's largest, so a
        # frame's pixel mapping would depend on which chunk it fell in: refuse up front.
        source_sizes = self._stream_source_sizes(selected_frames, ingest_artifact)
        if len(set(source_sizes.values())) > 1:
            sizes = {sid: f"{w}x{h}" for sid, (w, h) in source_sizes.items()}
            raise ValueError(
                f"VGGT-Long got streams of different frame sizes ({sizes}); its per-chunk padding "
                "makes the frame-to-geometry pixel mapping chunk-dependent, which is not recorded "
                "-- run the streams separately"
            )

        with self._backend_outputs(ingest_artifact) as (backend_output_root, keep_outputs):
            attempt_history: list[dict[str, Any]] = []
            auto_tune_recommendation: dict[str, Any] | None = None
            if self.auto_tune:
                predictions, image_paths, notes, attempt_history, auto_tune_recommendation = self._run_auto_tuned(
                    selected_frames,
                    ingest_artifact=ingest_artifact,
                    cache_root=cache_root,
                    backend_output_root=backend_output_root,
                )
            else:
                if len(selected_frames) > self.chunk_size and self.overlap <= 0:
                    raise ValueError(
                        "VGGT-Long requires overlap > 0 when more than one chunk is needed. "
                        f"Got {len(selected_frames)} frames, chunk_size={self.chunk_size}, overlap={self.overlap}."
                    )
                image_paths, notes = self._materialize_images(
                    selected_frames,
                    ingest_artifact,
                    cache_root,
                    resize_max_dim=self.resize_max_dim,
                )
                image_dir = Path(image_paths[0]).parent if image_paths else cache_root
                runner = self._create_runner(image_dir, backend_output_root)
                runner.img_list = [str(path) for path in image_paths]
                self._run_runner(runner)
                chunks = self._load_chunk_predictions(runner)
                global_c2w = self._load_global_camera_to_world(runner, len(selected_frames))
                scales = self._chunk_scales(runner, len(runner.chunk_indices))
                predictions = self._combined_predictions_from_chunks(
                    selected_frames, chunks, global_c2w=global_c2w, chunk_scales=scales
                )
                notes = list(notes)
                notes.append(
                    "Used VGGT-Long globally-aligned camera_poses.txt for camera poses; each chunk's "
                    "depth scaled by its alignment scale ("
                    + ", ".join(f"{v:.3f}" for v in scales) + ")."
                )
                notes.append(
                    f"Initialized geometry with VGGT-Long using chunk_size={self.chunk_size}, overlap={self.overlap}."
                )
                notes.append(f"VGGT-Long processed {len(selected_frames)} selected frames across {len(chunks)} chunk file(s).")

            artifact = self._initialize_from_predictions(
                selected_frames,
                predictions,
                ingest_artifact,
                notes=notes,
                image_paths=image_paths,
            )
            artifact.backend_name = self.backend_name
            artifact.notes.append("Chunk overlaps take the later chunk, as camera_poses.txt does.")
            artifact.metadata["geometry_backend_name"] = self.backend_name
            artifact.metadata["vggt_long_chunk_size"] = self.chunk_size
            artifact.metadata["vggt_long_overlap"] = self.overlap
            artifact.metadata["vggt_long_loop_enable"] = self.loop_enable
            artifact.metadata["vggt_long_auto_tune"] = bool(self.auto_tune)
            artifact.metadata["vggt_long_auto_tune_attempts"] = attempt_history
            if auto_tune_recommendation is not None:
                artifact.metadata["vggt_long_auto_tune_recommendation"] = auto_tune_recommendation
            artifact.metadata["vggt_long_resize_max_dim"] = self.resize_max_dim
            loader, target = self._input_loader()
            # version 2: the no-upscale loader crops tall frames (version 1 squashed them).
            artifact.metadata["vggt_long_input_loader"] = {
                "loader": loader, "target_size": target,
            }
            self._record_transforms(artifact, {
                stream_id: vggt_input_transform(
                    size, loader=loader, target_size=target, resize_max_dim=self.resize_max_dim
                )
                for stream_id, size in source_sizes.items()
            })
            artifact.metadata["vggt_long_outputs"] = {
                "dir": data_dir.record_name(backend_output_root), "kept": keep_outputs,
            }
            return artifact

    @contextmanager
    def _backend_outputs(self, ingest_artifact):
        """VGGT-Long's own output folder for one invocation: ``(path, kept)``.

        Its chunk predictions and ``camera_poses.txt`` are read back once, while the
        geometry is built, and are not needed after. A folder shared by every run of
        the same input (``vggt_long_run/<input folder>``) could be written by two runs
        at once, so each invocation gets its own, and it is removed afterwards (about
        11 GB for 426 frames) unless ``keep_backend_outputs`` is set. An explicit
        ``output_cache_dir`` (a name in the data directory's cache) is the caller's and is
        always kept.
        """
        import shutil
        import tempfile

        if self.output_cache_dir:
            root = data_dir.path(data_dir.CACHE, self.output_cache_dir)
            root.mkdir(parents=True, exist_ok=True)
            yield root, True
            return
        keep = bool(self.cfg.get("keep_backend_outputs", False))
        parent = data_dir.path(data_dir.CACHE, "vggt_long_run")
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{ingest_artifact.run_id}-", dir=parent))
        try:
            yield root, keep
        finally:
            if not keep:
                shutil.rmtree(root, ignore_errors=True)

    def _input_loader(self) -> tuple[str, int]:
        """The loader ``_patched_vggt_long_preprocess`` runs for the current resize setting.

        The no-upscale ``bounded`` loader replaces the upstream one only when
        ``prevent_input_upscale`` is on and the effective target is below 518.
        """
        target = self._effective_vggt_input_target_size(self.resize_max_dim)
        if self.prevent_input_upscale and target < 518:
            return "bounded", int(target)
        return "upstream", 518

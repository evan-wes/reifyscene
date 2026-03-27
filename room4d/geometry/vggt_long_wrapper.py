"""Wrapper module for VGGT-Long coarse geometry initialization.

This adapter preserves the existing ``GeometryArtifact`` contract while
allowing the long-sequence backend to be selected next to the standard VGGT
initializer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from room4d.common.types import FrameRef
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.geometry.vggt_wrapper import VGGTInitializer


class VGGTLongInitializer(VGGTInitializer):
    """Adapter for the VGGT-Long offline long-sequence geometry backend.

    The backend is expected to expose a ``VGGT_Long`` class taking
    ``(image_dir, save_dir, config)`` and to emit chunk-level ``.npy`` files
    containing arrays with the same semantic fields as VGGT predictions.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.backend_name = "vggt-long"
        self.chunk_size = int(self.cfg.get("chunk_size", 16))
        self.overlap = int(self.cfg.get("overlap", 4))
        self.loop_enable = bool(self.cfg.get("loop_enable", False))
        self.use_dbow = bool(self.cfg.get("use_dbow", False))
        self.delete_temp_files = bool(self.cfg.get("delete_temp_files", False))
        self.output_cache_dir = self.cfg.get("output_cache_dir")
        self.long_config = dict(self.cfg.get("long_config", {}))
        self._VGGT_Long: Any | None = None

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

    def _default_long_config(self) -> dict[str, Any]:
        """Build a VGGT-Long config compatible with upstream expectations.

        This mirrors the structure of upstream `configs/base_config.yaml`
        closely enough for the Python pipeline path, while still allowing
        room4d-specific overrides.
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
            },
            "IRLS": {
                "delta": float(self.cfg.get("irls_delta", 0.1)),
                "max_iters": int(self.cfg.get("irls_max_iters", 5)),
                "tol": float(self.cfg.get("irls_tol", 1e-9)),
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
                    "lambda_init": self.cfg.get("sim3_lambda_init", "1e-6"),
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
            chunks.append(payload)
        if not chunks:
            raise RuntimeError("VGGT-Long produced no readable chunk predictions.")
        return chunks

    def _combined_predictions_from_chunks(
        self,
        frames: list[FrameRef],
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
                chunk_depth = np.asarray(chunk_depth)
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
                if extrinsic[global_idx] is not None:
                    continue  # keep first occurrence in overlapping windows
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
        return preds

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

        cache_root = Path(self.frame_cache_dir) if self.frame_cache_dir else Path(".room4d_cache") / "vggt_long_inputs"
        image_paths, notes = self._materialize_images(selected_frames, ingest_artifact, cache_root)
        backend_output_root = Path(self.output_cache_dir) if self.output_cache_dir else cache_root.parent / "vggt_long_run"
        backend_output_root.mkdir(parents=True, exist_ok=True)

        runner = self._VGGT_Long(str(cache_root), str(backend_output_root), self._default_long_config())

        requested_device = str(self.device).lower()
        if requested_device == "cpu":
            import torch

            runner.device = "cpu"
            runner.dtype = torch.float32

            if getattr(runner, "model", None) is not None:
                runner.model.device = "cpu"
                if hasattr(runner.model, "dtype"):
                    runner.model.dtype = torch.float32
        else:
            if getattr(runner, "model", None) is not None:
                runner.model.device = requested_device
            runner.device = requested_device

        runner.img_list = [str(path) for path in image_paths]
        runner.run()

        chunks = self._load_chunk_predictions(runner)
        predictions = self._combined_predictions_from_chunks(selected_frames, chunks)
        notes = list(notes)
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
        artifact.notes.append("Chunk overlaps were merged by first occurrence; downstream artifact contract preserved.")
        artifact.metadata["geometry_backend_name"] = self.backend_name
        artifact.metadata["vggt_long_chunk_size"] = self.chunk_size
        artifact.metadata["vggt_long_overlap"] = self.overlap
        artifact.metadata["vggt_long_loop_enable"] = self.loop_enable
        return artifact

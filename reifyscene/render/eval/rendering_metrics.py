"""Evaluation helpers for render quality and held-out observed views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class RenderingMetrics:
    """Compute simple render-vs-observed comparisons."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_views = max(1, int(self.cfg.get("max_views", 4)))

    @staticmethod
    def _camera_from_state(frame_ref, state) -> dict[str, Any]:
        Twc = np.asarray(state.pose.T_world_cam, dtype=np.float32)
        eye = Twc[:3, 3]
        forward = Twc[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        up = Twc[:3, :3] @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        return {
            "eye": eye.tolist(),
            "target": (eye + forward).tolist(),
            "up": up.tolist(),
            "time_sec": float(frame_ref.timestamp_sec),
            "frame_key": f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}",
            "frame_ref": frame_ref,
        }

    @staticmethod
    def _find_state(geometry_artifact, frame_ref):
        states = (getattr(geometry_artifact, "camera_states_by_stream", {}) or {}).get(frame_ref.stream_id, [])
        best = None
        best_dt = None
        for st in states:
            dt = abs(float(st.time_sec) - float(frame_ref.timestamp_sec))
            if best is None or dt < best_dt:
                best = st
                best_dt = dt
        return best

    @staticmethod
    def _compare_images(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
        pred_f = pred.astype(np.float32) / 255.0
        gt_f = gt.astype(np.float32) / 255.0
        mse = float(np.mean((pred_f - gt_f) ** 2))
        mae = float(np.mean(np.abs(pred_f - gt_f)))
        psnr = 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))
        return {"mse": mse, "mae": mae, "psnr": psnr}

    def compute(self, scene_artifact=None, render_artifact=None, geometry_artifact=None, renderer=None, *args, **kwargs):
        if scene_artifact is None or geometry_artifact is None or renderer is None:
            return {}
        frames = list(getattr(scene_artifact, "frame_refs", []) or [])
        if not frames:
            return {"num_heldout_views": 0}
        if len(frames) > self.max_views:
            idx = np.linspace(0, len(frames) - 1, self.max_views).round().astype(int).tolist()
            frames = [frames[i] for i in idx]
        image_map = dict((getattr(geometry_artifact, "metadata", {}) or {}).get("debug_image_paths", {}) or {})
        results: list[dict[str, Any]] = []
        for fr in frames:
            state = self._find_state(geometry_artifact, fr)
            gt_path = image_map.get(f"{fr.stream_id}:{int(fr.frame_idx)}")
            if state is None or gt_path is None or not Path(gt_path).exists():
                continue
            camera = self._camera_from_state(fr, state)
            try:
                images, _ = renderer._render_frame_bundle(scene_artifact, camera)
            except Exception:
                continue
            pred = np.asarray(images["full"].convert("RGB"), dtype=np.uint8)
            gt = np.asarray(Image.open(gt_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
            score = self._compare_images(pred, gt)
            results.append({"frame_key": camera["frame_key"], **score})
        if not results:
            return {"num_heldout_views": 0}
        psnr = np.asarray([r["psnr"] for r in results], dtype=np.float32)
        mae = np.asarray([r["mae"] for r in results], dtype=np.float32)
        return {
            "num_heldout_views": int(len(results)),
            "heldout_psnr_mean": float(np.mean(psnr)),
            "heldout_psnr_min": float(np.min(psnr)),
            "heldout_mae_mean": float(np.mean(mae)),
            "heldout_details": results,
        }

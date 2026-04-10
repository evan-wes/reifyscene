"""Evaluation helpers for geometry and world-frame diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


class GeometryMetrics:
    """Compute lightweight geometry health metrics."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_frames = max(1, int(self.cfg.get("max_frames", 12)))
        self.max_points_per_frame = max(128, int(self.cfg.get("max_points_per_frame", 2000)))

    def _sample_heights(self, geometry_artifact) -> np.ndarray:
        frames = list(getattr(geometry_artifact, "selected_frames", []) or [])
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int).tolist()
            frames = [frames[i] for i in idx]
        heights: list[np.ndarray] = []
        rng = np.random.default_rng(17)
        for fr in frames:
            frame_key = f"{fr.stream_id}:{int(fr.frame_idx)}"
            payload = (getattr(geometry_artifact, "pointmaps_by_frame", {}) or {}).get(frame_key)
            if not payload:
                continue
            pts = np.asarray(payload.get("points"), dtype=np.float32)
            if pts.ndim != 3 or pts.shape[-1] != 3:
                continue
            flat = pts.reshape(-1, 3)
            valid = np.isfinite(flat).all(axis=1)
            flat = flat[valid]
            if flat.shape[0] == 0:
                continue
            if flat.shape[0] > self.max_points_per_frame:
                choose = rng.choice(flat.shape[0], size=self.max_points_per_frame, replace=False)
                flat = flat[choose]
            heights.append(flat[:, 2].astype(np.float32))
        if not heights:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(heights, axis=0)

    def compute(self, geometry_artifact, *args, **kwargs):
        world_frame = dict((getattr(geometry_artifact, "metadata", {}) or {}).get("world_frame", {}) or {})
        heights = self._sample_heights(geometry_artifact)
        metrics: dict[str, Any] = {
            "num_selected_frames": int(len(getattr(geometry_artifact, "selected_frames", []) or [])),
            "camera_streams": int(len(getattr(geometry_artifact, "camera_states_by_stream", {}) or {})),
        }
        if heights.size:
            metrics["height_abs_median"] = float(np.median(np.abs(heights)))
            metrics["height_abs_p90"] = float(np.quantile(np.abs(heights), 0.90))
            metrics["height_p05"] = float(np.quantile(heights, 0.05))
            metrics["height_p50"] = float(np.quantile(heights, 0.50))
            metrics["height_p95"] = float(np.quantile(heights, 0.95))
        if world_frame:
            metrics["ground_plane_applied"] = True
            metrics["ground_inlier_ratio"] = float(world_frame.get("inlier_ratio", 0.0))
            metrics["ground_num_inliers"] = int(world_frame.get("num_inliers", 0))
            metrics["ground_num_candidates"] = int(len(world_frame.get("candidate_planes", []) or []))
            normal = np.asarray(world_frame.get("normal_world", [0.0, 0.0, 1.0]), dtype=np.float32)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                metrics["ground_normal_alignment_to_z"] = float(abs(normal[2] / norm))
        else:
            metrics["ground_plane_applied"] = False
        return metrics

"""Evaluation helpers for tracking, lifting, and semantic association."""

from __future__ import annotations

from typing import Any

import numpy as np


class TrackingMetrics:
    """Compute tracking/lifting and human-association metrics."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_reproj_samples = max(32, int(self.cfg.get("max_reproj_samples", 20000)))

    @staticmethod
    def _nearest_camera_state(geometry_artifact, stream_id: str, time_sec: float):
        states = (getattr(geometry_artifact, "camera_states_by_stream", {}) or {}).get(stream_id, [])
        best = None
        best_dt = None
        for st in states:
            dt = abs(float(st.time_sec) - float(time_sec))
            if best is None or dt < best_dt:
                best = st
                best_dt = dt
        return best

    def _project_world(self, xyz: np.ndarray, state) -> np.ndarray:
        Twc = np.asarray(state.pose.T_world_cam, dtype=np.float32)
        Tcw = np.linalg.inv(Twc)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        cam = xyz_h @ Tcw.T
        valid = np.isfinite(cam).all(axis=1) & (np.abs(cam[:, 2]) > 1e-5)
        xy = np.full((xyz.shape[0], 2), np.nan, dtype=np.float32)
        if not np.any(valid):
            return xy
        intr = state.intrinsics
        z = cam[valid, 2]
        xy[valid, 0] = float(intr.cx) + float(intr.fx) * (cam[valid, 0] / z)
        xy[valid, 1] = float(intr.cy) + float(intr.fy) * (cam[valid, 1] / z)
        return xy

    def _reprojection_metrics(self, track_artifact, geometry_artifact) -> dict[str, Any]:
        if track_artifact is None or geometry_artifact is None:
            return {}
        errors: list[np.ndarray] = []
        total = 0
        for stream_id, payload in (getattr(track_artifact, "track_payloads_by_stream", {}) or {}).items():
            frame_indices = np.asarray(payload.get("frame_indices", []), dtype=int)
            points_xy = np.asarray(payload.get("points_xy", np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
            lifted_idx = np.asarray(payload.get("refined_lifted_frame_indices", payload.get("lifted_frame_indices", [])), dtype=int)
            lifted_xyz = np.asarray(payload.get("refined_lifted_xyz", payload.get("lifted_xyz", np.zeros((0, 0, 3), dtype=np.float32))), dtype=np.float32)
            lifted_valid = np.asarray(payload.get("refined_lifted_valid", payload.get("lifted_valid", np.zeros((0, 0), dtype=bool))))
            if points_xy.ndim != 3 or lifted_xyz.ndim != 3 or lifted_idx.size == 0:
                continue
            frame_lookup = {int(fi): t for t, fi in enumerate(frame_indices.tolist())}
            for k, fi in enumerate(lifted_idx.tolist()):
                t = frame_lookup.get(int(fi))
                if t is None:
                    continue
                frame_ref = None
                for fr in (getattr(track_artifact, "selected_frames", []) or []):
                    if fr.stream_id == stream_id and int(fr.frame_idx) == int(fi):
                        frame_ref = fr
                        break
                time_sec = float(frame_ref.timestamp_sec) if frame_ref is not None else float(fi)
                state = self._nearest_camera_state(geometry_artifact, stream_id, time_sec)
                if state is None:
                    continue
                xyz = lifted_xyz[k]
                valid = np.asarray(lifted_valid[k], dtype=bool) & np.isfinite(xyz).all(axis=1)
                if not np.any(valid):
                    continue
                proj = self._project_world(xyz[valid], state)
                gt = points_xy[t][valid]
                good = np.isfinite(proj).all(axis=1) & np.isfinite(gt).all(axis=1)
                if not np.any(good):
                    continue
                err = np.linalg.norm(proj[good] - gt[good], axis=1)
                errors.append(err.astype(np.float32))
                total += int(err.size)
        if not errors:
            return {"num_reprojection_samples": 0}
        errs = np.concatenate(errors, axis=0)
        if errs.size > self.max_reproj_samples:
            choose = np.linspace(0, errs.size - 1, self.max_reproj_samples).round().astype(int)
            errs = errs[choose]
        return {
            "num_reprojection_samples": int(errs.size),
            "reprojection_error_mean_px": float(np.mean(errs)),
            "reprojection_error_median_px": float(np.median(errs)),
            "reprojection_error_p90_px": float(np.quantile(errs, 0.90)),
        }

    @staticmethod
    def _human_continuity_metrics(human_artifact) -> dict[str, Any]:
        if human_artifact is None:
            return {}
        grouped: dict[str, list[int]] = {}
        for h in getattr(human_artifact, "humans", []) or []:
            grouped.setdefault(str(h.human_id), []).append(int(h.frame_ref.frame_idx))
        if not grouped:
            return {"num_human_tracks": 0}
        fragment_counts: list[int] = []
        lifespans: list[int] = []
        gaps: list[int] = []
        for _, frames in grouped.items():
            uniq = sorted(set(frames))
            if not uniq:
                continue
            lifespans.append(int(uniq[-1] - uniq[0] + 1))
            fragments = 1
            for a, b in zip(uniq[:-1], uniq[1:]):
                gap = int(b - a)
                if gap > 1:
                    fragments += 1
                    gaps.append(gap)
            fragment_counts.append(fragments)
        summaries = dict((getattr(human_artifact, "linked_ids", {}) or {}).get("track_summaries", {}) or {})
        return {
            "num_human_tracks": int(len(grouped)),
            "mean_human_lifespan_frames": float(np.mean(lifespans)) if lifespans else 0.0,
            "mean_human_fragments": float(np.mean(fragment_counts)) if fragment_counts else 0.0,
            "num_fragmented_tracks": int(sum(1 for v in fragment_counts if v > 1)),
            "max_human_gap_frames": int(max(gaps) if gaps else 0),
            "mean_human_quality_score": float(np.mean([float(v.get("quality_score", 0.0)) for v in summaries.values()])) if summaries else 0.0,
        }

    def compute(self, track_artifact=None, geometry_artifact=None, human_artifact=None, *args, **kwargs):
        metrics = {}
        metrics.update(self._reprojection_metrics(track_artifact, geometry_artifact))
        metrics.update(self._human_continuity_metrics(human_artifact))
        return metrics

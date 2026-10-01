"""Joint spatiotemporal solver scaffold."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.image_transform import (
    ImageTransform,
    MissingImageTransform,
    image_transform_for,
)
from reifyscene.lifting.lift import lift_pixels, lift_record
from reifyscene.optimize.factors.priors import PriorFactor
from reifyscene.optimize.factors.reprojection import ReprojectionFactor
from reifyscene.optimize.factors.time_alignment import TimeAlignmentFactor
from reifyscene.optimize.factors.track_consistency import TrackConsistencyFactor
from reifyscene.optimize.optimization_artifact import OptimizationArtifact
from reifyscene.optimize.stages.schedule import OptimizationSchedule


class JointSpatiotemporalSolver:
    """Coordinate staged optimization across cameras, time, tracks, and actors.

    The initial scaffold mostly reported factor diagnostics. This version adds a
    first real refinement pass: high-confidence *static* stitched tracks are
    used to estimate small rigid corrections for keyframe camera poses.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """Store solver configuration and stage schedule."""
        self.cfg = cfg or {}
        self.schedule = OptimizationSchedule()
        self.static_track_cfg = {
            "enabled": bool(self.cfg.get("enable_static_track_refinement", True)),
            "iterations": max(int(self.cfg.get("static_track_refine_iterations", 2)), 1),
            "min_static_tracks_per_frame": max(int(self.cfg.get("static_track_min_tracks", 3)), 3),
            "min_track_observations": max(int(self.cfg.get("static_track_min_observations", 2)), 2),
            "max_rotation_deg": float(self.cfg.get("static_track_max_rotation_deg", 8.0)),
            "max_translation": float(self.cfg.get("static_track_max_translation", 0.10)),
            "accept_improvement_ratio": float(self.cfg.get("static_track_accept_improvement_ratio", 0.98)),
            "relabel_static_threshold_world": float(self.cfg.get("relabel_static_threshold_world", 0.05)),
            "relabel_min_lifted_frames": max(int(self.cfg.get("relabel_min_lifted_frames", 2)), 2),
        }

    @staticmethod
    def _copy_intrinsics(intr: CameraIntrinsics) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            distortion_model=str(intr.distortion_model),
            distortion_params=None if intr.distortion_params is None else np.asarray(intr.distortion_params, dtype=float).copy(),
        )

    @classmethod
    def _copy_camera_state(cls, state: TimedCameraState) -> TimedCameraState:
        return TimedCameraState(
            stream_id=str(state.stream_id),
            time_sec=float(state.time_sec),
            intrinsics=cls._copy_intrinsics(state.intrinsics),
            pose=CameraPose(T_world_cam=np.asarray(state.pose.T_world_cam, dtype=float).copy()),
        )

    def build_state(
        self,
        sync_artifact,
        geometry_artifact,
        track_artifact,
        human_artifact=None,
        object_artifact=None,
        graph_artifact=None,
    ) -> dict[str, Any]:
        """Build an initial solver state dictionary.

        The state is intentionally plain-Python so it can be inspected and later
        migrated to torch or scipy parameter containers. The geometry's image
        transforms and the tracking lift's settings travel with it, so the re-lift
        maps track pixels exactly as the tracking stage did.
        """
        # Each stream's transform, or why it has none (raised only if a re-lift needs it).
        image_transforms: dict[str, dict[str, Any] | str] = {}
        for stream_id, payload in track_artifact.track_payloads_by_stream.items():
            original_wh = tuple(int(v) for v in payload.get("original_size_wh", [0, 0]))
            try:
                image_transforms[str(stream_id)] = image_transform_for(
                    geometry_artifact, str(stream_id), source_size=original_wh
                ).to_dict()
            except MissingImageTransform as exc:
                image_transforms[str(stream_id)] = str(exc)
        lift_settings = dict((getattr(track_artifact, "metadata", {}) or {}).get("lift") or {})
        return {
            "run_id": geometry_artifact.run_id,
            "stream_ids": geometry_artifact.stream_ids() or sync_artifact.stream_ids(),
            "time_offsets": dict(sync_artifact.offsets_by_stream),
            "camera_intrinsics": {k: self._copy_intrinsics(v) for k, v in geometry_artifact.intrinsics_by_stream.items()},
            "camera_states": {
                k: [self._copy_camera_state(state) for state in states]
                for k, states in geometry_artifact.camera_states_by_stream.items()
            },
            "selected_frames": list(geometry_artifact.selected_frames),
            "depthmaps_by_frame": {
                k: {
                    "depth": np.asarray(v.get("depth")).copy(),
                    "confidence": None if v.get("confidence") is None else np.asarray(v.get("confidence")).copy(),
                }
                for k, v in geometry_artifact.depthmaps_by_frame.items()
            },
            "track_observations": dict(track_artifact.per_stream_observations),
            "track_payloads": {
                k: {
                    kk: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
                    for kk, v in payload.items()
                }
                for k, payload in track_artifact.track_payloads_by_stream.items()
            },
            "image_transforms": image_transforms,
            "lift_settings": lift_settings,
            "graph_nodes": list(graph_artifact.nodes) if graph_artifact is not None else [],
            "graph_edges": list(graph_artifact.edges) if graph_artifact is not None else [],
            "human_artifact_present": human_artifact is not None,
            "object_artifact_present": object_artifact is not None,
            "stage_history": [],
            "track_refinement_summary": {},
            "notes": [
                "TODO: replace plain dict state with structured parameter containers.",
                "TODO: add robust outlier rejection and sparse Jacobian optimization.",
            ],
        }

    def build_default_factors(self, state: dict[str, Any], stage_cfg: dict[str, Any]) -> list[Any]:
        """Construct a minimal factor list for the requested stage."""
        factors: list[Any] = []
        factor_names = stage_cfg.get("factor_names", [])
        if "time_alignment" in factor_names:
            factors.append(TimeAlignmentFactor(offsets_by_stream=state.get("time_offsets", {}), weight=1.0))
        if "track_consistency" in factor_names:
            per_stream_counts = {
                stream_id: len(observations)
                for stream_id, observations in state.get("track_observations", {}).items()
            }
            factors.append(TrackConsistencyFactor(per_stream_counts=per_stream_counts, weight=1.0))
        if "reprojection" in factor_names:
            observations = [
                observation
                for obs_list in state.get("track_observations", {}).values()
                for observation in obs_list
            ]
            factors.append(ReprojectionFactor(observations=observations, weight=1.0))
        if "priors" in factor_names:
            num_priors = len(state.get("camera_intrinsics", {})) + len(state.get("time_offsets", {}))
            factors.append(PriorFactor(num_priors=num_priors, weight=1.0))
        return factors

    @staticmethod
    def _find_state_for_frame(states: list[TimedCameraState], frame: FrameRef) -> TimedCameraState | None:
        best = None
        best_dt = None
        for state in states:
            dt = abs(float(state.time_sec) - float(frame.timestamp_sec))
            if best is None or dt < best_dt:
                best = state
                best_dt = dt
        return best

    @staticmethod
    def _estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if src.shape != dst.shape or src.shape[0] < 3:
            return None
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)
        src0 = src - src_center[None, :]
        dst0 = dst - dst_center[None, :]
        H = src0.T @ dst0
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return None
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = dst_center - (R @ src_center)
        return R.astype(np.float64), t.astype(np.float64)

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray) -> float:
        trace = float(np.trace(R))
        cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
        return float(np.degrees(np.arccos(cos_theta)))

    def _clamp_transform(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_rot_deg = self.static_track_cfg["max_rotation_deg"]
        max_trans = self.static_track_cfg["max_translation"]
        angle_deg = self._rotation_angle_deg(R)
        t_norm = float(np.linalg.norm(t))
        if angle_deg > max_rot_deg or not np.isfinite(angle_deg):
            R = np.eye(3, dtype=np.float64)
        if t_norm > max_trans and t_norm > 1e-9:
            t = t * (max_trans / t_norm)
        return R, t

    @staticmethod
    def _apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (points @ R.T) + t[None, :]

    def _static_track_error_stats(
        self,
        refined_xyz: np.ndarray,
        lifted_valid: np.ndarray,
        static_mask: np.ndarray,
    ) -> dict[str, float | None]:
        errors: list[float] = []
        refs = np.full((refined_xyz.shape[1], 3), np.nan, dtype=np.float64)
        for tidx in np.where(static_mask)[0]:
            valid = lifted_valid[:, tidx].astype(bool)
            xyz = refined_xyz[valid, tidx]
            if xyz.shape[0] < self.static_track_cfg["min_track_observations"]:
                continue
            refs[tidx] = np.nanmedian(xyz, axis=0)
        for frame_idx in range(refined_xyz.shape[0]):
            valid = lifted_valid[frame_idx].astype(bool) & static_mask & np.isfinite(refs).all(axis=1)
            if not np.any(valid):
                continue
            errs = np.linalg.norm(refined_xyz[frame_idx, valid] - refs[valid], axis=1)
            errors.extend(errs.tolist())
        if not errors:
            return {"mean": None, "median": None}
        err_arr = np.asarray(errors, dtype=np.float64)
        return {"mean": float(np.mean(err_arr)), "median": float(np.median(err_arr))}

    def _relabel_payload(self, payload: dict[str, Any], *, xyz_key: str, valid_key: str, label_key: str, score_key: str) -> dict[str, Any]:
        xyz = np.asarray(payload.get(xyz_key, np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        valid = np.asarray(payload.get(valid_key, np.zeros((0, 0), dtype=bool)))
        num_tracks = xyz.shape[1] if xyz.ndim == 3 else 0
        labels = np.full((num_tracks,), -1, dtype=np.int8)
        scores = np.full((num_tracks,), np.nan, dtype=np.float32)
        min_frames = int(self.static_track_cfg["relabel_min_lifted_frames"])
        thr = float(self.static_track_cfg["relabel_static_threshold_world"])
        for track_idx in range(num_tracks):
            track_valid = valid[:, track_idx].astype(bool) if valid.size else np.isfinite(xyz[:, track_idx]).all(axis=1)
            pts = xyz[track_valid, track_idx]
            if pts.shape[0] < min_frames:
                continue
            center = np.nanmedian(pts, axis=0)
            spread = np.linalg.norm(pts - center[None, :], axis=1)
            score = float(np.nanmedian(spread))
            scores[track_idx] = score
            labels[track_idx] = 1 if score <= thr else 0
        payload[label_key] = labels
        payload[score_key] = scores
        return {
            "static_count": int(np.sum(labels > 0)),
            "dynamic_count": int(np.sum(labels == 0)),
            "uncertain_count": int(np.sum(labels < 0)),
            "threshold_world": thr,
            "min_lifted_frames": min_frames,
        }

    def _relift_tracks_with_refined_cameras(self, state: dict[str, Any]) -> None:
        """Re-lift every track with the refined cameras, through the tracking stage's lift.

        Same function (:func:`lift_pixels`), transform and confidence threshold the
        tracking stage used; only the cameras differ. Records ``state['relift']``.
        """
        selected_frames = list(state.get("selected_frames", []))
        depthmaps_by_frame = state.get("depthmaps_by_frame", {})
        camera_states = state.get("camera_states", {})
        lift_settings = state.get("lift_settings", {}) or {}
        conf_threshold = float(lift_settings.get("point_conf_threshold", 0.0))
        used_transforms: dict[str, Any] = {}
        for stream_id, payload in state.get("track_payloads", {}).items():
            frame_indices = np.asarray(payload.get("frame_indices", []), dtype=int)
            points_xy = np.asarray(payload.get("points_xy", np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
            visibility = np.asarray(payload.get("visibility", np.zeros((0, 0), dtype=bool)))
            num_tracks = points_xy.shape[1] if points_xy.ndim == 3 else 0
            stream_frames = [f for f in selected_frames if f.stream_id == stream_id]
            refined_xyz = np.full((len(stream_frames), num_tracks, 3), np.nan, dtype=np.float32)
            refined_valid = np.zeros((len(stream_frames), num_tracks), dtype=bool)
            lifted_frame_indices = np.asarray([f.frame_idx for f in stream_frames], dtype=np.int32)
            original_wh = tuple(int(v) for v in payload.get("original_size_wh", [0, 0]))
            transform = None
            if stream_frames and num_tracks:
                raw = (state.get("image_transforms", {}) or {}).get(str(stream_id))
                if not isinstance(raw, dict):
                    raise MissingImageTransform(
                        raw or f"stream {stream_id!r}: no image transform to re-lift its tracks with"
                    )
                transform = ImageTransform.from_dict(raw)
                used_transforms[str(stream_id)] = raw
            for k, frame in enumerate(stream_frames):
                matches = np.where(frame_indices == int(frame.frame_idx))[0]
                if matches.size == 0 or transform is None:
                    continue
                t = int(matches[0])
                frame_key = make_frame_key(frame.stream_id, frame.frame_idx)
                depth_payload = depthmaps_by_frame.get(frame_key)
                state_for_frame = self._find_state_for_frame(camera_states.get(stream_id, []), frame)
                if depth_payload is None or state_for_frame is None:
                    continue
                vis = visibility[t].astype(bool) if visibility.size else np.ones((num_tracks,), dtype=bool)
                xyz, valid = lift_pixels(
                    points_xy[t], transform, depth_payload.get("depth"), state_for_frame.intrinsics,
                    state_for_frame.pose.T_world_cam, confidence=depth_payload.get("confidence"),
                    conf_threshold=conf_threshold, from_size=original_wh,
                )
                valid &= vis
                xyz[~valid] = np.nan
                refined_xyz[k] = xyz.astype(np.float32)
                refined_valid[k] = valid
            payload["refined_lifted_frame_indices"] = lifted_frame_indices
            payload["refined_lifted_xyz"] = refined_xyz
            payload["refined_lifted_valid"] = refined_valid
            payload["refined_classification_summary"] = self._relabel_payload(
                payload,
                xyz_key="refined_lifted_xyz",
                valid_key="refined_lifted_valid",
                label_key="refined_static_labels",
                score_key="refined_static_scores",
            )
        state["relift"] = lift_record(confidence="depth", conf_threshold=conf_threshold, transforms=used_transforms)

    def _refine_static_track_geometry(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.static_track_cfg["enabled"]:
            return {"enabled": False, "applied": False}

        payloads = state.get("track_payloads", {})
        selected_frames = list(state.get("selected_frames", []))
        camera_states = state.get("camera_states", {})
        summary: dict[str, Any] = {"enabled": True, "iterations": int(self.static_track_cfg["iterations"]), "streams": {}}

        for stream_id, payload in payloads.items():
            track_ids = np.asarray(payload.get("track_ids", []), dtype=np.int32)
            lifted_xyz = np.asarray(payload.get("lifted_xyz", np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float64)
            lifted_valid = np.asarray(payload.get("lifted_valid", np.zeros((0, 0), dtype=bool)))
            static_labels = np.asarray(payload.get("static_labels", np.zeros((track_ids.shape[0],), dtype=np.int8)), dtype=np.int8)
            if lifted_xyz.ndim != 3 or lifted_xyz.shape[1] == 0:
                summary["streams"][stream_id] = {"applied": False, "reason": "no_lifted_tracks"}
                continue
            static_mask = static_labels == 1
            if int(np.count_nonzero(static_mask)) < self.static_track_cfg["min_static_tracks_per_frame"]:
                summary["streams"][stream_id] = {
                    "applied": False,
                    "reason": "insufficient_static_tracks",
                    "num_static_tracks": int(np.count_nonzero(static_mask)),
                }
                continue

            stream_frames = [f for f in selected_frames if f.stream_id == stream_id]
            stream_states = camera_states.get(stream_id, [])
            if len(stream_frames) != lifted_xyz.shape[0] or not stream_states:
                summary["streams"][stream_id] = {
                    "applied": False,
                    "reason": "frame_state_mismatch",
                    "num_frames": len(stream_frames),
                    "lifted_frames": int(lifted_xyz.shape[0]),
                }
                continue

            refined_xyz = lifted_xyz.copy()
            frame_updates: list[dict[str, Any]] = []
            accepted_updates = 0
            initial_error_stats = self._static_track_error_stats(refined_xyz, lifted_valid, static_mask)

            for _iter in range(self.static_track_cfg["iterations"]):
                refs = np.full((track_ids.shape[0], 3), np.nan, dtype=np.float64)
                for tidx in np.where(static_mask)[0]:
                    valid = lifted_valid[:, tidx].astype(bool)
                    xyz = refined_xyz[valid, tidx]
                    if xyz.shape[0] < self.static_track_cfg["min_track_observations"]:
                        continue
                    refs[tidx] = np.nanmedian(xyz, axis=0)

                for frame_idx, frame in enumerate(stream_frames):
                    state_for_frame = self._find_state_for_frame(stream_states, frame)
                    if state_for_frame is None:
                        continue
                    valid = lifted_valid[frame_idx].astype(bool) & static_mask & np.isfinite(refs).all(axis=1)
                    idxs = np.where(valid)[0]
                    if idxs.size < self.static_track_cfg["min_static_tracks_per_frame"]:
                        continue
                    src = refined_xyz[frame_idx, idxs]
                    dst = refs[idxs]
                    before_err = np.linalg.norm(src - dst, axis=1)
                    transform = self._estimate_rigid_transform(src, dst)
                    if transform is None:
                        continue
                    R, t = transform
                    R, t = self._clamp_transform(R, t)
                    candidate = self._apply_transform(src, R, t)
                    after_err = np.linalg.norm(candidate - dst, axis=1)
                    before_med = float(np.median(before_err)) if before_err.size else float("nan")
                    after_med = float(np.median(after_err)) if after_err.size else float("nan")
                    improvement = after_med / max(before_med, 1e-9)
                    accepted = bool(np.isfinite(after_med) and np.isfinite(before_med) and after_med <= before_med * self.static_track_cfg["accept_improvement_ratio"])
                    frame_updates.append({
                        "frame_idx": int(frame.frame_idx),
                        "timestamp_sec": float(frame.timestamp_sec),
                        "num_static_tracks": int(idxs.size),
                        "median_error_before": before_med,
                        "median_error_after": after_med,
                        "rotation_deg": self._rotation_angle_deg(R),
                        "translation_norm": float(np.linalg.norm(t)),
                        "accepted": accepted,
                    })
                    if not accepted:
                        continue
                    accepted_updates += 1
                    # Update all lifted points for this frame in the refined world frame.
                    valid_all = lifted_valid[frame_idx].astype(bool)
                    refined_xyz[frame_idx, valid_all] = self._apply_transform(refined_xyz[frame_idx, valid_all], R, t)
                    A = np.eye(4, dtype=np.float64)
                    A[:3, :3] = R
                    A[:3, 3] = t
                    state_for_frame.pose.T_world_cam = (A @ np.asarray(state_for_frame.pose.T_world_cam, dtype=np.float64)).astype(np.float32)

            payload['refined_lifted_xyz'] = refined_xyz.astype(np.float32)
            final_error_stats = self._static_track_error_stats(refined_xyz, lifted_valid, static_mask)
            payload['refinement_summary'] = {
                "applied": accepted_updates > 0,
                "accepted_updates": int(accepted_updates),
                "num_static_tracks": int(np.count_nonzero(static_mask)),
                "mean_error_before": initial_error_stats["mean"],
                "mean_error_after": final_error_stats["mean"],
                "median_error_before": initial_error_stats["median"],
                "median_error_after": final_error_stats["median"],
                "frame_updates": frame_updates,
            }
            summary["streams"][stream_id] = payload['refinement_summary']

        self._relift_tracks_with_refined_cameras(state)
        for stream_id, payload in payloads.items():
            refined_summary = payload.get("refined_classification_summary", {})
            if stream_id in summary["streams"]:
                summary["streams"][stream_id]["refined_static_count"] = int(refined_summary.get("static_count", 0))
                summary["streams"][stream_id]["refined_dynamic_count"] = int(refined_summary.get("dynamic_count", 0))
                summary["streams"][stream_id]["refined_uncertain_count"] = int(refined_summary.get("uncertain_count", 0))
        state['track_refinement_summary'] = summary
        return summary

    def solve_stage(self, state: dict[str, Any], factors: list[Any], stage_cfg: dict[str, Any]) -> dict[str, Any]:
        """Run one optimization stage and return updated state."""
        stage_extra: dict[str, Any] = {}
        if stage_cfg.get("name") == "stage1_sync_pose":
            stage_extra["static_track_refinement"] = self._refine_static_track_geometry(state)

        factor_diagnostics: list[dict[str, Any]] = []
        for factor in factors:
            residual, diagnostics = factor.residual(state)
            factor_diagnostics.append(
                {
                    "name": factor.name,
                    "residual": residual,
                    "diagnostics": diagnostics,
                }
            )

        entry = {
            "stage_name": stage_cfg["name"],
            "active_parameter_groups": list(stage_cfg.get("active_parameter_groups", [])),
            "factor_diagnostics": factor_diagnostics,
            "todo": list(stage_cfg.get("todo", [])),
        }
        entry.update(stage_extra)
        state["stage_history"].append(entry)
        return state

    def solve_all(
        self,
        sync_artifact,
        geometry_artifact,
        track_artifact,
        human_artifact=None,
        object_artifact=None,
        graph_artifact=None,
    ) -> OptimizationArtifact:
        """Run all configured stages and return an optimization artifact."""
        state = self.build_state(
            sync_artifact,
            geometry_artifact,
            track_artifact,
            human_artifact=human_artifact,
            object_artifact=object_artifact,
            graph_artifact=graph_artifact,
        )
        pending_todos: list[str] = []
        schedule_used: list[str] = []

        for stage_cfg in self.schedule.all_stages():
            factors = self.build_default_factors(state, stage_cfg)
            state = self.solve_stage(state, factors, stage_cfg)
            schedule_used.append(stage_cfg["name"])
            pending_todos.extend(stage_cfg.get("todo", []))

        return OptimizationArtifact(
            run_id=geometry_artifact.run_id,
            stage_name="optimization",  # the stage registry's name (was "optimize")
            save_dir=Path("."),
            refined_cameras=state.get("camera_states", {}),
            refined_offsets=state.get("time_offsets", {}),
            tracks_3d=state.get("track_refinement_summary", {}),
            refined_track_payloads_by_stream=state.get("track_payloads", {}),
            selected_frames=list(state.get("selected_frames", [])),
            object_trajectories={},
            human_trajectories={},
            stage_diagnostics=list(state.get("stage_history", [])),
            schedule_used=schedule_used,
            pending_todos=pending_todos + state.get("notes", []),
            metadata={"relift": state["relift"]} if "relift" in state else {},
        )

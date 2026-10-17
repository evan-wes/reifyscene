"""Ground-plane detection and world-frame canonicalization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.types import CameraPose, TimedCameraState


@dataclass
class GroundPlaneEstimate:
    normal_world: np.ndarray
    offset: float
    inlier_ratio: float
    num_points: int
    num_inliers: int
    transform_world_old_to_new: np.ndarray
    chosen_score: float = 0.0
    preferred_up: np.ndarray | None = None
    candidate_planes: list[dict[str, Any]] = field(default_factory=list)
    lock_gravity_vector: bool = False
    manual_up_vector: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "normal_world": np.asarray(self.normal_world, dtype=float).tolist(),
            "offset": float(self.offset),
            "inlier_ratio": float(self.inlier_ratio),
            "num_points": int(self.num_points),
            "num_inliers": int(self.num_inliers),
            "chosen_score": float(self.chosen_score),
            "preferred_up": None if self.preferred_up is None else np.asarray(self.preferred_up, dtype=float).tolist(),
            "candidate_planes": list(self.candidate_planes),
            "lock_gravity_vector": bool(self.lock_gravity_vector),
            "manual_up_vector": None if self.manual_up_vector is None else np.asarray(self.manual_up_vector, dtype=float).tolist(),
            "transform_world_old_to_new": np.asarray(self.transform_world_old_to_new, dtype=float).tolist(),
        }


class GroundPlaneCanonicalizer:
    """Estimate a dominant ground plane and rotate the scene to global Z-up."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.max_frames = max(1, int(cfg.get("max_frames", 24)))
        self.max_points_per_frame = max(256, int(cfg.get("max_points_per_frame", 3000)))
        self.ransac_iterations = max(16, int(cfg.get("ransac_iterations", 160)))
        self.distance_threshold = float(cfg.get("distance_threshold", 0.035))
        self.min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.06))
        self.min_alignment_cos = float(cfg.get("min_alignment_cos", 0.45))
        self.alignment_weight = float(cfg.get("alignment_weight", 0.18))
        self.camera_up_axis = str(cfg.get("camera_up_axis", "-y")).strip().lower()
        self.random_seed = int(cfg.get("random_seed", 7))
        self.manual_up_vector = None if cfg.get("manual_up_vector") in (None, "", []) else np.asarray(cfg.get("manual_up_vector"), dtype=np.float64)
        self.lock_gravity_vector = bool(cfg.get("lock_gravity_vector", False))
        self.floor_height_quantile = float(cfg.get("floor_height_quantile", 0.06))
        self.max_saved_candidates = max(4, int(cfg.get("max_saved_candidates", 12)))

    @staticmethod
    def _frame_key(stream_id: str, frame_idx: int) -> str:
        return make_frame_key(stream_id, frame_idx)

    @staticmethod
    def _normalize(vec: np.ndarray | None) -> np.ndarray | None:
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            return None
        return arr / norm

    @staticmethod
    def _rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a /= max(np.linalg.norm(a), 1e-9)
        b /= max(np.linalg.norm(b), 1e-9)
        v = np.cross(a, b)
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        s = float(np.linalg.norm(v))
        if s < 1e-8:
            if c > 0.0:
                return np.eye(3, dtype=np.float64)
            axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            axis = axis - np.dot(axis, a) * a
            axis /= max(np.linalg.norm(axis), 1e-9)
            x, y, z = axis
            K = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
            return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
        K = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + K + K @ K * ((1.0 - c) / (s * s))

    def _sample_points(self, geometry_artifact) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        frames = list(getattr(geometry_artifact, "selected_frames", []) or [])
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int)
            frames = [frames[int(i)] for i in indices.tolist()]
        sampled: list[np.ndarray] = []
        for frame in frames:
            payload = (getattr(geometry_artifact, "pointmaps_by_frame", {}) or {}).get(self._frame_key(frame.stream_id, frame.frame_idx))
            if payload is None:
                continue
            pts = np.asarray(payload.get("points"), dtype=np.float32)
            if pts.ndim != 3 or pts.shape[-1] != 3:
                continue
            valid = np.isfinite(pts).all(axis=2)
            conf = payload.get("confidence")
            if conf is not None:
                conf_arr = np.asarray(conf, dtype=np.float32)
                if conf_arr.shape == valid.shape and np.any(valid):
                    conf_valid = conf_arr[valid]
                    threshold = float(np.nanmedian(conf_valid)) if conf_valid.size else 0.0
                    valid &= conf_arr >= threshold
            pts_flat = pts[valid]
            if pts_flat.shape[0] == 0:
                continue
            if pts_flat.shape[0] > self.max_points_per_frame:
                choose = rng.choice(pts_flat.shape[0], size=self.max_points_per_frame, replace=False)
                pts_flat = pts_flat[choose]
            sampled.append(pts_flat.astype(np.float64))
        if not sampled:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(sampled, axis=0)

    @staticmethod
    def _camera_centers(geometry_artifact) -> np.ndarray:
        centers: list[np.ndarray] = []
        for states in (getattr(geometry_artifact, "camera_states_by_stream", {}) or {}).values():
            for state in states:
                Twc = np.asarray(state.pose.T_world_cam, dtype=np.float64)
                if Twc.shape == (4, 4) and np.isfinite(Twc[:3, 3]).all():
                    centers.append(Twc[:3, 3].astype(np.float64))
        if not centers:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(centers, axis=0)

    def _preferred_up_direction(self, geometry_artifact) -> np.ndarray | None:
        if self.manual_up_vector is not None:
            return self._normalize(self.manual_up_vector)
        axes: list[np.ndarray] = []
        axis_map = {
            "x": lambda R: R[:, 0], "+x": lambda R: R[:, 0], "-x": lambda R: -R[:, 0],
            "y": lambda R: R[:, 1], "+y": lambda R: R[:, 1], "-y": lambda R: -R[:, 1],
            "z": lambda R: R[:, 2], "+z": lambda R: R[:, 2], "-z": lambda R: -R[:, 2],
        }
        extractor = axis_map.get(self.camera_up_axis, axis_map["-y"])
        ref: np.ndarray | None = None
        for states in (getattr(geometry_artifact, "camera_states_by_stream", {}) or {}).values():
            for state in states:
                Twc = np.asarray(state.pose.T_world_cam, dtype=np.float64)
                if Twc.shape != (4, 4) or not np.isfinite(Twc[:3, :3]).all():
                    continue
                up = np.asarray(extractor(Twc[:3, :3]), dtype=np.float64)
                norm = float(np.linalg.norm(up))
                if norm < 1e-8:
                    continue
                up = up / norm
                if ref is None:
                    ref = up
                elif float(np.dot(up, ref)) < 0.0:
                    up = -up
                axes.append(up)
        if not axes:
            return None
        mean = np.mean(np.stack(axes, axis=0), axis=0)
        return self._normalize(mean)

    def _build_transform(self, normal: np.ndarray, offset: float) -> np.ndarray:
        rotation = self._rotation_align_a_to_b(normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transformed_origin_height = float(rotation[2, :].dot(np.zeros((3,), dtype=np.float64)) + offset)
        transform[:3, 3] = np.asarray([0.0, 0.0, offset], dtype=np.float64)
        return transform

    def _locked_estimate(self, points: np.ndarray, preferred_up: np.ndarray) -> GroundPlaneEstimate | None:
        preferred_up = self._normalize(preferred_up)
        if preferred_up is None:
            return None
        signed = points @ preferred_up
        floor_height = float(np.quantile(signed, np.clip(self.floor_height_quantile, 0.0, 0.25)))
        offset = -floor_height
        distances = np.abs(signed - floor_height)
        inliers = distances <= self.distance_threshold
        num_inliers = int(np.count_nonzero(inliers))
        inlier_ratio = num_inliers / float(max(points.shape[0], 1))
        if inlier_ratio < self.min_inlier_ratio:
            return None
        candidates = [{
            "score": float(inlier_ratio + self.alignment_weight),
            "inlier_ratio": float(inlier_ratio),
            "alignment": 1.0,
            "offset": float(offset),
            "normal_world": preferred_up.astype(float).tolist(),
            "mode": "locked_manual_up",
        }]
        transform = self._build_transform(preferred_up, offset)
        return GroundPlaneEstimate(
            normal_world=preferred_up.astype(np.float32),
            offset=float(offset),
            inlier_ratio=float(inlier_ratio),
            num_points=int(points.shape[0]),
            num_inliers=int(num_inliers),
            transform_world_old_to_new=transform.astype(np.float32),
            chosen_score=float(candidates[0]["score"]),
            preferred_up=preferred_up.astype(np.float32),
            candidate_planes=candidates,
            lock_gravity_vector=True,
            manual_up_vector=None if self.manual_up_vector is None else self.manual_up_vector.astype(np.float32),
        )

    def estimate(self, geometry_artifact) -> GroundPlaneEstimate | None:
        if not self.enabled:
            return None
        points = self._sample_points(geometry_artifact)
        if points.shape[0] < 32:
            return None
        cameras = self._camera_centers(geometry_artifact)
        preferred_up = self._preferred_up_direction(geometry_artifact)
        if self.lock_gravity_vector and preferred_up is not None:
            return self._locked_estimate(points, preferred_up)
        rng = np.random.default_rng(self.random_seed)
        best: dict[str, Any] | None = None
        candidates: list[dict[str, Any]] = []
        n_points = points.shape[0]
        for _ in range(self.ransac_iterations):
            idx = rng.choice(n_points, size=3, replace=False)
            tri = points[idx]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal = normal / norm
            offset = -float(np.dot(normal, tri[0]))
            if preferred_up is not None:
                if float(np.dot(normal, preferred_up)) < 0.0:
                    normal = -normal
                    offset = -offset
            elif cameras.size:
                cam_signed = cameras @ normal + offset
                if np.median(cam_signed) < 0.0:
                    normal = -normal
                    offset = -offset
            signed = points @ normal + offset
            distances = np.abs(signed)
            inliers = distances <= self.distance_threshold
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < 3:
                continue
            inlier_ratio = inlier_count / float(n_points)
            low_height = float(np.percentile(signed, 8.0))
            low_penalty = abs(low_height)
            alignment = 1.0
            if preferred_up is not None:
                alignment = float(abs(np.dot(normal, preferred_up)))
                if alignment < self.min_alignment_cos:
                    continue
            score = inlier_ratio + self.alignment_weight * alignment - 0.04 * low_penalty
            candidate = {
                "score": float(score),
                "inlier_ratio": float(inlier_ratio),
                "alignment": float(alignment),
                "offset": float(offset),
                "low_height_penalty": float(low_penalty),
                "normal_world": normal.astype(float).tolist(),
            }
            candidates.append(candidate)
            if best is None or score > best["score"]:
                best = {"normal": normal, "offset": offset, "inliers": inliers, "score": score, "ratio": inlier_ratio}
        if best is None or float(best["ratio"]) < self.min_inlier_ratio:
            return None
        inlier_points = points[best["inliers"]]
        centroid = inlier_points.mean(axis=0)
        demeaned = inlier_points - centroid[None, :]
        _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
        normal = vh[-1]
        normal = normal / max(np.linalg.norm(normal), 1e-9)
        offset = -float(np.dot(normal, centroid))
        if preferred_up is not None:
            if float(np.dot(normal, preferred_up)) < 0.0:
                normal = -normal
                offset = -offset
        elif cameras.size:
            cam_signed = cameras @ normal + offset
            if np.median(cam_signed) < 0.0:
                normal = -normal
                offset = -offset
        transform = self._build_transform(normal, offset)
        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[: self.max_saved_candidates]
        return GroundPlaneEstimate(
            normal_world=normal.astype(np.float32),
            offset=float(offset),
            inlier_ratio=float(best["ratio"]),
            num_points=int(points.shape[0]),
            num_inliers=int(inlier_points.shape[0]),
            transform_world_old_to_new=transform.astype(np.float32),
            chosen_score=float(best["score"]),
            preferred_up=None if preferred_up is None else preferred_up.astype(np.float32),
            candidate_planes=candidates,
            lock_gravity_vector=False,
            manual_up_vector=None if self.manual_up_vector is None else self.manual_up_vector.astype(np.float32),
        )

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32)
        rot = np.asarray(transform[:3, :3], dtype=np.float32)
        trans = np.asarray(transform[:3, 3], dtype=np.float32)
        return np.einsum('ij,...j->...i', rot, arr) + trans

    @staticmethod
    def _transform_pose(T_world_cam: np.ndarray, transform: np.ndarray) -> np.ndarray:
        T = np.asarray(T_world_cam, dtype=np.float32)
        A = np.asarray(transform, dtype=np.float32)
        return A @ T

    def apply(self, geometry_artifact, estimate: GroundPlaneEstimate) -> None:
        transform = np.asarray(estimate.transform_world_old_to_new, dtype=np.float32)
        for frame_key, payload in list((geometry_artifact.pointmaps_by_frame or {}).items()):
            points = payload.get("points")
            if points is None:
                continue
            geometry_artifact.pointmaps_by_frame[frame_key] = {**payload, "points": self._transform_points(np.asarray(points, dtype=np.float32), transform)}
        for stream_id, states in list((geometry_artifact.camera_states_by_stream or {}).items()):
            new_states: list[TimedCameraState] = []
            for state in states:
                Twc = self._transform_pose(state.pose.T_world_cam, transform)
                new_states.append(TimedCameraState(stream_id=state.stream_id, frame_idx=int(state.frame_idx), time_sec=state.time_sec, intrinsics=state.intrinsics, pose=CameraPose(T_world_cam=Twc)))
            geometry_artifact.camera_states_by_stream[stream_id] = new_states
        geometry_artifact.metadata["world_frame"] = {"type": "ground_plane_z_up", **estimate.summary()}
        geometry_artifact.notes.append("Canonicalized world frame using a detected ground plane so Z points upward and the floor lies near z=0.")

    def estimate_and_apply(self, geometry_artifact) -> GroundPlaneEstimate | None:
        estimate = self.estimate(geometry_artifact)
        if estimate is None:
            return None
        self.apply(geometry_artifact, estimate)
        return estimate

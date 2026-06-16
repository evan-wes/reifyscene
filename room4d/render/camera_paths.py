"""Camera-path generation helpers for lightweight novel-view playback."""

from __future__ import annotations

from typing import Any

import numpy as np


def _z_up_rot() -> np.ndarray:
    """Return the canonical world rotation.

    The pipeline now canonicalizes geometry into a global Z-up frame before
    downstream stages run, so this helper returns identity for new artifacts.
    It remains as a named utility to preserve call sites and keep a single seam
    if legacy compatibility handling is reintroduced later.
    """
    return np.eye(3, dtype=np.float32)


def _to_z_up_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    rot = _z_up_rot()
    if arr.ndim >= 1 and arr.shape[-1] == 3:
        return np.einsum('ij,...j->...i', rot, arr)
    raise ValueError(f'Expected points with last dimension 3, got {arr.shape}.')


def _to_z_up_vec(vec: np.ndarray) -> np.ndarray:
    return _to_z_up_points(np.asarray(vec, dtype=np.float32)).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def orbit_cameras(
    center,
    radius: float,
    num_frames: int,
    *,
    up=(0.0, 0.0, 1.0),
    start_dir=None,
    arc_degrees: float = 90.0,
    height_frac: float = 0.2,
):
    """Arc of look-at cameras around ``center`` in the plane perpendicular to ``up``.

    The sweep is centered on ``start_dir`` (projected into that plane), so the
    path hugs the input trajectory rather than orbiting behind the scene where
    monocular 4D collapses. Returns a list of {"eye", "target", "up"} dicts.
    """
    center = np.asarray(center, dtype=np.float32)
    u = _unit(up)
    if start_dir is not None:
        start = np.asarray(start_dir, dtype=np.float32)
        a = _unit(start - float(np.dot(start, u)) * u)
    else:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        a = _unit(seed - float(np.dot(seed, u)) * u)
    b = _unit(np.cross(u, a))
    half = np.radians(float(arc_degrees)) / 2.0
    cameras = []
    for i in range(int(num_frames)):
        frac = 0.0 if num_frames <= 1 else i / (num_frames - 1)
        theta = -half + frac * 2.0 * half
        direction = np.cos(theta) * a + np.sin(theta) * b
        eye = center + float(radius) * direction + float(height_frac) * float(radius) * u
        cameras.append({"eye": eye.tolist(), "target": center.tolist(), "up": u.tolist()})
    return cameras


class CameraPathGenerator:
    """Generate canonical or query-driven camera trajectories."""

    def _time_grid(self, scene_artifact) -> list[dict[str, Any]]:
        return [
            {
                "frame_key": f"{fr.stream_id}:{int(fr.frame_idx)}",
                "time_sec": float(fr.timestamp_sec),
                "frame_ref": fr,
            }
            for fr in scene_artifact.frame_refs
        ]

    def orbit(self, scene_artifact, radius: float | None = None) -> dict[str, Any]:
        center = _to_z_up_vec(np.asarray(scene_artifact.static_scene.get("center_xyz", [0.0, 0.0, 0.0]), dtype=np.float32))
        times = self._time_grid(scene_artifact)
        static_points = _to_z_up_points(np.asarray(scene_artifact.static_scene.get("points_xyz", np.zeros((0, 3))), dtype=np.float32))
        if radius is None:
            if static_points.size:
                radius = float(np.percentile(np.linalg.norm(static_points - center[None, :], axis=1), 80))
            else:
                radius = 3.0
            radius = max(radius, 1.5)
        cameras = []
        for idx, item in enumerate(times):
            theta = 2.0 * np.pi * (idx / max(len(times), 1))
            eye = center + np.asarray([radius * np.cos(theta), radius * np.sin(theta), 0.35 * radius], dtype=np.float32)
            cameras.append({"eye": eye.tolist(), "target": center.tolist(), "up": [0.0, 0.0, 1.0], **item})
        return {"path_name": "orbit", "cameras": cameras}

    def overhead(self, scene_artifact, height_scale: float = 1.8) -> dict[str, Any]:
        center = _to_z_up_vec(np.asarray(scene_artifact.static_scene.get("center_xyz", [0.0, 0.0, 0.0]), dtype=np.float32))
        static_points = _to_z_up_points(np.asarray(scene_artifact.static_scene.get("points_xyz", np.zeros((0, 3))), dtype=np.float32))
        if static_points.size:
            spread = float(np.percentile(np.linalg.norm(static_points[:, :2] - center[None, :2], axis=1), 85))
        else:
            spread = 2.0
        height = max(2.0, spread * height_scale)
        cameras = []
        for item in self._time_grid(scene_artifact):
            eye = center + np.asarray([0.0, 0.0, height], dtype=np.float32)
            cameras.append({"eye": eye.tolist(), "target": center.tolist(), "up": [0.0, 1.0, 0.0], **item})
        return {"path_name": "overhead", "cameras": cameras}

    def _entity_path(self, scene_artifact, entity_type: str, entity_id: str, *, path_name: str) -> dict[str, Any]:
        entities = {ent["entity_id"]: ent for ent in scene_artifact.dynamic_entities.get(entity_type, [])}
        entity = entities.get(entity_id)
        if entity is None:
            return {"path_name": path_name, "cameras": []}
        times = self._time_grid(scene_artifact)
        entity_times = np.asarray(entity.get("times_sec", []), dtype=np.float32)
        centers = entity.get("centers_3d", [])
        cameras = []
        for item in times:
            time_sec = item["time_sec"]
            if entity_times.size == 0:
                target = _to_z_up_vec(np.asarray(entity.get("anchor_xyz", [0.0, 0.0, 0.0]), dtype=np.float32))
            else:
                idx = int(np.argmin(np.abs(entity_times - float(time_sec))))
                c = centers[idx] if idx < len(centers) else None
                target = _to_z_up_vec(np.asarray(c if c is not None else entity.get("anchor_xyz", [0.0, 0.0, 0.0]), dtype=np.float32))
            eye = target + np.asarray([-1.5, -1.5, 0.9], dtype=np.float32)
            cameras.append({"eye": eye.tolist(), "target": target.tolist(), "up": [0.0, 0.0, 1.0], **item})
        return {"path_name": path_name, "cameras": cameras, "entity_id": entity_id}

    def follow_subject(self, scene_artifact, node_id: str) -> dict[str, Any]:
        for kind in ("humans", "objects"):
            ids = {ent["entity_id"] for ent in scene_artifact.dynamic_entities.get(kind, [])}
            if node_id in ids:
                return self._entity_path(scene_artifact, kind, node_id, path_name=f"follow_{kind[:-1]}_{node_id}")
        return {"path_name": f"follow_{node_id}", "cameras": []}

    def replay(self, scene_artifact, optimization_artifact=None, geometry_artifact=None) -> dict[str, Any]:
        cameras = []
        state_map = {}
        if optimization_artifact is not None:
            for state in getattr(optimization_artifact, "refined_camera_states", []):
                state_map[(state.stream_id, float(state.time_sec))] = state
        if not state_map and geometry_artifact is not None:
            for stream_id, states in getattr(geometry_artifact, "camera_states_by_stream", {}).items():
                for state in states:
                    state_map[(stream_id, float(state.time_sec))] = state
            for state in getattr(geometry_artifact, "camera_states", []):
                state_map[(state.stream_id, float(state.time_sec))] = state
        for item in self._time_grid(scene_artifact):
            key = (item["frame_ref"].stream_id, float(item["time_sec"]))
            state = state_map.get(key)
            if state is None:
                continue
            Twc = np.asarray(state.pose.T_world_cam, dtype=np.float32)
            eye = _to_z_up_vec(Twc[:3, 3])
            forward = _to_z_up_vec(Twc[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
            up = _to_z_up_vec(Twc[:3, :3] @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
            target = eye + forward
            cameras.append({"eye": eye.tolist(), "target": target.tolist(), "up": up.tolist(), **item})
        return {"path_name": "replay", "cameras": cameras}

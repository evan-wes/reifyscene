"""Helpers for densifying camera trajectories onto downstream frame timelines."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np

from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState


def _copy_intrinsics(intr: CameraIntrinsics) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.cx),
        cy=float(intr.cy),
        distortion_model=str(intr.distortion_model),
        distortion_params=None if intr.distortion_params is None else np.asarray(intr.distortion_params, dtype=float).copy(),
    )


def _copy_state(state: TimedCameraState, *, frame: FrameRef | None = None) -> TimedCameraState:
    """A copy of ``state``, re-labelled as ``frame``'s when one is given."""
    return TimedCameraState(
        stream_id=str(state.stream_id),
        frame_idx=int(state.frame_idx if frame is None else frame.frame_idx),
        time_sec=state.time_sec if frame is None else frame.timestamp_sec,
        intrinsics=_copy_intrinsics(state.intrinsics),
        pose=CameraPose(T_world_cam=np.asarray(state.pose.T_world_cam, dtype=float).copy()),
    )


def _interp_scalar(a: float, b: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(a) + alpha * float(b))


def _interp_array(a: np.ndarray | None, b: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return np.asarray(b, dtype=float).copy()
    if b is None:
        return np.asarray(a, dtype=float).copy()
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        return arr_a.copy()
    return ((1.0 - alpha) * arr_a + alpha * arr_b).astype(float)


def _interp_intrinsics(a: CameraIntrinsics, b: CameraIntrinsics, alpha: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=_interp_scalar(a.fx, b.fx, alpha),
        fy=_interp_scalar(a.fy, b.fy, alpha),
        cx=_interp_scalar(a.cx, b.cx, alpha),
        cy=_interp_scalar(a.cy, b.cy, alpha),
        distortion_model=str(a.distortion_model),
        distortion_params=_interp_array(a.distortion_params, b.distortion_params, alpha),
    )


def _rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {r.shape}.")

    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        diag = np.diag(r)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    q = q / norm
    qw, qx, qy, qz = q
    return np.asarray([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    qa = np.asarray(q0, dtype=float)
    qb = np.asarray(q1, dtype=float)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)

    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    if dot > 0.9995:
        out = qa + alpha * (qb - qa)
        out /= max(float(np.linalg.norm(out)), 1e-12)
        return out

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if abs(sin_theta_0) < 1e-12:
        return qa.copy()
    theta = theta_0 * float(alpha)
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    out = s0 * qa + s1 * qb
    out /= max(float(np.linalg.norm(out)), 1e-12)
    return out


def _interp_pose(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    Ta = np.asarray(a, dtype=float)
    Tb = np.asarray(b, dtype=float)
    if Ta.shape != (4, 4) or Tb.shape != (4, 4):
        raise ValueError(f"Expected 4x4 poses, got {Ta.shape} and {Tb.shape}.")
    qa = _rotmat_to_quat(Ta[:3, :3])
    qb = _rotmat_to_quat(Tb[:3, :3])
    q = _slerp_quat(qa, qb, alpha)
    t = (1.0 - alpha) * Ta[:3, 3] + alpha * Tb[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = _quat_to_rotmat(q)
    out[:3, 3] = t
    return out


def _interpolate_state_pair(
    state_a: TimedCameraState,
    state_b: TimedCameraState,
    target: FrameRef,
) -> TimedCameraState:
    fa, fb = int(state_a.frame_idx), int(state_b.frame_idx)
    if fa == fb:
        return _copy_state(state_a, frame=target)
    alpha = float((int(target.frame_idx) - fa) / (fb - fa))
    intr = _interp_intrinsics(state_a.intrinsics, state_b.intrinsics, alpha)
    pose = _interp_pose(state_a.pose.T_world_cam, state_b.pose.T_world_cam, alpha)
    return TimedCameraState(
        stream_id=str(state_a.stream_id),
        frame_idx=int(target.frame_idx),
        time_sec=target.timestamp_sec,
        intrinsics=intr,
        pose=CameraPose(T_world_cam=pose.astype(float)),
    )


def _state_for_target_frame(states: list[TimedCameraState], target: FrameRef) -> tuple[TimedCameraState | None, str]:
    """``states`` sorted by frame number."""
    if not states:
        return None, "missing"
    numbers = [int(state.frame_idx) for state in states]
    idx = bisect_left(numbers, int(target.frame_idx))
    if idx < len(states) and numbers[idx] == int(target.frame_idx):
        return states[idx], "exact"
    if len(states) == 1:
        return _copy_state(states[0], frame=target), "copied_single"
    if idx <= 0:
        return _interpolate_state_pair(states[0], states[1], target), "extrapolated_before"
    if idx >= len(states):
        return _interpolate_state_pair(states[-2], states[-1], target), "extrapolated_after"
    return _interpolate_state_pair(states[idx - 1], states[idx], target), "interpolated"


def densify_camera_states_for_frames(
    camera_states_by_stream: dict[str, list[TimedCameraState]],
    target_frames: list[FrameRef],
) -> dict[str, Any]:
    """Mutate ``camera_states_by_stream`` so it covers every target frame.

    Existing states are preserved. A missing frame is filled by pose interpolation between
    the nearest states of the same stream (by frame number), with boundary extrapolation
    when the frame lies just outside the geometry's span.
    """
    targets_by_stream: dict[str, list[FrameRef]] = {}
    for frame in target_frames:
        targets_by_stream.setdefault(str(frame.stream_id), []).append(frame)

    summary: dict[str, Any] = {
        "target_frame_count": int(len(target_frames)),
        "streams": {},
        "added_camera_states": 0,
    }

    for stream_id, frames in sorted(targets_by_stream.items()):
        existing_states = list(camera_states_by_stream.get(stream_id, []))
        stream_summary = {
            "target_frames": int(len(frames)),
            "exact": 0,
            "interpolated": 0,
            "extrapolated_before": 0,
            "extrapolated_after": 0,
            "copied_single": 0,
            "missing": 0,
            "added": 0,
            "camera_states_after": int(len(existing_states)),
        }
        if not existing_states:
            summary["streams"][stream_id] = stream_summary
            continue

        augmented = sorted((_copy_state(state) for state in existing_states),
                           key=lambda state: int(state.frame_idx))
        for frame in sorted(frames, key=lambda fr: int(fr.frame_idx)):
            state, mode = _state_for_target_frame(augmented, frame)
            stream_summary[mode] = int(stream_summary.get(mode, 0)) + 1
            if state is None:
                continue
            if mode != "exact":
                augmented.append(state)
                augmented.sort(key=lambda item: int(item.frame_idx))
                stream_summary["added"] = int(stream_summary["added"]) + 1

        deduped: list[TimedCameraState] = []
        for state in augmented:
            if deduped and int(deduped[-1].frame_idx) == int(state.frame_idx):
                continue
            deduped.append(state)
        camera_states_by_stream[stream_id] = deduped
        stream_summary["camera_states_after"] = int(len(deduped))
        summary["streams"][stream_id] = stream_summary
        summary["added_camera_states"] = int(summary["added_camera_states"]) + int(stream_summary["added"])

    return summary

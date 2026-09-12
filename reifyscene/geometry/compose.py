"""Fuse a pose-channel + depth-channel geometry into one artifact (FP6b step 2, §4A.1).

The geometry stage is **two fused ladders** (design §4A.1): a *pose* channel (VGGT joint →
COLMAP-SfM → dataset/GT → …) and a *depth* channel (VGGT joint → MoGe-2 → …). When the top
joint rung (VGGT) is unavailable the channels decompose — COLMAP gives poses-without-depth,
MoGe gives depth-without-poses — and this composer stitches the two decomposed artifacts back
into a single, *complete* :class:`GeometryArtifact`.

Two things the design (§4A.1) calls out are handled here:

1. **Metric scale.** COLMAP is scale-free, MoGe is metric. :class:`ScaleReconciler` solves the
   single global scale first (gauge-free, camera-space), and the composer applies it to the
   COLMAP camera centers so poses and depth share metric units.
2. **Per-frame intrinsics.** MoGe estimates its *own* per-frame intrinsics; its camera-space
   point map is self-consistent with them. We never re-project MoGe depth through the pose
   source's focal — instead we **rigidly** transform MoGe's own 3D points by the (reconciled)
   pose, ``X_world = R_wc · X_cam^moge + s·C``. A pose is SE(3), focal-agnostic, so this is
   exact. The resulting ``pointmaps_by_frame`` are **world-space**, matching VGGT's convention
   so the ground-plane canonicalizer and every downstream consumer work unchanged.

Graceful degradation (design §4A invariant 2/3): if a channel came back empty, the composite
falls back to depth-only / pose-only / empty rather than crashing — completeness is preserved,
only fidelity drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.types import CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.scale_reconcile import (
    ScaleReconcileResult,
    extract_colmap_observations,
    solve_global_scale,
)


def _frame_key(frame: FrameRef) -> str:
    return f"{frame.stream_id}:{frame.frame_idx}"


@dataclass
class ComposeResult:
    """What the composer produced: the fused artifact + how it resolved."""

    artifact: GeometryArtifact
    fidelity: str  # 'fused_metric' | 'poses_only' | 'depth_only' | 'empty'
    reconcile: ScaleReconcileResult | None = None


class GeometryComposer:
    """Compose pose-channel + depth-channel geometry, reconciling metric scale."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        # min sparse-point depth samples required to trust the scale solve
        self.min_obs = int(self.cfg.get("scale_min_obs", 32))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _pose_by_key(pose_artifact: GeometryArtifact) -> dict[str, TimedCameraState]:
        """Map ``frame_key -> TimedCameraState`` from a pose artifact.

        The pose rung builds one state per registered ``FrameRef`` with
        ``time_sec = frame.timestamp_sec``; match each selected frame back to its state
        by (stream, exact time), so the composer can key poses by frame like depth is.
        """
        out: dict[str, TimedCameraState] = {}
        for frame in pose_artifact.selected_frames or []:
            states = pose_artifact.camera_states_by_stream.get(frame.stream_id, [])
            match = None
            for state in states:
                if abs(float(state.time_sec) - float(frame.timestamp_sec)) < 1e-9:
                    match = state
                    break
            if match is None and states:
                match = min(states, key=lambda s: abs(float(s.time_sec) - float(frame.timestamp_sec)))
            if match is not None:
                out[_frame_key(frame)] = match
        return out

    @staticmethod
    def _points_cam_to_world(points_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
        """Rigidly map camera-space points ``HxWx3`` into world via ``T_world_cam``."""
        pts = np.asarray(points_cam, dtype=np.float64)
        h, w = pts.shape[:2]
        flat = pts.reshape(-1, 3)
        r = np.asarray(T_world_cam, dtype=np.float64)[:3, :3]
        c = np.asarray(T_world_cam, dtype=np.float64)[:3, 3]
        world = flat @ r.T + c  # (R · x)^T = x^T · R^T; broadcast + C
        return world.reshape(h, w, 3).astype(np.float32)

    @staticmethod
    def _scaled_pose(T_world_cam: np.ndarray, scale: float) -> np.ndarray:
        """Scale a camera center by ``scale`` (metric), rotation unchanged."""
        out = np.asarray(T_world_cam, dtype=np.float64).copy()
        out[:3, 3] *= float(scale)
        return out.astype(np.float32)

    # -- composition -------------------------------------------------------

    def compose(
        self,
        *,
        pose_artifact: GeometryArtifact,
        depth_artifact: GeometryArtifact,
        shared_frames: list[FrameRef],
        pose_backend: str,
        depth_backend: str,
        run_id: str,
        save_dir: Path,
    ) -> ComposeResult:
        """Fuse the two channels into one metric ``GeometryArtifact``.

        ``shared_frames`` is the identical frame list both channels ran on; its position
        indices are how the COLMAP rung named its images, so they map sparse observations
        back to depth frames for reconciliation.
        """
        pose_by_key = self._pose_by_key(pose_artifact)
        depth_keys = set((depth_artifact.depthmaps_by_frame or {}).keys())
        pointmap_keys = set((depth_artifact.pointmaps_by_frame or {}).keys())
        have_poses = bool(pose_by_key)
        have_depth = bool(depth_keys)

        backend_name = f"{pose_backend}+{depth_backend}"

        # ---- degradation: a channel came back empty --------------------------
        if not have_poses and not have_depth:
            return ComposeResult(
                artifact=GeometryArtifact(
                    run_id=run_id, stage_name="geometry", save_dir=save_dir,
                    backend_name="empty", selected_frames=[],
                    notes=[f"fused {backend_name}: both channels empty → empty geometry."],
                ),
                fidelity="empty",
            )
        if not have_poses:
            depth_artifact.backend_name = backend_name
            depth_artifact.metadata.setdefault("fused", {})
            depth_artifact.metadata["fused"] = {"pose_backend": pose_backend, "depth_backend": depth_backend, "resolved": "depth_only"}
            depth_artifact.notes.append(f"fused {backend_name}: pose channel empty → depth-only (camera-space pointmaps, no world frame / scale).")
            return ComposeResult(artifact=depth_artifact, fidelity="depth_only")
        if not have_depth:
            pose_artifact.backend_name = backend_name
            pose_artifact.metadata["fused"] = {"pose_backend": pose_backend, "depth_backend": depth_backend, "resolved": "poses_only"}
            pose_artifact.notes.append(f"fused {backend_name}: depth channel empty → poses-only (no metric depth to reconcile).")
            return ComposeResult(artifact=pose_artifact, fidelity="poses_only")

        # ---- full fusion -----------------------------------------------------
        # Reconcile scale FIRST (gauge-free, camera-space) so ground-plane later sees a
        # scale-consistent reconstruction (design §4A.1: scale is the one DOF it doesn't pin).
        index_to_key = {i: _frame_key(f) for i, f in enumerate(shared_frames)}
        both_keys = [k for k in index_to_key.values() if k in pose_by_key and k in depth_keys]
        depth_by_key = {k: depth_artifact.depthmaps_by_frame[k]["depth"] for k in both_keys}
        mask_by_key = {
            k: depth_artifact.depthmaps_by_frame[k].get("confidence")
            for k in both_keys
            if depth_artifact.depthmaps_by_frame[k].get("confidence") is not None
        }
        model_dir = pose_artifact.metadata.get("colmap_model_dir")
        obs_by_key: dict[str, list[tuple[float, float, float]]] = {}
        if model_dir:
            obs_by_key = extract_colmap_observations(model_dir, index_to_key)
        reconcile = solve_global_scale(
            obs_by_key, depth_by_key, min_obs=self.min_obs, mask_by_key=mask_by_key or None,
        )
        scale = reconcile.scale

        # Build the fused artifact over frames that have BOTH pose and depth.
        by_key_frame = {_frame_key(f): f for f in shared_frames}
        selected: list[FrameRef] = [by_key_frame[k] for k in both_keys if k in pointmap_keys]
        intrinsics_by_stream = dict(pose_artifact.intrinsics_by_stream)
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}
        for frame in selected:
            key = _frame_key(frame)
            state = pose_by_key[key]
            T_metric = self._scaled_pose(state.pose.T_world_cam, scale)
            camera_states_by_stream.setdefault(frame.stream_id, []).append(
                TimedCameraState(
                    stream_id=frame.stream_id,
                    time_sec=float(frame.timestamp_sec),
                    intrinsics=state.intrinsics,  # pose-source (COLMAP) camera model
                    pose=CameraPose(T_world_cam=T_metric),
                )
            )
            depth_payload = depth_artifact.depthmaps_by_frame[key]
            depthmaps_by_frame[key] = depth_payload
            pm = depth_artifact.pointmaps_by_frame[key]
            world_pts = self._points_cam_to_world(pm["points"], T_metric)
            pointmaps_by_frame[key] = {"points": world_pts, "confidence": pm.get("confidence")}
        for states in camera_states_by_stream.values():
            states.sort(key=lambda s: (s.time_sec, s.stream_id))

        artifact = GeometryArtifact(
            run_id=run_id, stage_name="geometry", save_dir=save_dir,
            backend_name=backend_name, selected_frames=selected,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            depthmaps_by_frame=depthmaps_by_frame,
            pointmaps_by_frame=pointmaps_by_frame,
            notes=[
                f"fused geometry: poses={pose_backend} + depth={depth_backend} over "
                f"{len(selected)} frames with both channels.",
                reconcile.note(),
                "MoGe camera-space points rigidly transformed into world by the reconciled "
                "metric poses (per-frame MoGe intrinsics kept self-consistent; §4A.1).",
            ],
        )
        artifact.metadata["fused"] = {
            "pose_backend": pose_backend, "depth_backend": depth_backend, "resolved": "fused_metric",
        }
        artifact.metadata["scale_reconcile"] = reconcile.summary()
        artifact.metadata["scale_reconcile_per_frame"] = reconcile.per_frame_residual
        # FP6b step 3: carry the pose rung's GT-free health-check into the fused artifact and
        # downgrade fidelity to 'fused_metric_suspect' if the trajectory looks degenerate, so
        # a silent bad SfM draw is observable (never silently fused as clean).
        pose_health = pose_artifact.metadata.get("pose_health")
        fidelity = "fused_metric"
        if pose_health is not None:
            artifact.metadata["pose_health"] = pose_health
            if not pose_health.get("healthy", True):
                fidelity = "fused_metric_suspect"
                artifact.notes.append(f"⚠ pose health SUSPECT (degenerate SfM draw?) — {pose_health.get('reason', '')}")
        # provenance: MoGe's own intrinsics (used to build its camera-space points), kept
        # distinct from the pose-source intrinsics the camera_states carry.
        if depth_artifact.intrinsics_by_stream:
            artifact.metadata["moge_intrinsics_by_stream"] = {
                sid: {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
                for sid, intr in depth_artifact.intrinsics_by_stream.items()
            }
        return ComposeResult(artifact=artifact, fidelity=fidelity, reconcile=reconcile)

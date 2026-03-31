"""Backprojection and 3D track lifting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Backprojector:
    """Lift pixels to 3D using camera and depth estimates."""

    point_conf_threshold: float = 0.0

    def backproject(self, pixels: np.ndarray, camera_state, depth: np.ndarray) -> np.ndarray:
        """Backproject model-resolution pixels to world coordinates.

        Parameters
        ----------
        pixels:
            Array shaped ``(N, 2)`` containing ``(x, y)`` pixel coordinates in
            the same resolution as ``depth`` and the camera intrinsics.
        camera_state:
            Timed camera state whose pose is interpreted as ``world_from_cam``.
        depth:
            Depth map shaped ``(H, W)`` or ``(H, W, 1)``.
        """
        depth_map = np.asarray(depth, dtype=np.float32)
        if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        h, w = depth_map.shape[:2]
        xy = np.asarray(pixels, dtype=np.float32)
        u = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        z = depth_map[v, u]
        intr = camera_state.intrinsics
        x_cam = (xy[:, 0] - float(intr.cx)) / max(float(intr.fx), 1e-6) * z
        y_cam = (xy[:, 1] - float(intr.cy)) / max(float(intr.fy), 1e-6) * z
        xyz_cam = np.stack([x_cam, y_cam, z], axis=-1)
        T_world_cam = np.asarray(camera_state.pose.T_world_cam, dtype=np.float32)
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        xyz_world = xyz_cam @ R.T + t[None, :]
        invalid = ~np.isfinite(z)
        xyz_world[invalid] = np.nan
        return xyz_world.astype(np.float32)


class TrackLifter3D:
    """Lift sparse 2D tracks into approximate world-space samples."""

    def __init__(self, point_conf_threshold: float = 0.0) -> None:
        self.point_conf_threshold = float(point_conf_threshold)
        self.backprojector = Backprojector(point_conf_threshold=point_conf_threshold)

    @staticmethod
    def _stream_lookup(ingest_artifact) -> dict[str, Any]:
        return {stream.stream_id: stream for stream in getattr(ingest_artifact, 'streams', [])}

    @staticmethod
    def _find_camera_state(geometry_artifact, frame) -> Any | None:
        states = geometry_artifact.camera_states_by_stream.get(frame.stream_id, [])
        if not states:
            return None
        best = None
        best_dt = None
        for state in states:
            dt = abs(float(state.time_sec) - float(frame.timestamp_sec))
            if best is None or dt < best_dt:
                best = state
                best_dt = dt
        return best

    def _sample_pointmap(self, xy: np.ndarray, point_payload: dict[str, Any], *, original_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(point_payload['points'], dtype=np.float32)
        conf = point_payload.get('confidence')
        if points.ndim == 3 and points.shape[0] == 3:
            points = np.moveaxis(points, 0, -1)
        h, w, _ = points.shape
        scale_x = float(w) / max(float(original_wh[0]), 1e-6)
        scale_y = float(h) / max(float(original_wh[1]), 1e-6)
        x = np.clip(np.round(np.asarray(xy[:, 0], dtype=np.float32) * scale_x).astype(int), 0, w - 1)
        y = np.clip(np.round(np.asarray(xy[:, 1], dtype=np.float32) * scale_y).astype(int), 0, h - 1)
        xyz = points[y, x]
        valid = np.isfinite(xyz).all(axis=1)
        if conf is not None:
            conf_arr = np.asarray(conf, dtype=np.float32)
            if conf_arr.ndim == 3 and conf_arr.shape[-1] == 1:
                conf_arr = conf_arr[..., 0]
            sampled_conf = conf_arr[y, x]
            valid &= np.isfinite(sampled_conf)
            valid &= sampled_conf > self.point_conf_threshold
        xyz = xyz.astype(np.float32)
        xyz[~valid] = np.nan
        return xyz, valid

    def lift(self, track_artifact, geometry_artifact, ingest_artifact) -> Any:
        """Attach lifted 3D point samples to a track artifact.

        Only the geometry keyframes are lifted in this first slice. Dense track
        observations are preserved unchanged, but the lifted representation gives
        downstream stages a world-space handle on each track.
        """
        stream_lookup = self._stream_lookup(ingest_artifact)
        geometry_selected_frames = list(geometry_artifact.selected_frames)
        # Preserve the tracking timeline already chosen upstream. Scene/render
        # uses TrackArtifact.selected_frames when timeline_source=tracking.
        # Lifting itself still operates on the geometry keyframes below.
        if not getattr(track_artifact, 'selected_frames', None):
            track_artifact.selected_frames = list(geometry_selected_frames)
        track_artifact.metadata['geometry_selected_frames'] = [
            {'stream_id': frame.stream_id, 'frame_idx': int(frame.frame_idx), 'timestamp_sec': float(frame.timestamp_sec)}
            for frame in geometry_selected_frames
        ]
        track_artifact.depthmaps_by_frame = dict(geometry_artifact.depthmaps_by_frame)
        track_artifact.pointmaps_by_frame = dict(geometry_artifact.pointmaps_by_frame)
        track_artifact.metadata.setdefault('debug_image_paths', dict(geometry_artifact.metadata.get('debug_image_paths', {})))

        for stream_id, payload in track_artifact.track_payloads_by_stream.items():
            frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
            points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
            visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
            num_tracks = points_xy.shape[1] if points_xy.ndim == 3 else 0
            stream_frames = [f for f in geometry_selected_frames if f.stream_id == stream_id]
            lifted_xyz = np.full((len(stream_frames), num_tracks, 3), np.nan, dtype=np.float32)
            lifted_valid = np.zeros((len(stream_frames), num_tracks), dtype=bool)
            lifted_frame_indices = np.asarray([f.frame_idx for f in stream_frames], dtype=np.int32)
            stream = stream_lookup.get(stream_id)
            original_wh = tuple(int(v) for v in payload.get('original_size_wh', [stream.width if stream else 0, stream.height if stream else 0]))

            for k, frame in enumerate(stream_frames):
                matches = np.where(frame_indices == int(frame.frame_idx))[0]
                if matches.size == 0:
                    continue
                t = int(matches[0])
                xy = points_xy[t]
                vis = visibility[t].astype(bool) if visibility.size else np.ones((num_tracks,), dtype=bool)
                frame_key = f'{frame.stream_id}:{frame.frame_idx}'
                point_payload = geometry_artifact.pointmaps_by_frame.get(frame_key)
                if point_payload is not None:
                    xyz, valid = self._sample_pointmap(xy, point_payload, original_wh=original_wh)
                else:
                    depth_payload = geometry_artifact.depthmaps_by_frame.get(frame_key)
                    camera_state = self._find_camera_state(geometry_artifact, frame)
                    if depth_payload is None or camera_state is None:
                        continue
                    depth = depth_payload['depth']
                    depth_map = np.asarray(depth)
                    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
                        depth_map = depth_map[..., 0]
                    h, w = depth_map.shape[:2]
                    scale_x = float(w) / max(float(original_wh[0]), 1e-6)
                    scale_y = float(h) / max(float(original_wh[1]), 1e-6)
                    xy_model = np.asarray(xy, dtype=np.float32).copy()
                    xy_model[:, 0] *= scale_x
                    xy_model[:, 1] *= scale_y
                    xyz = self.backprojector.backproject(xy_model, camera_state, depth_map)
                    valid = np.isfinite(xyz).all(axis=1)
                valid &= vis
                xyz[~valid] = np.nan
                lifted_xyz[k] = xyz
                lifted_valid[k] = valid

            payload['lifted_frame_indices'] = lifted_frame_indices
            payload['lifted_xyz'] = lifted_xyz
            payload['lifted_valid'] = lifted_valid
        track_artifact.notes.append(
            f'Lifted tracks onto {len(geometry_selected_frames)} geometry keyframes using VGGT point/depth outputs.'
        )
        return track_artifact

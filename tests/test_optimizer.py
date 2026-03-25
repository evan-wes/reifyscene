"""Basic tests for optimizer scaffolding and static-track refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState, TrackObservation
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.optimize.factors.reprojection import ReprojectionFactor
from room4d.optimize.optimization_artifact import OptimizationArtifact
from room4d.optimize.solver import JointSpatiotemporalSolver
from room4d.sync.sync_artifact import SyncArtifact
from room4d.tracking.track_artifact import TrackArtifact


def test_reprojection_factor_placeholder() -> None:
    residual, diag = ReprojectionFactor().residual({})
    assert residual == 0.0
    assert isinstance(diag, dict)


def _make_geometry_for_refinement(tmp_path: Path) -> GeometryArtifact:
    stream_id = 'cam_00'
    frames = [
        FrameRef(stream_id=stream_id, frame_idx=0, timestamp_sec=0.0),
        FrameRef(stream_id=stream_id, frame_idx=1, timestamp_sec=0.1),
        FrameRef(stream_id=stream_id, frame_idx=2, timestamp_sec=0.2),
    ]
    intr = CameraIntrinsics(fx=40.0, fy=40.0, cx=32.0, cy=24.0)
    states = [
        TimedCameraState(stream_id=stream_id, time_sec=0.0, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, time_sec=0.1, intrinsics=intr, pose=CameraPose(T_world_cam=np.asarray([
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, time_sec=0.2, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
    ]
    return GeometryArtifact(
        run_id='test',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry',
        selected_frames=frames,
        intrinsics_by_stream={stream_id: intr},
        camera_states_by_stream={stream_id: states},
    )


def _make_track_artifact(tmp_path: Path) -> TrackArtifact:
    stream_id = 'cam_00'
    track_ids = np.arange(4, dtype=np.int32)
    refs = np.asarray([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.0, 0.1, 1.0],
        [0.1, 0.1, 1.0],
    ], dtype=np.float32)
    lifted_xyz = np.stack([
        refs,
        refs + np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
        refs,
    ], axis=0)
    lifted_valid = np.ones((3, 4), dtype=bool)
    payload = {
        'frame_indices': np.asarray([0, 1, 2], dtype=np.int32),
        'timestamps_sec': np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        'track_ids': track_ids,
        'points_xy': np.zeros((3, 4, 2), dtype=np.float32),
        'visibility': np.ones((3, 4), dtype=bool),
        'confidence': np.ones((3, 4), dtype=np.float32),
        'backend_used': 'test',
        'windowing': {'windowed': False},
        'lifted_frame_indices': np.asarray([0, 1, 2], dtype=np.int32),
        'lifted_xyz': lifted_xyz,
        'lifted_valid': lifted_valid,
        'static_labels': np.ones((4,), dtype=np.int8),
        'static_scores': np.zeros((4,), dtype=np.float32),
    }
    obs = [
        TrackObservation(stream_id=stream_id, frame_idx=fidx, track_id=int(tid), xy=np.zeros((2,), dtype=np.float32), visible=True, confidence=1.0)
        for fidx in [0, 1, 2] for tid in track_ids.tolist()
    ]
    return TrackArtifact(
        run_id='test',
        stage_name='tracking',
        save_dir=tmp_path / 'tracking',
        per_stream_observations={stream_id: obs},
        track_payloads_by_stream={stream_id: payload},
    )


def test_static_track_refinement_reduces_camera_error(tmp_path: Path) -> None:
    geometry = _make_geometry_for_refinement(tmp_path)
    tracks = _make_track_artifact(tmp_path)
    sync = SyncArtifact(run_id='test', stage_name='sync', save_dir=tmp_path / 'sync', offsets_by_stream={'cam_00': 0.0})

    solver = JointSpatiotemporalSolver({
        'enable_static_track_refinement': True,
        'static_track_refine_iterations': 2,
        'static_track_max_translation': 0.5,
        'static_track_max_rotation_deg': 5.0,
    })
    artifact = solver.solve_all(sync, geometry, tracks)

    stage1 = artifact.stage_diagnostics[0]
    refinement = stage1['static_track_refinement']
    stream_info = refinement['streams']['cam_00']
    assert stream_info['applied'] is True
    assert stream_info['accepted_updates'] >= 1
    assert stream_info['mean_error_after'] < stream_info['mean_error_before']

    refined_states = artifact.refined_cameras['cam_00']
    # The middle frame should be pulled back toward x ~= 0.
    assert abs(float(refined_states[1].pose.T_world_cam[0, 3])) < 0.05

    artifact.save_dir = tmp_path / 'optimization'
    artifact.save()
    reloaded = OptimizationArtifact.load(tmp_path / 'optimization')
    assert 'cam_00' in reloaded.refined_cameras
    assert reloaded.tracks_3d['streams']['cam_00']['applied'] is True

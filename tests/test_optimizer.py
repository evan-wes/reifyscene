"""Basic tests for optimizer scaffolding and static-track refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.common.data_dir import record_name
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState, TrackObservation
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.optimize.factors.reprojection import ReprojectionFactor
from reifyscene.optimize.optimization_artifact import OptimizationArtifact
from reifyscene.optimize.solver import JointSpatiotemporalSolver
from reifyscene.sync.sync_artifact import SyncArtifact
from reifyscene.tracking.track_artifact import TrackArtifact
from reifyscene.geometry.image_transform import ImageTransform, transform_record


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
        TimedCameraState(stream_id=stream_id, frame_idx=0, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, frame_idx=1, intrinsics=intr, pose=CameraPose(T_world_cam=np.asarray([
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, frame_idx=2, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
    ]
    depthmaps = {}
    pointmaps = {}
    image_paths = {}
    for frame in frames:
        frame_key = f'{frame.stream_id}:{frame.frame_idx}'
        depth = np.ones((48, 64), dtype=np.float32)
        xs, ys = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(48, dtype=np.float32))
        pts = np.stack([xs / 40.0, ys / 40.0, np.ones_like(xs)], axis=-1).astype(np.float32)
        depthmaps[frame_key] = {'depth': depth, 'confidence': np.ones_like(depth)}
        pointmaps[frame_key] = {'points': pts, 'confidence': np.ones((48, 64), dtype=np.float32)}
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[..., 1] = 120
        img_path = tmp_path / f'{frame_key.replace(":", "__")}.png'
        from PIL import Image
        Image.fromarray(img).save(img_path)
        image_paths[frame_key] = record_name(img_path)
    return GeometryArtifact(
        run_id='test',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry',
        selected_frames=frames,
        intrinsics_by_stream={stream_id: intr},
        camera_states_by_stream={stream_id: states},
        depthmaps_by_frame=depthmaps,
        pointmaps_by_frame=pointmaps,
        metadata={'debug_image_paths': image_paths, 'image_transforms': {'cam_00': transform_record(
            ImageTransform.identity((64, 48)), recorded_by='test')}},
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
        'track_ids': track_ids,
        'points_xy': np.zeros((3, 4, 2), dtype=np.float32),
        # the frame the track positions are measured in (the tracker always records it);
        # here the geometry's own 64x48 pixels
        'original_size_wh': [64, 48],
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


def test_refinement_viewer_and_relabel_outputs(tmp_path: Path) -> None:
    geometry = _make_geometry_for_refinement(tmp_path)
    tracks = _make_track_artifact(tmp_path)
    sync = SyncArtifact(run_id='test', stage_name='sync', save_dir=tmp_path / 'sync', offsets_by_stream={'cam_00': 0.0})
    solver = JointSpatiotemporalSolver({'enable_static_track_refinement': True, 'static_track_max_translation': 0.5})
    artifact = solver.solve_all(sync, geometry, tracks)
    artifact.save_dir = tmp_path / 'optimization'
    artifact.metadata.update({'save_refinement_viewer': True, 'debug_max_points': 1000, 'debug_max_trails': 4, 'debug_trail_window': 4})
    artifact.export_refinement_viewer(tracks, geometry)
    artifact.save()
    payload = artifact.refined_track_payloads_by_stream['cam_00']
    assert 'refined_lifted_xyz' in payload
    assert payload['refined_lifted_xyz'].shape == payload['lifted_xyz'].shape
    assert 'refined_static_labels' in payload
    assert (artifact.save_dir / 'refinement_debug_viewer.html').exists()

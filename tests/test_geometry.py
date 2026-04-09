"""Basic tests for geometry scaffolding and world-frame canonicalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.geometry.world_frame import GroundPlaneCanonicalizer


def test_vggt_initializer_constructs() -> None:
    init = VGGTInitializer()
    assert init.cfg == {}


def test_ground_plane_detection_canonicalizes_to_z_up(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=1, timestamp_sec=0.1)
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 24), np.linspace(-1.0, 1.0, 24), indexing='xy')
    base_z = 0.4 * xs + 0.05 * ys + 0.3
    pts = np.stack([xs, ys, base_z], axis=2).astype(np.float32)
    pts1 = pts + np.array([0.0, 0.0, 0.02], dtype=np.float32)

    geometry = GeometryArtifact(
        run_id='geom',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry',
        selected_frames=[frame0, frame1],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 12, 12)},
        camera_states_by_stream={
            'cam_00': [
                TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 12, 12), CameraPose(np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float32))),
                TimedCameraState('cam_00', 0.1, CameraIntrinsics(500, 500, 12, 12), CameraPose(np.array([
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.55],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float32))),
            ]
        },
        pointmaps_by_frame={
            'cam_00:0': {'points': pts, 'confidence': np.ones((24, 24), dtype=np.float32)},
            'cam_00:1': {'points': pts1, 'confidence': np.ones((24, 24), dtype=np.float32)},
        },
        depthmaps_by_frame={},
        metadata={},
        notes=[],
    )

    canonicalizer = GroundPlaneCanonicalizer({'ransac_iterations': 96, 'distance_threshold': 0.05})
    estimate = canonicalizer.estimate_and_apply(geometry)
    assert estimate is not None
    assert geometry.metadata['world_frame']['type'] == 'ground_plane_z_up'

    transformed = np.asarray(geometry.pointmaps_by_frame['cam_00:0']['points'], dtype=np.float32)
    plane = transformed.reshape(-1, 3)
    assert np.quantile(np.abs(plane[:, 2]), 0.9) < 0.08

    cam_zs = [state.pose.T_world_cam[2, 3] for state in geometry.camera_states_by_stream['cam_00']]
    assert min(cam_zs) > 1.0
    assert any('Z points upward' in note for note in geometry.notes)


def test_geometry_sample_point_cloud_callable_from_class() -> None:
    points = np.zeros((4, 5, 3), dtype=np.float32)
    points[..., 2] = 1.0
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    pts, cols = GeometryArtifact._sample_point_cloud(points, rgb=rgb, max_points=16)
    assert pts.shape[1] == 3
    assert cols is not None
    assert cols.shape[1] == 3


"""Basic tests for geometry scaffolding and world-frame canonicalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.vggt_wrapper import VGGTInitializer
from reifyscene.geometry.world_frame import GroundPlaneCanonicalizer


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
                TimedCameraState('cam_00', 0, CameraIntrinsics(500, 500, 12, 12), CameraPose(np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float32))),
                TimedCameraState('cam_00', 1, CameraIntrinsics(500, 500, 12, 12), CameraPose(np.array([
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

    canonicalizer = GroundPlaneCanonicalizer({'ransac_iterations': 96, 'distance_threshold': 0.05, 'camera_up_axis': 'z'})
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




def test_geometry_debug_exports_world_frame_diagnostics(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    rgb_path = tmp_path / 'frame0.png'
    import numpy as np
    from PIL import Image
    Image.fromarray(np.full((24, 24, 3), 180, dtype=np.uint8)).save(rgb_path)
    pts = np.zeros((24, 24, 3), dtype=np.float32)
    pts[..., 0], pts[..., 1] = np.meshgrid(np.linspace(-1.0, 1.0, 24), np.linspace(-1.0, 1.0, 24), indexing='xy')
    pts[..., 2] = 0.1 * pts[..., 0] + 0.2
    geometry = GeometryArtifact(
        run_id='geom_diag',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry_diag',
        selected_frames=[frame0],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(200, 200, 12, 12)},
        camera_states_by_stream={'cam_00': [TimedCameraState('cam_00', 0, CameraIntrinsics(200, 200, 12, 12), CameraPose(np.eye(4, dtype=np.float32)))]},
        pointmaps_by_frame={'cam_00:0': {'points': pts, 'confidence': np.ones((24, 24), dtype=np.float32)}},
        depthmaps_by_frame={},
        metadata={'debug_image_paths': {'cam_00:0': str(rgb_path)}, 'save_debug_assets': True},
        notes=[],
    )
    canonicalizer = GroundPlaneCanonicalizer({'camera_up_axis': 'z', 'ransac_iterations': 64})
    estimate = canonicalizer.estimate_and_apply(geometry)
    assert estimate is not None
    geometry.save()
    assert (tmp_path / 'geometry_diag' / 'ground_plane_diagnostics.json').exists()
    overlay_dir = tmp_path / 'geometry_diag' / 'world_frame_overlays'
    assert overlay_dir.exists()
    assert any(p.suffix == '.png' for p in overlay_dir.iterdir())

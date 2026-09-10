from __future__ import annotations

import json

import numpy as np
from PIL import Image

from reifyscene.common.types import CameraIntrinsics
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.render.gaussian4d.seed import _backproject_view


def _write_sequence(root, *, orientation, position, depth_value=2.0, size=10) -> None:
    """Write a minimal one-frame DyCheck-format sequence under ``root``."""
    (root / "camera").mkdir(parents=True)
    (root / "depth" / "1x").mkdir(parents=True)
    (root / "rgb" / "1x").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    cam = {
        "focal_length": 100.0,
        "pixel_aspect_ratio": 1.0,
        "principal_point": [size / 2.0, size / 2.0],
        "image_size": [size, size],
        "orientation": orientation,
        "position": position,
        "radial_distortion": [0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "skew": 0.0,
    }
    (root / "camera" / "0_00000.json").write_text(json.dumps(cam))
    np.save(root / "depth" / "1x" / "0_00000.npy", np.full((size, size), depth_value, dtype=np.float32))
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(root / "rgb" / "1x" / "0_00000.png")
    (root / "splits" / "train.json").write_text(
        json.dumps({"camera_ids": ["0"], "frame_names": ["0_00000"], "time_ids": [0]})
    )


def test_load_camera_pose_convention(tmp_path) -> None:
    # Identity orientation + a known center: T_world_cam must place the camera
    # origin at `position` (it is the inverse of w2c = [R | -R p]).
    _write_sequence(tmp_path, orientation=np.eye(3).tolist(), position=[1.0, 2.0, 3.0])
    backend = DatasetGeometryInitializer({"dataset_root": str(tmp_path), "factor": 1})
    intrinsics, t_world_cam, (width, height) = backend.load_camera("0_00000")

    assert (width, height) == (10, 10)
    assert intrinsics.fx == 100.0 and intrinsics.fy == 100.0
    assert intrinsics.cx == 5.0 and intrinsics.cy == 5.0
    # Camera center in world == position; rotation block is identity.
    cam_center = t_world_cam @ np.asarray([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(cam_center[:3], [1.0, 2.0, 3.0], atol=1e-5)
    assert np.allclose(t_world_cam[:3, :3], np.eye(3), atol=1e-5)


def test_intrinsics_scale_with_factor(tmp_path) -> None:
    _write_sequence(tmp_path, orientation=np.eye(3).tolist(), position=[0.0, 0.0, 0.0])
    backend = DatasetGeometryInitializer({"dataset_root": str(tmp_path), "factor": 2})
    intrinsics, _t, (width, height) = backend.load_camera("0_00000")
    assert intrinsics.fx == 50.0 and intrinsics.cx == 2.5
    assert (width, height) == (5, 5)


def test_initialize_builds_geometry_artifact(tmp_path) -> None:
    _write_sequence(tmp_path, orientation=np.eye(3).tolist(), position=[0.0, 0.0, 1.0])
    backend = DatasetGeometryInitializer({"dataset_root": str(tmp_path), "factor": 1})
    artifact = backend.initialize()  # standalone mode enumerates the split

    assert artifact.backend_name == "dataset"
    assert len(artifact.selected_frames) == 1
    assert "0" in artifact.camera_states_by_stream
    assert len(artifact.camera_states_by_stream["0"]) == 1
    assert "0:0" in artifact.depthmaps_by_frame
    assert artifact.depthmaps_by_frame["0:0"]["depth"].shape == (10, 10)


def test_backproject_view_identity_pose() -> None:
    # Identity pose: world points equal camera-frame points. A pixel at the
    # principal point maps to (0, 0, z).
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    xyz, colors, scales = _backproject_view(
        depth, rgb, intr, np.eye(4, dtype=np.float32),
        pixel_stride=1, depth_near=1e-3, depth_far=None,
        scale_mult=1.0, scale_min=1e-5, scale_max=1.0,
    )
    assert xyz.shape == (16, 3)
    assert colors.shape == (16, 3) and scales.shape == (16, 3)
    # All z == depth; the (u=1, v=1)->principal-point-ish corner check:
    assert np.allclose(xyz[:, 2], 2.0)
    # Pixel (u=cx, v=cy) would be (0,0,z); pixel (u=0,v=0) -> ((0-1.5)/2*2, ...).
    assert np.allclose(xyz[0], [-1.5, -1.5, 2.0], atol=1e-5)


def test_backproject_view_translated_pose() -> None:
    # A pure translation in T_world_cam shifts every world point by it.
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10.0, 0.0, 0.0]
    xyz, _c, _s = _backproject_view(
        depth, rgb, intr, pose,
        pixel_stride=1, depth_near=1e-3, depth_far=None,
        scale_mult=1.0, scale_min=1e-5, scale_max=1.0,
    )
    assert np.allclose(xyz[0], [-1.5 + 10.0, -1.5, 2.0], atol=1e-5)

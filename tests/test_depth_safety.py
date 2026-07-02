"""Depth zero/NaN safety across producers + consumers, and the completed GT oracle.

Zero depth is *finite*, so a plain isfinite guard would back-project a hole to the camera
origin. These lock in the shared `valid_depth_mask` guard, the hardened `Backprojector`, the
dense `unproject_depth_to_world`, and the dataset oracle's NaN-at-holes world point maps.
"""

from __future__ import annotations

import json

import numpy as np

from room4d.common.depth import unproject_depth_to_world, valid_depth_mask
from room4d.common.types import CameraIntrinsics, CameraPose, TimedCameraState
from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.lifting.backproject import Backprojector


def test_valid_depth_mask_excludes_zero_nan_negative():
    d = np.array([[1.0, 0.0], [np.nan, -0.5]], dtype=np.float32)
    assert valid_depth_mask(d).tolist() == [[True, False], [False, False]]
    assert valid_depth_mask(np.array([[1.0, 10.0]]), far=5.0).tolist() == [[True, False]]
    assert valid_depth_mask(np.zeros((2, 2, 1), dtype=np.float32)).shape == (2, 2)  # squeezes HxWx1


def test_unproject_identity_pose_and_holes():
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    depth = np.array([[3.0, 0.0], [3.0, 3.0]], dtype=np.float32)  # (row0,col1) is a hole
    world, valid = unproject_depth_to_world(depth, intr, np.eye(4, dtype=np.float32))
    assert valid.tolist() == [[True, False], [True, True]]
    # pixel (u=0,v=0): x=(0-1)/2*3=-1.5, y=-1.5, z=3 → world == camera (identity pose)
    assert np.allclose(world[0, 0], [-1.5, -1.5, 3.0])
    assert np.all(np.isnan(world[0, 1]))                         # hole → NaN, not origin


def test_backprojector_zero_and_nan_depth_are_nan_not_origin():
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    state = TimedCameraState("cam", 0.0, intr, CameraPose(T_world_cam=np.eye(4, dtype=np.float32)))
    depth = np.array([[3.0, 0.0], [3.0, np.nan]], dtype=np.float32)
    pix = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)   # (u,v): valid, zero-hole, nan
    xyz = Backprojector().backproject(pix, state, depth)
    assert np.all(np.isfinite(xyz[0]))                           # valid
    assert np.all(np.isnan(xyz[1]))                              # z=0 → NaN (was camera origin)
    assert np.all(np.isnan(xyz[2]))                              # z=NaN → NaN


def _fake_dataset(root):
    import cv2
    (root / "splits").mkdir(parents=True)
    (root / "camera").mkdir()
    (root / "depth" / "1x").mkdir(parents=True)
    (root / "rgb" / "1x").mkdir(parents=True)
    json.dump({"camera_ids": ["0", "0"], "time_ids": [0, 1]}, open(root / "splits" / "train.json", "w"))
    for t in (0, 1):
        name = f"0_{t:05d}"
        json.dump(
            {"image_size": [4, 4], "focal_length": 2.0, "principal_point": [2, 2],
             "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "position": [0, 0, 0]},
            open(root / "camera" / f"{name}.json", "w"),
        )
        d = np.full((4, 4), 3.0, dtype=np.float32)
        d[0, 0] = 0.0                                            # a depth hole
        np.save(root / "depth" / "1x" / f"{name}.npy", d)
        cv2.imwrite(str(root / "rgb" / "1x" / f"{name}.png"), np.zeros((4, 4, 3), np.uint8))


def test_dataset_oracle_emits_complete_safe_pointmaps(tmp_path):
    _fake_dataset(tmp_path / "ds")
    art = DatasetGeometryInitializer(
        {"dataset_root": str(tmp_path / "ds"), "factor": 1, "split": "train"}
    ).initialize()

    assert len(art.pointmaps_by_frame) == 2 and len(art.depthmaps_by_frame) == 2
    pm = art.pointmaps_by_frame["0:0"]["points"]
    assert pm.shape == (4, 4, 3)
    assert np.all(np.isnan(pm[0, 0]))                           # hole → NaN
    assert np.isfinite(pm[1, 1]).all()                          # valid → finite world point
    # confidence = validity mask, on both channels
    assert art.pointmaps_by_frame["0:0"]["confidence"][0, 0] == 0.0
    assert art.pointmaps_by_frame["0:0"]["confidence"][1, 1] == 1.0
    assert art.depthmaps_by_frame["0:0"]["confidence"][0, 0] == 0.0
    # RGB linkage: GT frames wired for scene/render seeding (renderable oracle)
    assert set(art.metadata["debug_image_paths"]) == {"0:0", "0:1"}
    assert art.metadata["debug_image_paths"]["0:0"].endswith("rgb/1x/0_00000.png")

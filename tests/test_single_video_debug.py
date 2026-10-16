from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from scripts.run_single_video_vggt import build_config


def test_build_config_debug_enables_geometry_debug(tmp_path: Path) -> None:
    args = Namespace(
        video=str(tmp_path / "clip.mp4"),
        run_dir=str(tmp_path / "run"),
        config=None,
        backend_mode="real",
        device="cpu",
        frame_cache_dir=None,
        max_keyframes=2,
        keyframe_stride=None,
        keyframe_stride_sec=None,
        debug=True,
        debug_max_points=1234,
    )
    cfg = build_config(args)
    assert cfg["geometry"]["debug_shapes"] is True
    assert cfg["geometry"]["save_depth_pngs"] is True
    assert cfg["geometry"]["debug_max_points"] == 1234


def test_geometry_artifact_saves_depth_preview_pngs(tmp_path: Path) -> None:
    artifact = GeometryArtifact(
        run_id="test",
        stage_name="geometry",
        save_dir=tmp_path,
        selected_frames=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
        depthmaps_by_frame={
            "cam_00:0": {
                "depth": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                "confidence": None,
            }
        },
        metadata={"save_depth_pngs": True},
    )
    artifact.save()
    assert (tmp_path / "depthmaps_png" / "cam_00__0.png").exists()


def test_geometry_artifact_writes_camera_json_ply_and_html(tmp_path: Path) -> None:
    src_image = tmp_path / "source.png"
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)).save(src_image)

    frame = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=1.0, cy=1.0)
    state = TimedCameraState(
        stream_id="cam_00",
        frame_idx=0,
        intrinsics=intr,
        pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32)),
    )

    artifact = GeometryArtifact(
        run_id="test",
        stage_name="geometry",
        save_dir=tmp_path,
        selected_frames=[frame],
        camera_states_by_stream={"cam_00": [state]},
        depthmaps_by_frame={
            "cam_00:0": {
                "depth": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "confidence": np.ones((2, 2), dtype=np.float32),
            }
        },
        pointmaps_by_frame={
            "cam_00:0": {
                "points": np.array(
                    [
                        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                        [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
                "confidence": np.ones((2, 2), dtype=np.float32),
            }
        },
        metadata={
            "save_depth_pngs": True,
            "save_debug_assets": True,
            "save_camera_json": True,
            "save_pointcloud_plys": True,
            "save_debug_html": True,
            "debug_max_points": 16,
            "debug_image_paths": {"cam_00:0": str(src_image)},
        },
    )
    artifact.save()
    assert (tmp_path / "cameras.json").exists()
    assert (tmp_path / "pointclouds" / "cam_00__0.ply").exists()
    assert (tmp_path / "debug_viewer.html").exists()
    assert (tmp_path / "images" / "cam_00__0.png").exists()


def test_geometry_artifact_resizes_rgb_for_pointcloud_debug_export(tmp_path: Path) -> None:
    src_image = tmp_path / "source_large.png"
    large_rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    large_rgb[..., 0] = 255
    Image.fromarray(large_rgb).save(src_image)

    frame = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    artifact = GeometryArtifact(
        run_id="test",
        stage_name="geometry",
        save_dir=tmp_path,
        selected_frames=[frame],
        depthmaps_by_frame={
            "cam_00:0": {
                "depth": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "confidence": np.ones((2, 2), dtype=np.float32),
            }
        },
        pointmaps_by_frame={
            "cam_00:0": {
                "points": np.array(
                    [
                        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                        [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
                "confidence": np.ones((2, 2), dtype=np.float32),
            }
        },
        metadata={
            "save_debug_assets": True,
            "save_pointcloud_plys": True,
            "save_debug_html": True,
            "debug_max_points": 16,
            "debug_image_paths": {"cam_00:0": str(src_image)},
        },
    )
    artifact.save()
    assert (tmp_path / "pointclouds" / "cam_00__0.ply").exists()
    assert (tmp_path / "debug_viewer.html").exists()

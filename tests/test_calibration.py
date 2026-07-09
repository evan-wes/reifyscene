from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.api.pipeline import Room4DPipeline
from room4d.calibration.artifact import CalibrationArtifact
from room4d.calibration.model_io import parse_colmap_text_model
from room4d.calibration.runner import CalibrationRunner
from room4d.common.io import write_json
from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState, VideoStreamRef


def _write_fake_colmap_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n"
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n"
        "2 SIMPLE_PINHOLE 640 480 500 320 240\n",
        encoding="utf-8",
    )
    (model_dir / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 cam_00/frame_000000.png\n"
        "100 100 1 120 120 1\n"
        "2 1 0 0 0 1 0 0 2 cam_01/frame_000000.png\n"
        "100 100 1 120 120 1\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "# 3D point list\n"
        "1 0.0 0.0 3.0 255 0 0 0.5 1 0 2 0\n",
        encoding="utf-8",
    )


def test_parse_colmap_text_model_connected_components(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_fake_colmap_model(model_dir)
    parsed = parse_colmap_text_model(model_dir)
    assert len(parsed.intrinsics_by_camera_id) == 2
    assert parsed.reprojection_stats["registered_images"] == 2
    assert parsed.reprojection_stats["num_points3d"] == 1
    assert len(parsed.connected_components) == 1
    assert parsed.connected_components[0] == ["cam_00/frame_000000.png", "cam_01/frame_000000.png"]


def test_calibration_artifact_roundtrip(tmp_path: Path) -> None:
    save_dir = tmp_path / "calibration"
    artifact = CalibrationArtifact(
        run_id="demo",
        stage_name="calibration",
        save_dir=save_dir,
        streams=[VideoStreamRef("cam_00", Path("/tmp/a.mp4"), 30.0, 10, 640, 480, False)],
        sampled_frames=[FrameRef("cam_00", 0, 0.0)],
        sampled_image_index={"cam_00:0": "cam_00/frame_000000.png"},
        intrinsics_by_stream={"cam_00": CameraIntrinsics(500.0, 500.0, 320.0, 240.0)},
        camera_states_by_stream={
            "cam_00": [TimedCameraState("cam_00", 0.0, CameraIntrinsics(500.0, 500.0, 320.0, 240.0), CameraPose(np.eye(4, dtype=np.float32)))]
        },
        sparse_points_xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        sparse_points_rgb=np.asarray([[255, 0, 0]], dtype=np.uint8),
        sparse_point_errors=np.asarray([0.25], dtype=np.float32),
        connected_components=[["cam_00/frame_000000.png"]],
        reprojection_stats={"mean": 0.25},
        notes=["ok"],
        metadata={"viewer_html": "calibration_viewer.html"},
    )
    artifact.save()
    loaded = CalibrationArtifact.load(save_dir)
    assert loaded.sampled_image_index == artifact.sampled_image_index
    assert loaded.connected_components == artifact.connected_components
    assert loaded.reprojection_stats == artifact.reprojection_stats
    assert loaded.sparse_points_xyz.shape == (1, 3)


def test_calibration_runner_exports_report_and_viewer(tmp_path: Path) -> None:
    model_dir = tmp_path / "mock_model"
    _write_fake_colmap_model(model_dir)
    image_dir = tmp_path / "images"
    (image_dir / "cam_00").mkdir(parents=True)
    (image_dir / "cam_01").mkdir(parents=True)
    write_json(
        image_dir / "image_manifest.json",
        {
            "images": {
                "cam_00/frame_000000.png": {"stream_id": "cam_00", "frame_idx": 0, "timestamp_sec": 0.0},
                "cam_01/frame_000000.png": {"stream_id": "cam_01", "frame_idx": 0, "timestamp_sec": 0.0},
            }
        },
    )
    runner = CalibrationRunner({})
    streams = [
        VideoStreamRef("cam_00", Path("/tmp/a.mp4"), 30.0, 1, 640, 480, False),
        VideoStreamRef("cam_01", Path("/tmp/b.mp4"), 30.0, 1, 640, 480, False),
    ]
    refs = [FrameRef("cam_00", 0, 0.0), FrameRef("cam_01", 0, 0.0)]
    runner.sampler.sample = lambda videos: (streams, refs)
    runner.sampler.materialize_images = lambda streams, refs, output_dir: {
        "cam_00:0": "cam_00/frame_000000.png",
        "cam_01:0": "cam_01/frame_000000.png",
    }
    runner.sfm_runner.run = lambda image_dir, output_dir: model_dir
    artifact = runner.run(run_id="demo", save_dir=tmp_path / "run" / "calibration", video_paths=["a.mp4", "b.mp4"])
    assert (artifact.save_dir / "calibration_viewer.html").exists()
    assert (artifact.save_dir / "calibration_report.html").exists()
    assert artifact.reprojection_stats["registered_images"] == 2
    assert len(artifact.connected_components) == 1


def test_pipeline_calibrate_uses_runner_and_stamps_metadata(tmp_path: Path) -> None:
    pipeline = Room4DPipeline({"run_dir": str(tmp_path / "run"), "run_id": "demo"})
    artifact = CalibrationArtifact(
        run_id="demo",
        stage_name="calibration",
        save_dir=tmp_path / "run" / "calibration",
    )
    pipeline.calibration_runner.run = lambda run_id, save_dir, video_paths: artifact
    out = pipeline.calibrate([__file__])
    assert out is artifact
    assert out.metadata["compatibility"]["stage_name"] == "calibration"

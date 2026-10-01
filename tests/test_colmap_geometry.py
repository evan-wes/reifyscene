"""Tests for the FP6b step-1 COLMAP pose rung + its geometry-ladder wiring.

The real SfM core (`reconstruct`) is mocked, so the wrapper's artifact-building + the
ladder/manifest are exercised without hloc/COLMAP or a GPU.
"""

from __future__ import annotations

import cv2
import numpy as np

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import CameraIntrinsics, FrameRef
from reifyscene.geometry.colmap_wrapper import ColmapPoseInitializer
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact


def _fake_poses(indices):
    """{index: (T_world_cam 4x4, intrinsics)} — identity poses, PINHOLE-ish intrinsics."""
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=12.0)
    return {i: (np.eye(4, dtype=np.float32), intr) for i in indices}


def _write_test_video(path, *, width=32, height=24, num_frames=2, fps=5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., :] = 20 + idx
        writer.write(frame)
    writer.release()


def _ingest_for(tmp_path, name="ingest", num_frames=2):
    video = tmp_path / f"{name}.mp4"
    _write_test_video(video, num_frames=num_frames)
    stream = VideoDecoder().decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="t", stage_name="ingest", save_dir=tmp_path / name,
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(num_frames)],
    )


def test_colmap_initialize_empty_in_empty_out():
    init = ColmapPoseInitializer({})
    art = init.initialize(frames=[], ingest_artifact=None)
    assert isinstance(art, GeometryArtifact)
    assert art.backend_name == "colmap"
    assert art.camera_states_by_stream == {} and art.selected_frames == []


def test_colmap_initialize_fills_poses_not_depth(tmp_path, monkeypatch):
    ingest = _ingest_for(tmp_path, num_frames=2)
    init = ColmapPoseInitializer({"colmap_work_dir": str(tmp_path / "cw")})
    # mock the SfM core: both frames registered
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: _fake_poses([0, 1]))
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)

    assert art.backend_name == "colmap"
    assert len(art.camera_states_by_stream["cam_00"]) == 2          # poses filled
    assert art.intrinsics_by_stream["cam_00"].fx == 100.0
    assert art.depthmaps_by_frame == {} and art.pointmaps_by_frame == {}  # pose-only: no depth
    assert [f.frame_idx for f in art.selected_frames] == [0, 1]


def test_colmap_partial_registration(tmp_path, monkeypatch):
    ingest = _ingest_for(tmp_path, num_frames=2)
    init = ColmapPoseInitializer({"colmap_work_dir": str(tmp_path / "cw")})
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: _fake_poses([0]))  # only frame 0
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)
    assert len(art.camera_states_by_stream["cam_00"]) == 1
    assert [f.frame_idx for f in art.selected_frames] == [0]


def test_geometry_ladder_selects_colmap_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: _fake_poses([0]))
    ingest = _ingest_for(tmp_path, num_frames=2)
    pipe = ReifyScenePipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "colmap"}})
    assert isinstance(pipe.geometry_init, ColmapPoseInitializer)

    art = pipe.initialize_geometry(ingest, sync_artifact=None)
    cap = art.capability
    assert cap is not None
    assert cap.backend_used == "colmap" == art.backend_name
    assert cap.fidelity == "sfm_poses_no_depth"
    assert art.camera_states_by_stream  # poses present
    assert art.depthmaps_by_frame == {}


def test_geometry_ladder_colmap_degrades_to_empty(tmp_path, monkeypatch):
    def _boom(self, img_dir):
        raise RuntimeError("colmap reconstruction failed")

    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", _boom)
    ingest = _ingest_for(tmp_path, num_frames=2)
    pipe = ReifyScenePipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "colmap"}})

    art = pipe.initialize_geometry(ingest, sync_artifact=None)  # must not crash
    cap = art.capability
    assert cap is not None
    assert cap.backend_used == "empty" and art.backend_name == "empty"
    assert {r.name: r.status for r in cap.rungs}.get("colmap") == "failed"

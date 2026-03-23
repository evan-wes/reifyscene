"""Tests for the VGGT backend wrapper behaviour."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from room4d.common.types import FrameRef
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.ingest.decoder import VideoDecoder


def _write_test_video(path: Path, *, width: int = 32, height: int = 24, num_frames: int = 2, fps: float = 5.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Failed to create test video."
    for idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., :] = 20 + idx
        writer.write(frame)
    writer.release()


class _FakeModel:
    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, images):
        num = int(images.shape[0])
        extrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num, 1, 1)
        intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num, 1, 1)
        intrinsic[:, 0, 0] = 100.0
        intrinsic[:, 1, 1] = 120.0
        intrinsic[:, 0, 2] = 16.0
        intrinsic[:, 1, 2] = 12.0
        depth = torch.ones((num, 4, 4, 1), dtype=torch.float32)
        points = torch.zeros((num, 4, 4, 3), dtype=torch.float32)
        return {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth": depth,
            "point_map": points,
        }


class _FakeVGGTInitializer(VGGTInitializer):
    def _load_backend_components(self) -> None:
        self._VGGT = None
        self._load_and_preprocess_images = self._fake_preprocess
        self._pose_encoding_to_extri_intri = None
        self._unproject_depth_map_to_point_map = None

    def load_checkpoint(self) -> None:
        self._load_backend_components()
        self._model = _FakeModel()
        self._is_loaded = True

    def _fake_preprocess(self, image_paths: list[str]):
        return torch.zeros((len(image_paths), 3, 8, 8), dtype=torch.float32)


def test_vggt_initializer_real_adapter_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path)

    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="test",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    init = _FakeVGGTInitializer({"backend_mode": "real", "device": "cpu", "frame_cache_dir": str(tmp_path / "cache")})
    artifact = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)

    assert artifact.backend_name == "vggt"
    assert artifact.stream_ids() == ["cam_00"]
    assert artifact.intrinsics_by_stream["cam_00"].fx == 100.0
    assert len(artifact.camera_states_by_stream["cam_00"]) == 1
    assert artifact.depthmaps_by_frame["cam_00:0"]["depth"].shape == (4, 4, 1)


from room4d.geometry.geometry_artifact import GeometryArtifact


def test_geometry_artifact_save_load_roundtrip(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_roundtrip.mp4"
    _write_test_video(video_path)

    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="roundtrip",
        stage_name="ingest",
        save_dir=tmp_path / "ingest",
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    init = _FakeVGGTInitializer({"backend_mode": "real", "device": "cpu", "frame_cache_dir": str(tmp_path / "cache")})
    artifact = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)
    artifact.save_dir = tmp_path / "geometry"
    artifact.save()

    loaded = GeometryArtifact.load(tmp_path / "geometry")
    assert loaded.backend_name == "vggt"
    assert loaded.selected_frames[0].stream_id == "cam_00"
    assert loaded.depthmaps_by_frame["cam_00:0"]["depth"].shape == (4, 4, 1)
    assert loaded.pointmaps_by_frame["cam_00:0"]["points"].shape == (4, 4, 3)

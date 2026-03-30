"""Tests for the VGGT backend wrapper behaviour."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from room4d.common.types import FrameRef
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.geometry.vggt_long_wrapper import VGGTLongInitializer
from room4d.api.pipeline import Room4DPipeline
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


class _FakeVGGTLongInitializer(VGGTLongInitializer):
    def load_checkpoint(self) -> None:
        self._VGGT_Long = object()
        self._is_loaded = True

    def _load_chunk_predictions(self, runner):
        return [
            {
                '_chunk_range': (0, 2),
                'extrinsic': np.stack([np.eye(4, dtype=np.float32)[:3], np.eye(4, dtype=np.float32)[:3]], axis=0),
                'intrinsic': np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0),
                'depth': np.ones((2, 4, 4, 1), dtype=np.float32),
                'depth_conf': np.ones((2, 4, 4), dtype=np.float32),
                'world_points': np.zeros((2, 4, 4, 3), dtype=np.float32),
                'world_points_conf': np.ones((2, 4, 4), dtype=np.float32),
            }
        ]

    def initialize(self, frames, *, ingest_artifact=None, sync_artifact=None):
        selected_frames = self._select_frames(frames, ingest_artifact)
        if not selected_frames:
            raise ValueError('No frames')
        notes = ['fake VGGT-Long backend']
        predictions = self._combined_predictions_from_chunks(selected_frames, self._load_chunk_predictions(None))
        artifact = self._initialize_from_predictions(selected_frames, predictions, ingest_artifact, notes=notes, image_paths=None)
        artifact.backend_name = self.backend_name
        return artifact


def test_vggt_long_initializer_real_adapter_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_long.mp4"
    _write_test_video(video_path, num_frames=2)

    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="test_long",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[
            FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0),
            FrameRef(stream_id="cam_00", frame_idx=1, timestamp_sec=1.0 / 30.0),
        ],
    )

    init = _FakeVGGTLongInitializer({"backend_mode": "real", "device": "cpu", "chunk_size": 4, "overlap": 1})
    artifact = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)

    assert artifact.backend_name == "vggt-long"
    assert artifact.stream_ids() == ["cam_00"]
    assert len(artifact.camera_states_by_stream["cam_00"]) == 2
    assert artifact.depthmaps_by_frame["cam_00:0"]["depth"].shape == (4, 4, 1)
    assert artifact.pointmaps_by_frame["cam_00:1"]["points"].shape == (4, 4, 3)


def test_pipeline_selects_vggt_long_initializer(tmp_path: Path) -> None:
    pipe = Room4DPipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "vggt-long", "backend_mode": "heuristic"}})
    assert isinstance(pipe.geometry_init, VGGTLongInitializer)


class _CpuProbeRunner:
    def __init__(self, image_dir, save_dir, config):
        # Constructor should see CUDA as unavailable in CPU mode.
        self.constructor_cuda_available = torch.cuda.is_available()
        self.constructor_capability = torch.cuda.get_device_capability()
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.config = config
        self.device = None
        self.dtype = None
        self.model = type("_Model", (), {"device": None, "dtype": None})()
        self.run_cuda_available = None
        self.run_capability = None

    def run(self):
        self.run_cuda_available = torch.cuda.is_available()
        self.run_capability = torch.cuda.get_device_capability()


class _CpuProbeVGGTLong(VGGTLongInitializer):
    def load_checkpoint(self) -> None:
        self._VGGT_Long = _CpuProbeRunner
        self._is_loaded = True

    def _load_chunk_predictions(self, runner):
        return [
            {
                '_chunk_range': (0, 1),
                'extrinsic': np.stack([np.eye(4, dtype=np.float32)[:3]], axis=0),
                'intrinsic': np.stack([np.eye(3, dtype=np.float32)], axis=0),
                'depth': np.ones((1, 4, 4, 1), dtype=np.float32),
                'depth_conf': np.ones((1, 4, 4), dtype=np.float32),
                'world_points': np.zeros((1, 4, 4, 3), dtype=np.float32),
                'world_points_conf': np.ones((1, 4, 4), dtype=np.float32),
            }
        ]

    def initialize(self, frames, *, ingest_artifact=None, sync_artifact=None):
        if not self._is_loaded:
            self.load_checkpoint()
        selected_frames = self._select_frames(frames, ingest_artifact)
        cache_root = Path(self.frame_cache_dir) if self.frame_cache_dir else Path('.room4d_cache') / 'vggt_long_inputs'
        image_paths, notes = self._materialize_images(selected_frames, ingest_artifact, cache_root)
        backend_output_root = cache_root.parent / 'vggt_long_run'
        backend_output_root.mkdir(parents=True, exist_ok=True)
        runner = self._create_runner(cache_root, backend_output_root)
        runner.img_list = [str(path) for path in image_paths]
        self._run_runner(runner)
        self._last_probe_runner = runner
        predictions = self._combined_predictions_from_chunks(selected_frames, self._load_chunk_predictions(runner))
        artifact = self._initialize_from_predictions(selected_frames, predictions, ingest_artifact, notes=notes, image_paths=image_paths)
        artifact.backend_name = self.backend_name
        return artifact


def test_vggt_long_cpu_enforcement_patches_torch_cuda_during_runner_lifecycle(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_cpu_probe.mp4"
    _write_test_video(video_path, num_frames=1)

    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="cpu_probe",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    init = _CpuProbeVGGTLong({"backend_mode": "real", "device": "cpu", "chunk_size": 4, "overlap": 1})
    artifact = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)
    runner = init._last_probe_runner

    assert artifact.backend_name == "vggt-long"
    assert runner.constructor_cuda_available is False
    assert runner.run_cuda_available is False
    assert runner.device == "cpu"
    assert runner.model.device == "cpu"

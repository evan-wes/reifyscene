"""Tests for the VGGT backend wrapper behaviour."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import FrameRef
from reifyscene.geometry.vggt_derived import load_and_preprocess_images_without_upscale
from reifyscene.geometry.vggt_long_wrapper import VGGTLongInitializer
from reifyscene.geometry.vggt_wrapper import VGGTInitializer, input_cache_root
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact


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

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="test",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    init = _FakeVGGTInitializer({"backend_mode": "real", "device": "cpu", "frame_cache_dir": f"{tmp_path.name}/cache"})
    artifact = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)

    assert artifact.backend_name == "vggt"
    assert artifact.stream_ids() == ["cam_00"]
    assert artifact.intrinsics_by_stream["cam_00"].fx == 100.0
    assert len(artifact.camera_states_by_stream["cam_00"]) == 1
    assert artifact.depthmaps_by_frame["cam_00:0"]["depth"].shape == (4, 4, 1)


from reifyscene.geometry.geometry_artifact import GeometryArtifact


def test_geometry_artifact_save_load_roundtrip(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_roundtrip.mp4"
    _write_test_video(video_path)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="roundtrip",
        stage_name="ingest",
        save_dir=tmp_path / "ingest",
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    init = _FakeVGGTInitializer({"backend_mode": "real", "device": "cpu", "frame_cache_dir": f"{tmp_path.name}/cache"})
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
                '_chunk_index': 0,
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
        predictions = self._combined_predictions_from_chunks(
            selected_frames, self._load_chunk_predictions(None),
            global_c2w=np.stack([np.eye(4)] * len(selected_frames)), chunk_scales=[1.0],
        )
        artifact = self._initialize_from_predictions(selected_frames, predictions, ingest_artifact, notes=notes, image_paths=None)
        artifact.backend_name = self.backend_name
        return artifact


def test_vggt_long_initializer_real_adapter_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_long.mp4"
    _write_test_video(video_path, num_frames=2)

    decoder = VideoDecoder()
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
    pipe = ReifyScenePipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "vggt-long", "backend_mode": "heuristic"}})
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
                '_chunk_index': 0,
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
        cache_root = input_cache_root(self.frame_cache_dir)
        image_paths, notes = self._materialize_images(selected_frames, ingest_artifact, cache_root)
        backend_output_root = cache_root.parent / 'vggt_long_run'
        backend_output_root.mkdir(parents=True, exist_ok=True)
        runner = self._create_runner(cache_root, backend_output_root)
        runner.img_list = [str(path) for path in image_paths]
        self._run_runner(runner)
        self._last_probe_runner = runner
        predictions = self._combined_predictions_from_chunks(
            selected_frames, self._load_chunk_predictions(runner),
            global_c2w=np.stack([np.eye(4)] * len(selected_frames)), chunk_scales=[1.0],
        )
        artifact = self._initialize_from_predictions(selected_frames, predictions, ingest_artifact, notes=notes, image_paths=image_paths)
        artifact.backend_name = self.backend_name
        return artifact


class _AutoTuneVGGTLong(VGGTLongInitializer):
    def load_checkpoint(self) -> None:
        self._VGGT_Long = object()
        self._is_loaded = True

    def _create_runner(self, image_dir: Path, output_dir: Path):
        # Auto-tune sets resize_max_dim for the attempt before creating its runner; the
        # images live in that setting's content-keyed cache directory.
        runner = type("_Runner", (), {})()
        runner.image_dir = image_dir
        runner.output_dir = output_dir
        runner.resize_max_dim = self.resize_max_dim
        return runner

    def _run_runner(self, runner) -> None:
        if self.chunk_size > 4 or (runner.resize_max_dim is None or runner.resize_max_dim > 128):
            raise torch.OutOfMemoryError("CUDA out of memory during fake VGGT-Long run")
        # What a successful run leaves behind: one chunk, its global poses, no alignments.
        runner.chunk_indices = [(0, 8)]
        runner.sim3_list = []
        (Path(runner.output_dir) / "camera_poses.txt").write_text(
            "\n".join(" ".join(["1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "1"])
                      for _ in range(8)) + "\n"
        )

    def _load_chunk_predictions(self, runner):
        num_frames = 8
        return [
            {
                '_chunk_range': (0, num_frames),
                '_chunk_index': 0,
                'extrinsic': np.stack([np.eye(4, dtype=np.float32)[:3] for _ in range(num_frames)], axis=0),
                'intrinsic': np.stack([np.eye(3, dtype=np.float32) for _ in range(num_frames)], axis=0),
                'depth': np.ones((num_frames, 4, 4, 1), dtype=np.float32),
                'depth_conf': np.ones((num_frames, 4, 4), dtype=np.float32),
                'world_points': np.zeros((num_frames, 4, 4, 3), dtype=np.float32),
                'world_points_conf': np.ones((num_frames, 4, 4), dtype=np.float32),
            }
        ]


def test_vggt_long_auto_tune_backs_off_chunk_and_resize_on_oom(tmp_path: Path, capsys) -> None:
    video_path = tmp_path / "sample_auto_tune.mp4"
    _write_test_video(video_path, width=320, height=240, num_frames=8)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    frames = [FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(8)]
    ingest = IngestArtifact(
        run_id="auto_tune",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=frames,
    )

    init = _AutoTuneVGGTLong({
        "backend_mode": "real",
        "device": "cuda",
        "auto_tune": True,
        "chunk_size": 8,
        "overlap": 2,
        "resize_max_dim": 256,
        "min_resize_max_dim": 128,
        "min_chunk_size": 2,
        "system_report": {"memory": {"available_gib": 64.0}, "gpus": [{"memory_free_mb": 8192}]},
    })
    artifact = init.initialize(frames, ingest_artifact=ingest, sync_artifact=None)

    assert artifact.backend_name == "vggt-long"
    assert artifact.metadata["vggt_long_auto_tune"] is True
    assert artifact.metadata["vggt_long_chunk_size"] == 4
    assert artifact.metadata["vggt_long_resize_max_dim"] == 128
    attempts = artifact.metadata["vggt_long_auto_tune_attempts"]
    assert len(attempts) >= 2
    assert attempts[-1]["status"] == "success"
    assert any(item["status"] == "oom" for item in attempts[:-1])
    captured = capsys.readouterr()
    assert "[VGGT-Long auto-tune] attempt 1/" in captured.out
    assert "trying chunk_size=" in captured.out
    assert "failed with OOM" in captured.out
    assert "succeeded with" in captured.out


def test_vggt_long_oom_detection_handles_missing_top_level_torch_oom(monkeypatch) -> None:
    monkeypatch.delattr(torch, "OutOfMemoryError", raising=False)
    init = VGGTLongInitializer({"backend_mode": "real", "device": "cuda"})

    assert init._is_probable_oom_error(torch.cuda.OutOfMemoryError("CUDA out of memory")) is True
    assert init._is_probable_oom_error(RuntimeError("CUDA out of memory while loading model")) is True


def test_vggt_long_gpu_runs_still_deserialize_weights_on_cpu(monkeypatch) -> None:
    init = VGGTLongInitializer({"backend_mode": "real", "device": "cuda"})
    seen = {}

    def _fake_torch_load(*args, **kwargs):
        seen["map_location"] = kwargs.get("map_location")
        return "ok"

    monkeypatch.setattr(torch, "load", _fake_torch_load)
    with init._patched_torch_load_for_device():
        result = torch.load("weights.pt", map_location="cuda")

    assert result == "ok"
    assert isinstance(seen["map_location"], torch.device)
    assert seen["map_location"].type == "cpu"



def test_vggt_gpu_runs_still_deserialize_weights_on_cpu(monkeypatch) -> None:
    init = VGGTInitializer({"backend_mode": "real", "device": "cuda"})
    seen = {}

    def _fake_torch_load(*args, **kwargs):
        seen["map_location"] = kwargs.get("map_location")
        return "ok"

    monkeypatch.setattr(torch, "load", _fake_torch_load)
    with init._patched_torch_load_for_cpu_deserialize():
        result = torch.load("weights.pt", map_location=torch.device("cuda"))

    assert result == "ok"
    assert isinstance(seen["map_location"], torch.device)
    assert seen["map_location"].type == "cpu"



def test_vggt_long_cpu_enforcement_patches_torch_cuda_during_runner_lifecycle(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_cpu_probe.mp4"
    _write_test_video(video_path, num_frames=1)

    decoder = VideoDecoder()
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



def test_pipeline_dense_camera_interpolation_matches_tracking_timeline(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_dense_interp.mp4"
    _write_test_video(video_path, num_frames=5, fps=10.0)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="dense_interp",
        stage_name="ingest",
        save_dir=tmp_path / "ingest_dense",
        streams=[stream],
        keyframes=[
            FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0),
            FrameRef(stream_id="cam_00", frame_idx=4, timestamp_sec=0.4),
        ],
    )

    pipe = ReifyScenePipeline(
        {
            "run_dir": tmp_path / "run_dense",
            "geometry": {"backend_name": "vggt", "backend_mode": "heuristic", "frame_source": "keyframes"},
            "tracking": {"frame_stride": 1},
        }
    )
    artifact = pipe.initialize_geometry(ingest, sync_artifact=None)

    assert [frame.frame_idx for frame in artifact.selected_frames] == [0, 4]
    states = artifact.camera_states_by_stream["cam_00"]
    assert [round(float(state.time_sec), 6) for state in states] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert artifact.metadata["dense_camera_target"] == "tracking"
    assert artifact.metadata["dense_camera_summary"]["added_camera_states"] == 3


def test_pipeline_geometry_frame_source_tracking_uses_tracker_timeline(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_tracking_source.mp4"
    _write_test_video(video_path, num_frames=6, fps=10.0)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="tracking_source",
        stage_name="ingest",
        save_dir=tmp_path / "ingest_tracking",
        streams=[stream],
        keyframes=[
            FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0),
            FrameRef(stream_id="cam_00", frame_idx=5, timestamp_sec=0.5),
        ],
    )

    pipe = ReifyScenePipeline(
        {
            "run_dir": tmp_path / "run_tracking",
            "geometry": {"backend_name": "vggt", "backend_mode": "heuristic", "frame_source": "tracking"},
            "tracking": {"frame_stride": 2},
        }
    )
    artifact = pipe.initialize_geometry(ingest, sync_artifact=None)

    assert [frame.frame_idx for frame in artifact.selected_frames] == [0, 2, 4]
    states = artifact.camera_states_by_stream["cam_00"]
    assert [round(float(state.time_sec), 6) for state in states] == [0.0, 0.2, 0.4]
    assert artifact.metadata["geometry_frame_source"] == "tracking"


def test_pipeline_geometry_frame_source_all_uses_entire_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_all_source.mp4"
    _write_test_video(video_path, num_frames=4, fps=8.0)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    ingest = IngestArtifact(
        run_id="all_source",
        stage_name="ingest",
        save_dir=tmp_path / "ingest_all",
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )

    pipe = ReifyScenePipeline(
        {
            "run_dir": tmp_path / "run_all",
            "geometry": {"backend_name": "vggt", "backend_mode": "heuristic", "frame_source": "all"},
            "tracking": {"frame_stride": 2},
        }
    )
    artifact = pipe.initialize_geometry(ingest, sync_artifact=None)

    assert [frame.frame_idx for frame in artifact.selected_frames] == [0, 1, 2, 3]
    states = artifact.camera_states_by_stream["cam_00"]
    assert [round(float(state.time_sec), 6) for state in states] == [0.0, 0.125, 0.25, 0.375]
    assert artifact.metadata["geometry_frame_source"] == "all"


class _AlwaysFailAutoTuneVGGTLong(_AutoTuneVGGTLong):
    def _run_runner(self, runner) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory while loading fake VGGT-Long model")


def test_vggt_long_auto_tune_exhaustion_reports_attempt_summary(tmp_path: Path) -> None:
    video_path = tmp_path / "sample_auto_tune_fail.mp4"
    _write_test_video(video_path, width=320, height=240, num_frames=4)

    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id="cam_00")
    frames = [FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(4)]
    ingest = IngestArtifact(
        run_id="auto_tune_fail",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=frames,
    )

    init = _AlwaysFailAutoTuneVGGTLong({
        "backend_mode": "real",
        "device": "cuda",
        "auto_tune": True,
        "chunk_size": 4,
        "overlap": 0,
        "resize_max_dim": 256,
        "min_resize_max_dim": 128,
        "min_chunk_size": 2,
        "auto_tune_max_attempts": 3,
        "system_report": {"memory": {"available_gib": 64.0}, "gpus": [{"memory_free_mb": 8192}]},
    })

    with pytest.raises(RuntimeError, match="Attempts: attempt 1:") as excinfo:
        init.initialize(frames, ingest_artifact=ingest, sync_artifact=None)

    message = str(excinfo.value)
    assert "chunk_size=" in message
    assert "resize_max_dim=" in message
    assert "status=oom" in message


def test_vggt_long_auto_tune_cleanup_logs_memory(tmp_path, monkeypatch, capsys):
    initializer = VGGTLongInitializer({
        "backend_mode": "real",
        "device": "cuda",
        "auto_tune": True,
        "chunk_size": 8,
        "overlap": 2,
        "auto_tune_max_attempts": 1,
    })
    initializer._VGGT_Long = object()

    frame = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    ingest = IngestArtifact(
        run_id="cleanup-test",
        stage_name="ingest",
        save_dir=Path("unsaved"),
        streams=[SimpleNamespace(stream_id="cam_00", width=854, height=480)],
        keyframes=[frame],
    )

    monkeypatch.setattr(initializer, "_current_system_report", lambda: {"memory": {}, "gpus": []})
    monkeypatch.setattr(initializer, "_frame_source_max_dim", lambda frames, ingest_artifact: 854)
    monkeypatch.setattr(initializer, "_materialize_images", lambda *args, **kwargs: ([tmp_path / "f0.png"], []))
    monkeypatch.setattr(initializer, "_create_runner", lambda *args, **kwargs: SimpleNamespace(img_list=[], model=SimpleNamespace(model=None)))
    monkeypatch.setattr(initializer, "_run_runner", lambda runner: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    monkeypatch.setattr(initializer, "_release_runner_references", lambda runner: None)
    monkeypatch.setattr(
        "reifyscene.geometry.vggt_long_wrapper.process_memory_snapshot",
        lambda: {
            "rss_bytes": 3 * 1024 ** 3,
            "swap_bytes": 1024 ** 3,
            "cuda_allocated_bytes": 2 * 1024 ** 3,
            "cuda_reserved_bytes": 4 * 1024 ** 3,
        },
    )
    monkeypatch.setattr("reifyscene.geometry.vggt_long_wrapper.release_memory", lambda: None)

    with pytest.raises(RuntimeError):
        initializer._run_auto_tuned(
            [frame],
            ingest_artifact=ingest,
            cache_root=tmp_path / "cache",
            backend_output_root=tmp_path / "out",
        )

    captured = capsys.readouterr().out
    assert "cleanup starting; before cleanup rss=3.00 GiB" in captured
    assert "cleanup finished; after cleanup rss=3.00 GiB" in captured


def test_vggt_long_bounded_preprocess_avoids_reinflating_small_inputs(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[..., 1] = 127
    cv2.imwrite(str(image_path), image)

    init = VGGTLongInitializer({"backend_mode": "real", "device": "cuda", "resize_max_dim": 320})
    target_size = init._effective_vggt_input_target_size(320)
    batch = load_and_preprocess_images_without_upscale([str(image_path)], target_size=target_size)

    assert batch.shape[0] == 1
    assert batch.shape[-1] == target_size
    assert batch.shape[-1] < 518
    assert batch.shape[-2] <= target_size


def test_vggt_long_true_mixed_precision_patch_overrides_forward_without_third_party_edit(monkeypatch) -> None:
    base_models_module = ModuleType("base_models")
    vggt_pkg = ModuleType("base_models.vggt")
    models_pkg = ModuleType("base_models.vggt.models")
    vggt_module = ModuleType("base_models.vggt.models.vggt")

    class FakeVGGT:
        def __init__(self) -> None:
            self.camera_head = lambda tokens: [torch.ones((1, 2, 9), dtype=torch.float32)]
            self.depth_head = lambda tokens, images=None, patch_start_idx=None: (
                torch.ones((1, 2, 4, 4, 1), dtype=torch.float32),
                torch.ones((1, 2, 4, 4), dtype=torch.float32),
            )
            self.point_head = lambda tokens, images=None, patch_start_idx=None: (
                torch.ones((1, 2, 4, 4, 3), dtype=torch.float32),
                torch.ones((1, 2, 4, 4), dtype=torch.float32),
            )
            self.track_head = None

        def aggregator(self, images):
            return [torch.ones((1, 2, 4), dtype=torch.float32)], 0

        def forward(self, images, query_points=None):
            return {"original": True}

    vggt_module.VGGT = FakeVGGT
    monkeypatch.setitem(sys.modules, "base_models", base_models_module)
    monkeypatch.setitem(sys.modules, "base_models.vggt", vggt_pkg)
    monkeypatch.setitem(sys.modules, "base_models.vggt.models", models_pkg)
    monkeypatch.setitem(sys.modules, "base_models.vggt.models.vggt", vggt_module)

    init = VGGTLongInitializer({"backend_mode": "real", "device": "cuda", "enable_true_mixed_precision": True})
    model = FakeVGGT()

    assert model.forward(torch.zeros((2, 3, 28, 28))) == {"original": True}

    with init._patched_vggt_true_mixed_precision():
        predictions = model.forward(torch.zeros((2, 3, 28, 28)))
        assert "pose_enc" in predictions
        assert predictions["pose_enc"].dtype == torch.float32
        assert "depth" in predictions
        assert "world_points" in predictions
        assert "images" in predictions

    assert model.forward(torch.zeros((2, 3, 28, 28))) == {"original": True}


def test_vggt_promotes_camera_predictions_before_pose_decoding() -> None:
    init = VGGTInitializer({"backend_mode": "real", "device": "cuda"})

    pose_enc = torch.ones((1, 2, 9), dtype=torch.bfloat16)
    extrinsic = torch.ones((1, 2, 4, 4), dtype=torch.bfloat16)
    intrinsic = torch.ones((1, 2, 3, 3), dtype=torch.bfloat16)

    promoted = init._promote_camera_prediction_precision({
        "pose_enc": pose_enc,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": torch.ones((1, 2, 4, 4, 1), dtype=torch.bfloat16),
    })

    assert promoted["pose_enc"].dtype == torch.float32
    assert promoted["extrinsic"].dtype == torch.float32
    assert promoted["intrinsic"].dtype == torch.float32
    assert promoted["depth"].dtype == torch.bfloat16


def test_geometry_asked_for_as_real_fails_instead_of_going_empty(tmp_path) -> None:
    from types import SimpleNamespace

    from reifyscene.api.pipeline import ReifyScenePipeline
    from reifyscene.common.backend_ladder import BackendRequired

    class Crashing:
        backend_name = "vggt-long"

        def initialize(self, frames, **kwargs):
            raise RuntimeError("VGGT-Long produced no readable chunk predictions.")

    ingest = SimpleNamespace(run_id="r", streams=[])
    real = ReifyScenePipeline({"run_dir": str(tmp_path / "a"), "geometry": {"backend_mode": "real"}})
    with pytest.raises(BackendRequired, match="no readable chunk predictions"):
        real._geometry_ladder("geometry", Crashing(), "vggt-long", [], ingest, None)
    auto = ReifyScenePipeline({"run_dir": str(tmp_path / "b"), "geometry": {"backend_mode": "auto"}})
    artifact, manifest = auto._geometry_ladder("geometry", Crashing(), "vggt-long", [], ingest, None)
    assert manifest.backend_used == "empty" and manifest.rungs[0].status == "failed"

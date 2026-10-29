"""The VGGT-Omega geometry backend: its artifact, its recorded input transform, and its failures.

The model is faked except in the loader parity tests, which compare the recorded transform
with VGGT-Omega's own loader pixel for pixel when ``vggt_omega`` can be imported.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import FrameRef
from reifyscene.geometry.image_transform import (
    Crop,
    Resize,
    image_transform_for,
    materialize_resize,
    vggt_omega_input_transform,
)
from reifyscene.geometry.vggt_omega_wrapper import (
    VGGTOmegaInitializer,
    _import_vggt_omega,
    estimate_peak_gib,
)
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact


def _ingest(tmp_path: Path, num_frames: int = 3) -> IngestArtifact:
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    for idx in range(num_frames):
        writer.write(np.full((24, 32, 3), 20 + idx, np.uint8))
    writer.release()
    stream = VideoDecoder().decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="omega",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[
            FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=i / 5.0)
            for i in range(num_frames)
        ],
    )


class _FakeOmegaModel:
    """Outputs in VGGT-Omega's layout: depth (1, S, H, W, 1), depth_conf (1, S, H, W)."""

    def __call__(self, images):
        s, _, h, w = images.shape
        depth = (
            torch.arange(1, s + 1, dtype=torch.float32)
            .view(1, s, 1, 1, 1)
            .expand(1, s, h, w, 1)
            .clone()
        )
        return {
            "pose_enc": torch.zeros((1, s, 9)),
            "depth": depth,
            "depth_conf": torch.full((1, s, h, w), 2.0),
        }


class _OutOfMemoryModel:
    def __call__(self, images):
        raise torch.OutOfMemoryError("CUDA out of memory (fake)")


def _fake_encoding_to_camera(pose_enc, image_size_hw):
    """Frame i: camera centre at x = i, fx = 100 + 10 i."""
    s = pose_enc.shape[1]
    h, w = image_size_hw
    extrinsic = torch.zeros((1, s, 3, 4))
    extrinsic[0, :, :, :3] = torch.eye(3)
    extrinsic[0, :, 0, 3] = -torch.arange(s, dtype=torch.float32)
    intrinsic = torch.zeros((1, s, 3, 3))
    intrinsic[0, :, 0, 0] = 100.0 + 10.0 * torch.arange(s, dtype=torch.float32)
    intrinsic[0, :, 1, 1] = 100.0
    intrinsic[0, :, 0, 2] = w / 2
    intrinsic[0, :, 1, 2] = h / 2
    intrinsic[0, :, 2, 2] = 1.0
    return extrinsic, intrinsic


class _FakeOmega(VGGTOmegaInitializer):
    model = _FakeOmegaModel
    # The loader's output size (w, h); None means the size the recorded transform gives.
    loader_size: tuple[int, int] | None = None

    def load_checkpoint(self) -> None:
        self._load_and_preprocess_images = self._fake_loader
        self._encoding_to_camera = _fake_encoding_to_camera
        self._model = self.model()
        self._is_loaded = True

    def _fake_loader(self, paths, mode, image_resolution, patch_size):
        assert (mode, image_resolution, patch_size) == (self.omega_mode, self.omega_resolution, 16)
        w, h = (
            self.loader_size
            or vggt_omega_input_transform(
                (32, 24), mode=mode, resolution=image_resolution
            ).output_size
        )
        return torch.zeros((len(paths), 3, h, w))


def _init(tmp_path: Path, **cfg) -> VGGTOmegaInitializer:
    return _FakeOmega(
        {
            "backend_mode": "real",
            "device": "cpu",
            "frame_cache_dir": f"{tmp_path.name}/cache",
            "omega_resolution": 64,
            **cfg,
        }
    )


def test_artifact_has_per_frame_intrinsics_depth_and_lifted_points(tmp_path: Path) -> None:
    ingest = _ingest(tmp_path)
    artifact = _init(tmp_path).initialize(ingest.keyframes, ingest_artifact=ingest)

    # 32x24 balanced at resolution 64: 16 patches at aspect 0.75 -> 5 x 3 patches.
    assert artifact.backend_name == "vggt-omega"
    states = artifact.camera_states_by_stream["cam_00"]
    assert [s.intrinsics.fx for s in states] == [100.0, 110.0, 120.0]
    assert artifact.intrinsics_by_stream["cam_00"].fx == 110.0  # the median
    assert [s.pose.T_world_cam[0, 3] for s in states] == [0.0, 1.0, 2.0]
    depth = artifact.depthmaps_by_frame["cam_00:2"]
    assert depth["depth"].shape == (48, 80, 1) and float(depth["depth"][0, 0, 0]) == 3.0
    assert depth["confidence"].shape == (48, 80)
    points = artifact.pointmaps_by_frame["cam_00:1"]["points"]
    assert points.shape == (48, 80, 3)
    # Pixel (u, v) = (0, 0) of frame 1: depth 2, fx 110, centre (40, 24), camera at x = 1.
    np.testing.assert_allclose(
        points[0, 0], [1.0 + (0 - 40) / 110 * 2, (0 - 24) / 100 * 2, 2.0], rtol=1e-5
    )

    transform = image_transform_for(artifact, "cam_00")
    assert transform.output_size == (80, 48)
    assert transform.ops == (Resize((80, 48), "bicubic"),)
    assert artifact.metadata["image_transforms"]["cam_00"]["recorded_by"] == "vggt-omega"
    assert artifact.metadata["vggt_omega"]["mode"] == "balanced"
    assert artifact.metadata["vggt_omega"]["input_size"] == [80, 48]


def test_a_loader_size_the_transform_does_not_give_is_refused(tmp_path: Path) -> None:
    ingest = _ingest(tmp_path)
    init = _init(tmp_path)
    init.loader_size = (64, 48)
    with pytest.raises(RuntimeError, match="recorded VGGT-Omega input transform"):
        init.initialize(ingest.keyframes, ingest_artifact=ingest)


def test_running_out_of_memory_says_how_much_is_needed_and_the_ways_out(tmp_path: Path) -> None:
    ingest = _ingest(tmp_path)
    init = _init(tmp_path)
    init.model = _OutOfMemoryModel
    with pytest.raises(RuntimeError) as err:
        init.initialize(ingest.keyframes, ingest_artifact=ingest)
    message = str(err.value)
    assert "3 frames at 80x48" in message
    assert f"about {estimate_peak_gib(3, (80, 48)):.1f} GiB" in message
    assert "--vggt-omega-mode max_size" in message and "--geometry-frame-stride" in message


def test_peak_memory_estimate_matches_the_measured_runs() -> None:
    # RTX 5090, 2026-10-28: paper-windmill 277 frames at 448x592, spin 426 at 384x512.
    assert abs(estimate_peak_gib(277, (448, 592)) - 25.3) < 0.3
    assert abs(estimate_peak_gib(426, (384, 512)) - 28.4) < 0.3


def test_unknown_omega_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="omega_mode"):
        VGGTOmegaInitializer({"omega_mode": "crop"})


def test_pipeline_selects_vggt_omega_and_refuses_unknown_backends(tmp_path: Path) -> None:
    pipe = ReifyScenePipeline(
        {
            "run_dir": tmp_path / "run",
            "geometry": {"backend_name": "vggt-omega", "backend_mode": "heuristic"},
        }
    )
    assert isinstance(pipe.geometry_init, VGGTOmegaInitializer)
    with pytest.raises(ValueError, match="unknown geometry backend 'vggt-omgea'"):
        ReifyScenePipeline(
            {"run_dir": tmp_path / "run2", "geometry": {"backend_name": "vggt-omgea"}}
        )


def test_heuristic_fallback_carries_the_backend_name(tmp_path: Path) -> None:
    ingest = _ingest(tmp_path, num_frames=1)
    artifact = VGGTOmegaInitializer({"backend_mode": "heuristic"}).initialize(
        ingest.keyframes, ingest_artifact=ingest
    )
    assert artifact.backend_name == "vggt-omega"
    assert artifact.depthmaps_by_frame == {}


# -- loader parity ----------------------------------------------------------------
def _omega_loader():
    try:
        _import_vggt_omega()
        from vggt_omega.utils.load_fn import load_and_preprocess_images
    except ImportError:
        pytest.skip("vggt_omega is not importable")
    return load_and_preprocess_images


def _photo(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(width * height)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack(
        [xx * 255 // (width - 1), yy * 255 // (height - 1), rng.integers(0, 256, (height, width))],
        -1,
    )
    return img.astype(np.uint8)


@pytest.mark.parametrize("size", [(854, 480), (720, 960), (1000, 300), (300, 1000)])
@pytest.mark.parametrize("mode", ["balanced", "max_size"])
def test_recorded_transform_reproduces_the_omega_loader(tmp_path: Path, size, mode) -> None:
    load = _omega_loader()
    path = tmp_path / "frame.png"
    Image.fromarray(_photo(*size)).save(path)
    transform = vggt_omega_input_transform(size, mode=mode, resolution=512)
    theirs = (
        load([str(path)], mode=mode, image_resolution=512)[0].permute(1, 2, 0).numpy() * 255
    ).round()
    with Image.open(path) as img:
        ours = transform.apply_to_image(img)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs.astype(np.uint8))


def test_transform_sizes_for_the_test_clips() -> None:
    assert vggt_omega_input_transform((854, 480)).output_size == (688, 384)  # DAVIS
    assert vggt_omega_input_transform((720, 960)).output_size == (448, 592)  # DyCheck
    assert vggt_omega_input_transform((720, 960), mode="max_size").output_size == (384, 512)
    # Aspect ratios outside [0.5, 2] are centre-cropped first.
    assert vggt_omega_input_transform((1000, 300)).ops[0] == Crop((200, 0, 800, 300))


def test_recorded_transform_includes_the_cache_downscale(tmp_path: Path) -> None:
    load = _omega_loader()
    frame = Image.fromarray(_photo(854, 480))
    pre = materialize_resize((854, 480), 600)
    path = tmp_path / "frame.png"
    frame.resize(pre.size, resample=Image.BILINEAR).save(path)
    transform = vggt_omega_input_transform((854, 480), resize_max_dim=600)
    theirs = (load([str(path)])[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(transform.apply_to_image(frame), theirs)

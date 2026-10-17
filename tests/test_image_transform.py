"""The ingest-frame -> geometry-pixels transform, and who records it.

VGGT's cameras and depth describe a resized, centre-cropped square of each frame;
these tests pin that the recorded transform reproduces the model's own loaders
pixel for pixel, maps track positions the way the pipeline always has (plus the
crop), and never squashes anything to fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.geometry.image_transform import (
    Crop,
    ImageTransform,
    MissingImageTransform,
    Pad,
    Resize,
    image_transform_for,
    materialize_resize,
    pad_to_batch,
    vggt_input_transform,
)
from reifyscene.geometry.vggt_long_wrapper import VGGTLongInitializer
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact


def _photo(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Structured + noisy RGB, with distinct top and bottom bands."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack(
        [
            (xx * 255 // max(width - 1, 1)),
            (yy * 255 // max(height - 1, 1)),
            rng.integers(0, 256, (height, width)),
        ],
        -1,
    )
    img = img.astype(np.uint8)
    img[: height // 8] = (255, 0, 0)
    img[-(height // 8) :] = (0, 0, 255)
    return img


def _to_u8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


# -- loader parity ----------------------------------------------------------------
def test_upstream_loader_parity_is_pixel_identical(tmp_path: Path) -> None:
    load_fn = pytest.importorskip("vggt.utils.load_fn")
    path = tmp_path / "frame.png"
    Image.fromarray(_photo(720, 960)).save(path)

    transform = vggt_input_transform((720, 960), loader="upstream")
    assert transform.ops == (Resize((518, 686), "bicubic"), Crop((0, 84, 518, 602)))
    assert transform.output_size == (518, 518)

    theirs = _to_u8(load_fn.load_and_preprocess_images([str(path)], mode="crop")[0])
    with Image.open(path) as img:
        ours = transform.apply_to_image(img)
    assert ours.shape == theirs.shape == (518, 518, 3)
    assert np.array_equal(ours, theirs)


def test_upstream_parity_after_the_cache_downscale(tmp_path: Path) -> None:
    load_fn = pytest.importorskip("vggt.utils.load_fn")
    frame = _photo(720, 960, seed=1)
    # What VGGTInitializer._materialize_images writes with resize_max_dim=600.
    pre = materialize_resize((720, 960), 600)
    assert pre == Resize((450, 600), "bilinear")
    cached = tmp_path / "cached.png"
    Image.fromarray(frame).resize(pre.size, resample=Image.BILINEAR).save(cached)

    transform = vggt_input_transform((720, 960), loader="upstream", resize_max_dim=600)
    theirs = _to_u8(load_fn.load_and_preprocess_images([str(cached)], mode="crop")[0])
    # From the ingest frame (every op) and from the cached file (the remaining ops).
    assert np.array_equal(transform.apply_to_image(frame), theirs)
    with Image.open(cached) as img:
        assert np.array_equal(transform.apply_to_image(img), theirs)


@pytest.mark.parametrize("size,target", [((720, 960), 252), ((960, 720), 252), ((320, 240), 126)])
def test_bounded_loader_parity_is_pixel_identical(tmp_path: Path, size, target) -> None:
    path = tmp_path / "frame.png"
    Image.fromarray(_photo(*size, seed=2)).save(path)
    init = VGGTLongInitializer({"backend_mode": "heuristic", "device": "cpu"})
    theirs = _to_u8(
        init._load_and_preprocess_images_without_upscale(
            [str(path)], mode="crop", target_size=target
        )[0]
    )

    transform = vggt_input_transform(size, loader="bounded", target_size=target)
    with Image.open(path) as img:
        ours = transform.apply_to_image(img)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_bounded_loader_crops_a_portrait_frame_like_upstream() -> None:
    # Width to the target in whole patches, height in proportion, rows centred on the target.
    transform = vggt_input_transform((720, 960), loader="bounded", target_size=252)
    assert transform.ops == (Resize((252, 336), "bicubic"), Crop((0, 42, 252, 294)))
    assert transform.output_size == (252, 252)




def test_pad_mode_is_not_guessed() -> None:
    with pytest.raises(NotImplementedError):
        vggt_input_transform((720, 960), mode="pad")


# -- points -----------------------------------------------------------------------
def test_map_points_through_the_crop() -> None:
    transform = vggt_input_transform((720, 960), loader="upstream")
    pts, inside = transform.map_points(
        np.array([[360.0, 480.0], [100.0, 50.0], [100.0, 940.0], [0.0, 117.6]])
    )
    sx, sy = 518 / 720, 686 / 960
    assert np.allclose(pts[0], [360 * sx, 480 * sy - 84])
    # the top and bottom bands are cropped away: no geometry pixel, not a border one
    assert inside.tolist() == [True, False, False, True]


def test_crop_free_mapping_is_the_old_scale_exactly() -> None:
    transform = ImageTransform((720, 960), (Resize((360, 480), "unknown"),))
    xy = np.random.default_rng(0).uniform(0, 700, size=(50, 2)).astype(np.float32)
    pts, inside = transform.map_points(xy)
    old = np.asarray(xy, np.float64) * np.array([360 / 720, 480 / 960])
    assert np.array_equal(pts, old)
    assert inside.all()


def test_map_points_from_an_intermediate_size() -> None:
    transform = vggt_input_transform((720, 960), resize_max_dim=600)
    a, _ = transform.map_points([[300.0, 400.0]])
    b, _ = transform.map_points([[300.0 * 450 / 720, 400.0 * 600 / 960]], from_size=(450, 600))
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        transform.map_points([[1.0, 1.0]], from_size=(123, 456))


# -- images -----------------------------------------------------------------------
def test_apply_to_image_dispatches_on_size_and_never_squashes() -> None:
    transform = ImageTransform((20, 40), (Crop((0, 10, 20, 30)),))
    photo = _photo(20, 40)
    out = transform.apply_to_image(photo)
    assert np.array_equal(out, photo[10:30])
    assert np.array_equal(transform.apply_to_image(out), out)  # already geometry pixels
    with pytest.raises(ValueError, match="cannot be put into the geometry's pixels"):
        transform.apply_to_image(_photo(30, 30))


def test_masks_use_nearest_and_keep_their_dtype() -> None:
    transform = ImageTransform((20, 40), (Resize((10, 20), "bicubic"), Crop((0, 5, 10, 15))))
    mask = np.zeros((40, 20), dtype=bool)
    mask[18:22, 8:12] = True
    out = transform.apply_to_image(mask, kind="mask")
    assert out.dtype == bool and out.shape == (10, 10)
    assert set(np.unique(out).tolist()) <= {False, True} and out.any()
    fmask = np.clip(mask.astype(np.float32), 0, 1)
    assert set(np.unique(transform.apply_to_image(fmask, kind="mask")).tolist()) <= {0.0, 1.0}


def test_round_trip_and_bad_record() -> None:
    transform = vggt_input_transform((720, 960), resize_max_dim=600)
    raw = json.loads(json.dumps(transform.to_dict()))
    assert ImageTransform.from_dict(raw) == transform
    raw["output_size"] = [1, 1]
    with pytest.raises(ValueError):
        ImageTransform.from_dict(raw)
    with pytest.raises(ValueError):
        ImageTransform((10, 10), (Crop((0, 0, 20, 5)),))


def test_pad_to_batch_centres_smaller_images_like_the_loader() -> None:
    transforms = pad_to_batch(
        {
            "a": vggt_input_transform((720, 960)),  # 518x518 after the crop
            "b": vggt_input_transform((960, 720)),  # landscape: 518x392, no crop
        }
    )
    assert transforms["a"].output_size == transforms["b"].output_size == (518, 518)
    assert transforms["b"].ops[-1] == Pad((518, 518), (0, 63), 255)
    pts, inside = transforms["b"].map_points([[0.0, 0.0]])
    assert np.allclose(pts, [[0.0, 63.0]]) and inside.all()


# -- records ----------------------------------------------------------------------
class _Geom:
    def __init__(self, metadata=None, depth=None):
        self.metadata = metadata or {}
        self.depthmaps_by_frame = {} if depth is None else {"0:0": {"depth": depth}}


def test_image_transform_for_reads_the_record_and_never_guesses() -> None:
    transform = vggt_input_transform((720, 960))
    geom = _Geom({"image_transforms": {"0": transform.to_dict()}}, depth=np.ones((518, 518)))
    assert image_transform_for(geom, "0") == transform
    # no record: refused, even where an identity would fit the depth
    with pytest.raises(MissingImageTransform):
        image_transform_for(_Geom(depth=np.ones((48, 64))), "0")


def _write_dataset(root: Path, size=(8, 6)) -> None:
    w, h = size
    for sub in ("camera", "depth/2x", "rgb/2x", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cam = {
        "focal_length": 10.0,
        "principal_point": [w, h],
        "image_size": [2 * w, 2 * h],
        "orientation": np.eye(3).tolist(),
        "position": [0.0, 0.0, 0.0],
    }
    (root / "camera" / "0_00000.json").write_text(json.dumps(cam))
    np.save(root / "depth" / "2x" / "0_00000.npy", np.full((h, w), 2.0, np.float32))
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(root / "rgb" / "2x" / "0_00000.png")
    (root / "splits" / "train.json").write_text(json.dumps({"camera_ids": ["0"], "time_ids": [0]}))
    (root / "extra.json").write_text(json.dumps({"fps": 30.0, "factor": 2}))  # as DyCheck ships


def test_dataset_backend_records_its_factor_resize(tmp_path: Path) -> None:
    _write_dataset(tmp_path, size=(8, 6))
    stream = VideoStreamRef(stream_id="0", video_path=tmp_path, fps=30.0, num_frames=1,
                            width=16, height=12, has_audio=False)
    ingest = type("I", (), {"streams": [stream], "keyframes": [FrameRef("0", 0, 0.0)]})()
    art = DatasetGeometryInitializer({"dataset_root": str(tmp_path), "factor": 2}).initialize(
        ingest_artifact=ingest
    )
    record = art.metadata["image_transforms"]["0"]
    assert record["recorded_by"] == "dataset" and record["source"] == "recorded"
    transform = ImageTransform.from_dict(record)
    assert (
        transform.source_size == (16, 12)
        and transform.output_size == (8, 6)
        and transform.is_pure_resize
    )
    # standalone (no ingest): the geometry's own pixels are the frame
    art2 = DatasetGeometryInitializer({"dataset_root": str(tmp_path), "factor": 2}).initialize()
    assert ImageTransform.from_dict(art2.metadata["image_transforms"]["0"]).is_identity


# -- VGGT input cache + records ---------------------------------------------------
def _video(tmp_path: Path, name: str, width=64, height=48, num_frames=3) -> Path:
    import cv2

    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), 20 * i, np.uint8))
    writer.release()
    return path


def _ingest(tmp_path: Path, video: Path, frames=(0, 1)) -> IngestArtifact:
    stream = VideoDecoder().decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="1x",
        stage_name="ingest",
        save_dir=tmp_path,
        streams=[stream],
        keyframes=[FrameRef("cam_00", i, i / 30.0) for i in frames],
    )


def test_input_cache_is_keyed_by_content_and_never_overwritten(tmp_path: Path) -> None:
    from reifyscene.geometry.vggt_wrapper import VGGTInitializer

    ingest = _ingest(tmp_path, _video(tmp_path, "a.mp4"))
    init = VGGTInitializer({"backend_mode": "heuristic", "device": "cpu"})
    root = tmp_path / "cache"
    paths_a, _ = init._materialize_images(ingest.keyframes, ingest, root)
    key_a = init._last_input_cache["key"]
    assert all(p.is_absolute() and p.exists() for p in paths_a)
    mtimes = [p.stat().st_mtime_ns for p in paths_a]

    # same inputs: same key, files reused untouched
    paths_again, notes = init._materialize_images(ingest.keyframes, ingest, root)
    assert paths_again == paths_a and [p.stat().st_mtime_ns for p in paths_a] == mtimes
    assert "0 written, 2 reused" in notes[0]

    # another resize, other frames or another input: another directory; the first stays
    init._materialize_images(ingest.keyframes, ingest, root, resize_max_dim=32)
    key_b = init._last_input_cache["key"]
    other = _ingest(tmp_path, _video(tmp_path, "b.mp4", num_frames=4), frames=(0, 2))
    init._materialize_images(other.keyframes, other, root)
    key_c = init._last_input_cache["key"]
    assert len({key_a, key_b, key_c}) == 3
    assert all(p.exists() for p in paths_a) and [p.stat().st_mtime_ns for p in paths_a] == mtimes


def test_vggt_long_records_the_loader_it_ran(tmp_path: Path) -> None:
    from tests.test_geometry_backend import _AutoTuneVGGTLong, _write_test_video

    video = tmp_path / "v.mp4"
    _write_test_video(video, width=320, height=240, num_frames=8)
    stream = VideoDecoder().decode(video, stream_id="cam_00")
    frames = [FrameRef("cam_00", i, i / 30.0) for i in range(8)]
    ingest = IngestArtifact(
        run_id="t", stage_name="ingest", save_dir=tmp_path, streams=[stream], keyframes=frames
    )
    init = _AutoTuneVGGTLong(
        {
            "backend_mode": "real",
            "device": "cuda",
            "auto_tune": True,
            "chunk_size": 8,
            "overlap": 2,
            "resize_max_dim": 256,
            "min_resize_max_dim": 128,
            "min_chunk_size": 2,
            "frame_cache_dir": f"{tmp_path.name}/cache",
            "system_report": {
                "memory": {"available_gib": 64.0},
                "gpus": [{"memory_free_mb": 8192}],
            },
        }
    )
    art = init.initialize(frames, ingest_artifact=ingest, sync_artifact=None)
    assert art.metadata["vggt_long_input_loader"] == {"loader": "bounded", "target_size": 126}
    transform = ImageTransform.from_dict(art.metadata["image_transforms"]["cam_00"])
    assert transform == vggt_input_transform(
        (320, 240), loader="bounded", target_size=126, resize_max_dim=128
    )
    assert art.metadata["input_cache"]["identity"]["resize_max_dim"] == 128
    assert all(
        Path(p).parent == Path(art.metadata["input_cache"]["dir"])
        for p in art.metadata["debug_image_paths"].values()
    )

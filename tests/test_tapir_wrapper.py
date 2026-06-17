"""Unit tests for the native point-tracker backend + all-pairs track contract.

Dataset-free and CPU-only by default: the all-pairs format math, the extended
``TrackArtifact`` round-trip, and the wrapper's checkpoint-resolution logic. The
real model forward (GPU + checkpoint) is exercised by ``scripts/compute_tracks_native.py``
and the render-proof ``--own-tracker`` path; one forward test here is gated and
skips when CUDA or the checkpoint is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from room4d.tracking.all_pairs import AllPairsTracks
from room4d.tracking.tapir_wrapper import TAPIRWrapper
from room4d.tracking.track_artifact import TrackArtifact


def _ragged_all_pairs(seed: int = 0) -> AllPairsTracks:
    names = ["000", "001", "002"]
    t = len(names)
    rng = np.random.default_rng(seed)
    tracks = [rng.standard_normal((n, t, 4)).astype(np.float32) for n in (5, 3, 4)]
    return AllPairsTracks(frame_names=names, tracks=tracks, factor=1, image_hw=(48, 64))


# -- AllPairsTracks format -------------------------------------------------

def test_all_pairs_query_counts_and_shape_validation() -> None:
    ap = _ragged_all_pairs()
    assert ap.query_counts() == [5, 3, 4]
    assert ap.num_frames == 3
    # frame_names / tracks length mismatch
    with pytest.raises(ValueError):
        AllPairsTracks(frame_names=["a", "b"], tracks=[np.zeros((2, 2, 4), np.float32)], factor=1, image_hw=(8, 8))
    # wrong per-query shape (T or channels off)
    with pytest.raises(ValueError):
        AllPairsTracks(frame_names=["a"], tracks=[np.zeros((2, 1, 3), np.float32)], factor=1, image_hw=(8, 8))


def test_all_pairs_write_filenames_and_slicing(tmp_path: Path) -> None:
    ap = _ragged_all_pairs()
    out = ap.write(tmp_path / "1x")
    files = sorted(p.name for p in out.glob("*.npy"))
    assert len(files) == ap.num_frames ** 2  # all ordered pairs
    # file <name_i>_<name_j>.npy must equal tracks[i][:, j], shape (N_i, 4)
    assert np.array_equal(np.load(out / "000_002.npy"), ap.tracks[0][:, 2])
    assert np.load(out / "001_000.npy").shape == (3, 4)


def test_all_pairs_roundtrip(tmp_path: Path) -> None:
    ap = _ragged_all_pairs()
    ap.write(tmp_path / "1x")
    back = AllPairsTracks.load(tmp_path / "1x", ap.frame_names, factor=1, image_hw=(48, 64))
    assert back.frame_names == ap.frame_names and back.factor == 1 and back.image_hw == (48, 64)
    assert all(np.array_equal(a, b) for a, b in zip(ap.tracks, back.tracks))


# -- TrackArtifact contract extension --------------------------------------

def test_track_artifact_all_pairs_save_load(tmp_path: Path) -> None:
    ap = _ragged_all_pairs(seed=1)
    art = TrackArtifact(
        run_id="t", stage_name="tracking", save_dir=tmp_path / "art",
        all_pairs_by_stream={"cam0": ap},
    )
    art.save()
    re = TrackArtifact.load(tmp_path / "art")
    rap = re.all_pairs_by_stream["cam0"]
    assert rap.frame_names == ap.frame_names and rap.factor == 1 and rap.image_hw == (48, 64)
    assert all(np.array_equal(a, b) for a, b in zip(ap.tracks, rap.tracks))


def test_track_artifact_empty_all_pairs_is_unaffected(tmp_path: Path) -> None:
    # The CoTracker/scaffold path carries no all-pairs; save/load must not break.
    art = TrackArtifact(run_id="t", stage_name="tracking", save_dir=tmp_path / "art")
    art.save()
    re = TrackArtifact.load(tmp_path / "art")
    assert re.all_pairs_by_stream == {}


def test_write_2d_tracks_stream_selection(tmp_path: Path) -> None:
    ap = _ragged_all_pairs()
    art = TrackArtifact(
        run_id="t", stage_name="tracking", save_dir=tmp_path / "art",
        all_pairs_by_stream={"cam0": ap},
    )
    out = art.write_2d_tracks(tmp_path / "native" / "1x")  # single stream -> id optional
    assert (out / "002_001.npy").exists()
    # multi-stream requires an explicit id
    art.all_pairs_by_stream["cam1"] = _ragged_all_pairs(seed=2)
    with pytest.raises(ValueError):
        art.write_2d_tracks(tmp_path / "ambiguous")
    assert art.write_2d_tracks(tmp_path / "cam1" / "1x", stream_id="cam1").exists()


# -- wrapper config / checkpoint resolution --------------------------------

def test_checkpoint_path_resolution(tmp_path: Path) -> None:
    w = TAPIRWrapper({"checkpoint_dir": str(tmp_path), "model_type": "bootstapir"})
    assert w.checkpoint_path() == tmp_path / "bootstapir_checkpoint_v2.pt"
    w2 = TAPIRWrapper({"checkpoint_dir": str(tmp_path), "model_type": "tapir"})
    assert w2.checkpoint_path() == tmp_path / "tapir_checkpoint_panning.pt"
    explicit = tmp_path / "custom.pt"
    w3 = TAPIRWrapper({"checkpoint_path": str(explicit)})
    assert w3.checkpoint_path() == explicit


def test_unknown_model_type_raises() -> None:
    with pytest.raises(ValueError):
        TAPIRWrapper({"model_type": "nope"})


def test_default_config_values() -> None:
    w = TAPIRWrapper()
    assert w.model_type == "bootstapir"
    assert w.grid_size == 4  # SoM's value for the shipped DyCheck tracks
    assert (w.resize_height, w.resize_width) == (256, 256)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    w = TAPIRWrapper({"checkpoint_dir": str(tmp_path), "device": "cpu"})
    with pytest.raises(FileNotFoundError):
        w.compute_all_pairs(
            np.zeros((2, 16, 16, 3), np.uint8), np.ones((2, 16, 16), bool), ["a", "b"]
        )


# -- gated real forward (GPU + checkpoint) ---------------------------------

def test_compute_all_pairs_forward() -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("tapnet forward needs a CUDA device")
    w = TAPIRWrapper({"grid_size": 16})
    if not w.checkpoint_path().exists():
        pytest.skip(f"checkpoint not present: {w.checkpoint_path()}")

    t, h, win = 3, 96, 64
    rng = np.random.default_rng(0)
    frames = (rng.random((t, h, win, 3)) * 255).astype(np.uint8)
    masks = np.zeros((t, h, win), bool)
    masks[:, 20:80, 16:48] = True  # foreground box
    ap = w.compute_all_pairs(frames, masks, ["a", "b", "c"], factor=1)

    assert ap.num_frames == 3
    assert ap.tracks[0].shape[1:] == (3, 4)  # (N_0, T, 4)
    # diagonal slot holds exact integer grid query coords inside the FG box
    diag = ap.tracks[0][:, 0, :2]
    assert np.all(diag[:, 0] % w.grid_size == 0)
    assert diag[:, 0].min() >= 16 and diag[:, 1].min() >= 20
    # occ / dist logits are finite
    assert np.isfinite(ap.tracks[0][:, :, 2:]).all()

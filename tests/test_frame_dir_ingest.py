"""Tests for lossless frame-directory ingest (the FP6b-hardening ingest path).

A directory of image frames is ingested directly (no lossy re-encode), with the camera id
+ frame indices parsed from filenames so dataset ground-truth (`{cam}_{idx}.ext`) aligns.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from room4d.ingest.decoder import VideoDecoder, _parse_frame_name


def _write_frames(dir_path, names, *, h=8, w=6):
    """Write distinct solid-color frames; return {name: rgb}."""
    dir_path.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, name in enumerate(names):
        rgb = np.full((h, w, 3), (i * 7) % 256, dtype=np.uint8)
        rgb[0, 0] = [i % 256, (2 * i) % 256, (3 * i) % 256]  # make each frame unique
        cv2.imwrite(str(dir_path / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out[name] = rgb
    return out


def test_parse_frame_name():
    assert _parse_frame_name("0_00060") == ("0", 60)
    assert _parse_frame_name("cam01_000060") == ("cam01", 60)
    assert _parse_frame_name("000060") == (None, 60)
    assert _parse_frame_name("frame") == (None, None)


def test_decode_frame_dir_parses_camera_and_indices(tmp_path):
    _write_frames(tmp_path / "rgb", ["0_00000", "0_00001", "0_00002"])
    s = VideoDecoder().decode(tmp_path / "rgb")
    assert s.stream_id == "0"                       # camera parsed from filenames
    assert s.num_frames == 3 and (s.width, s.height) == (6, 8)
    assert s.metadata["source"] == "frames"
    assert set(s.metadata["frame_index"]) == {"0", "1", "2"}


def test_get_frame_is_lossless(tmp_path):
    frames = _write_frames(tmp_path / "rgb", ["0_00000", "0_00005", "0_00010"])
    d = VideoDecoder()
    s = d.decode(tmp_path / "rgb")
    # frame_idx 5 maps to the "0_00005" file exactly (not position 1)
    got = d.get_frame(s, 5)
    assert np.array_equal(got, frames["0_00005"])   # exact, lossless


def test_get_frame_missing_index_raises(tmp_path):
    _write_frames(tmp_path / "rgb", ["0_00000", "0_00010"])
    d = VideoDecoder()
    s = d.decode(tmp_path / "rgb")
    with pytest.raises(IndexError):
        d.get_frame(s, 3)                            # 3 not in {0, 10}


def test_iter_frames_in_index_order(tmp_path):
    _write_frames(tmp_path / "rgb", ["0_00002", "0_00000", "0_00001"])  # unsorted on disk
    d = VideoDecoder()
    s = d.decode(tmp_path / "rgb")
    assert len(list(d.iter_frames(s))) == 3


def test_no_camera_prefix_defaults(tmp_path):
    _write_frames(tmp_path / "rgb", ["000000", "000001"])
    s = VideoDecoder().decode(tmp_path / "rgb", stream_id="cam_00")
    assert s.stream_id == "cam_00"                   # fallback to passed id when no prefix
    assert set(s.metadata["frame_index"]) == {"0", "1"}


def test_multi_camera_dir_single_camera_scope(tmp_path):
    _write_frames(tmp_path / "rgb", ["0_00000", "0_00001", "1_00000"])
    s = VideoDecoder().decode(tmp_path / "rgb", stream_id="1")
    assert s.stream_id == "1"                        # target camera selected
    assert set(s.metadata["frame_index"]) == {"0"}   # only camera 1's frames


def test_empty_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        VideoDecoder().decode(tmp_path / "empty")


def test_preprocess_ingests_frame_dir(tmp_path):
    from room4d.api.pipeline import Room4DPipeline
    _write_frames(tmp_path / "rgb", [f"0_{i:05d}" for i in range(4)], h=16, w=12)
    pipe = Room4DPipeline({"run_dir": tmp_path / "run"})
    ingest = pipe.preprocess([str(tmp_path / "rgb")])
    assert ingest.num_streams() == 1
    assert ingest.streams[0].stream_id == "0"        # parsed camera id flows through ingest
    assert ingest.streams[0].metadata["source"] == "frames"

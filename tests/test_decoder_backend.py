"""Tests for the real OpenCV video decoder backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from room4d.ingest.decoder import VideoDecoder
from tests.conftest import write_test_video


def test_decoder_probes_and_reads_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_test_video(video_path)

    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id="cam_a")

    assert stream.metadata["probed"] is True
    assert stream.width == 32
    assert stream.height == 24
    assert stream.num_frames >= 5
    assert stream.fps > 0

    frame = decoder.get_frame(stream, 0)
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8

    frames = list(decoder.iter_frames(stream, stride=2))
    assert len(frames) >= 2


def test_decoder_raises_for_missing_video(tmp_path: Path) -> None:
    decoder = VideoDecoder(default_fps=30.0)
    missing = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        decoder.decode(missing, stream_id="cam_missing")

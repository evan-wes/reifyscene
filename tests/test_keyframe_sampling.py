"""Tests for MultiVideoDataset.sample_keyframes, incl. the --keyframes-all path."""

from __future__ import annotations

from pathlib import Path

import pytest

from reifyscene.common.types import VideoStreamRef
from reifyscene.data.dataset import MultiVideoDataset


def _stream(stream_id: str = "0", num_frames: int = 100,
            fps: float | None = 30.0) -> VideoStreamRef:
    return VideoStreamRef(stream_id, Path("clip.mp4"), fps, num_frames, 128, 128, False)


def test_stride_and_cap_default() -> None:
    ds = MultiVideoDataset([_stream(num_frames=100, fps=30.0)])
    # 1.0s stride at 30fps -> every 30th frame; capped at 4
    kf = ds.sample_keyframes(stride_sec=1.0, max_frames_per_stream=4)
    assert [f.frame_idx for f in kf] == [0, 30, 60, 90]


def test_all_frames_emits_every_frame_and_ignores_stride_and_cap() -> None:
    ds = MultiVideoDataset([_stream(num_frames=100, fps=30.0)])
    kf = ds.sample_keyframes(stride_sec=1.0, max_frames_per_stream=4, all_frames=True)
    assert [f.frame_idx for f in kf] == list(range(100))  # every frame, cap ignored
    assert kf[1].timestamp_sec == 1 / 30.0  # timestamps still derived from fps


def test_all_frames_multi_stream() -> None:
    ds = MultiVideoDataset([_stream("0", 5, 30.0), _stream("1", 3, 30.0)])
    kf = ds.sample_keyframes(max_frames_per_stream=2, all_frames=True)
    by_stream = {"0": [], "1": []}
    for f in kf:
        by_stream[f.stream_id].append(f.frame_idx)
    assert by_stream == {"0": [0, 1, 2, 3, 4], "1": [0, 1, 2]}


def test_a_stream_with_no_frames_is_refused_not_given_a_frame_0() -> None:
    ds = MultiVideoDataset([_stream(num_frames=0)])
    with pytest.raises(ValueError, match="no frames"):
        ds.sample_keyframes(all_frames=True)


def test_keyframes_are_the_streams_own_frame_numbers() -> None:
    """A DyCheck folder's frames are numbered by their names, not 0..n-1."""
    s = _stream(num_frames=3, fps=None)
    s.metadata = {"frame_index": {"60": "a", "61": "b", "62": "c"}}
    kf = MultiVideoDataset([s]).sample_keyframes(stride=2)
    assert [f.frame_idx for f in kf] == [60, 62]
    assert all(f.timestamp_sec is None for f in kf)


def test_a_stride_in_seconds_needs_a_recorded_frame_rate() -> None:
    from reifyscene.common.types import UnknownFrameRate

    with pytest.raises(UnknownFrameRate, match="give it in frames"):
        MultiVideoDataset([_stream(num_frames=10, fps=None)]).sample_keyframes(stride_sec=1.0)


def test_keyframes_all_changes_the_ingest_fingerprint_only_when_on() -> None:
    """keyframes_all off fingerprints like the strided default; on, the stride and cap
    select nothing, so only the flag is fingerprinted."""
    from reifyscene.api.contracts import effective_settings

    base = {"max_keyframes_per_stream": 4, "keyframe_stride_sec": 1.0}
    default = effective_settings("ingest", base)
    assert effective_settings("ingest", {**base, "keyframes_all": False}) == default

    all_frames = effective_settings("ingest", {**base, "keyframes_all": True})
    assert all_frames["keyframes_all"] is True
    assert "max_keyframes_per_stream" not in all_frames and "keyframe_stride_sec" not in all_frames
    assert all_frames != default

"""Tests for MultiVideoDataset.sample_keyframes, incl. the --keyframes-all path."""

from __future__ import annotations

from pathlib import Path

from reifyscene.common.types import VideoStreamRef
from reifyscene.data.dataset import MultiVideoDataset


def _stream(stream_id: str = "0", num_frames: int = 100, fps: float = 30.0) -> VideoStreamRef:
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


def test_zero_frame_stream_still_emits_frame0() -> None:
    ds = MultiVideoDataset([_stream(num_frames=0)])
    assert [f.frame_idx for f in ds.sample_keyframes(all_frames=True)] == [0]


def test_keyframes_all_resume_signature_is_backward_compatible() -> None:
    """Default (strided) signature stays identical to pre-flag artifacts; only an
    actual all-frames run gets a distinct signature that invalidates them."""
    from reifyscene.api.resume import _stage_cfg_signature

    base = {"max_keyframes_per_stream": 4, "keyframe_stride_sec": 1.0}
    default_sig = _stage_cfg_signature(base, "ingest")
    assert "keyframes_all" not in default_sig  # byte-identical to pre-flag signature
    assert _stage_cfg_signature({**base, "keyframes_all": False}, "ingest") == default_sig

    all_sig = _stage_cfg_signature({**base, "keyframes_all": True}, "ingest")
    assert all_sig.get("keyframes_all") is True
    assert all_sig != default_sig

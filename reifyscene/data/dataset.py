"""Dataset entrypoints for multi-video sessions."""

from __future__ import annotations

from reifyscene.common.types import FrameRef, VideoStreamRef


class MultiVideoDataset:
    """Container for a set of synchronized or unsynchronized video streams.

    The dataset abstraction is intentionally thin in the scaffold. It only owns
    frame-reference sampling logic and leaves actual pixel loading to the
    ingest/decoder layer.
    """

    def __init__(self, streams: list[VideoStreamRef]) -> None:
        """Store stream references for future sampling."""
        self.streams = streams

    def sample_keyframes(
        self,
        stride: int | None = None,
        stride_sec: float | None = None,
        max_frames_per_stream: int | None = 8,
        all_frames: bool = False,
    ) -> list[FrameRef]:
        """Sample coarse keyframes for geometry initialization.

        Keyframes are every ``stride``-th of a stream's own frames (``stride_sec`` converts
        with the recorded frame rate, and raises on a stream that records none), capped at
        ``max_frames_per_stream``. When ``all_frames`` is True, every frame of every stream is
        a keyframe -- the dense path for benchmark-protocol parity (train on the full
        sequence). A stream with no frames is refused, not given a frame 0.
        """
        keyframes: list[FrameRef] = []
        for stream in self.streams:
            numbers = stream.frame_numbers
            if not numbers:
                raise ValueError(f"stream {stream.stream_id!r} has no frames to sample")
            if not all_frames:
                step = stream.stride_frames(frames=stride, seconds=stride_sec,
                                            setting="keyframe stride")
                numbers = numbers[::step]
                if max_frames_per_stream is not None:
                    numbers = numbers[:max_frames_per_stream]
            keyframes.extend(stream.frame_ref(i) for i in numbers)
        return keyframes

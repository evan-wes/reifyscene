"""Dataset entrypoints for multi-video sessions."""

from __future__ import annotations

from math import ceil

from room4d.common.types import FrameRef, VideoStreamRef


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
        stride_sec: float = 1.0,
        max_frames_per_stream: int | None = 8,
    ) -> list[FrameRef]:
        """Sample coarse keyframes for geometry initialization.

        Notes
        -----
        If a stream does not yet expose frame count metadata, a single frame-0
        reference is emitted so the rest of the pipeline can still run.
        """
        keyframes: list[FrameRef] = []
        for stream in self.streams:
            if stream.num_frames <= 0:
                keyframes.append(FrameRef(stream_id=stream.stream_id, frame_idx=0, timestamp_sec=0.0))
                continue

            stride_frames = max(int(round(stride_sec * stream.fps)), 1)
            frame_indices = list(range(0, stream.num_frames, stride_frames))
            if max_frames_per_stream is not None:
                frame_indices = frame_indices[:max_frames_per_stream]
            for frame_idx in frame_indices:
                keyframes.append(
                    FrameRef(
                        stream_id=stream.stream_id,
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / max(stream.fps, 1e-6),
                    )
                )
        return keyframes

    def sample_dense_window(
        self,
        start_sec: float,
        end_sec: float,
        fps: float | None = None,
    ) -> list[FrameRef]:
        """Sample all streams in a dense temporal window."""
        window_refs: list[FrameRef] = []
        for stream in self.streams:
            sample_fps = fps or stream.fps or 1.0
            frame_start = max(int(start_sec * sample_fps), 0)
            frame_end = max(int(ceil(end_sec * sample_fps)), frame_start + 1)
            for frame_idx in range(frame_start, frame_end):
                window_refs.append(
                    FrameRef(
                        stream_id=stream.stream_id,
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / max(sample_fps, 1e-6),
                    )
                )
        return window_refs

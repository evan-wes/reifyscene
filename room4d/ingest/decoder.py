"""Video decoding entrypoints.

This module provides a real OpenCV-backed decoder for local video files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from room4d.common.types import VideoStreamRef


class VideoDecoder:
    """Decode videos and access frame content.

    Implementation order after this slice:
    1. Add a small LRU cache for random-access frame reads.
    2. Add ffprobe-based audio and codec probing.
    3. Add batch decode helpers for frame windows.
    4. Add optional frame extraction cache for repeated downstream use.
    """

    def __init__(self, default_fps: float = 30.0) -> None:
        """Configure the OpenCV-backed decoder.

        Parameters
        ----------
        default_fps:
            Used when container metadata is missing or invalid.
        """
        self.default_fps = default_fps

    def _probe_capture(self, path: Path) -> dict[str, float | int | bool | str]:
        """Probe a video container with OpenCV.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        RuntimeError
            If OpenCV cannot open the file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Video file does not exist: '{path}'.")

        capture = cv2.VideoCapture(str(path))
        opened = bool(capture.isOpened())
        if not opened:
            capture.release()
            raise RuntimeError(
                f"Could not open video stream '{path}'. Verify the file exists and OpenCV can decode it."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        return {
            "fps": fps if fps > 0 else self.default_fps,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "has_audio": False,
            "decoder": "opencv",
            "probed": True,
            "exists": True,
        }

    def decode(self, video_path: str | Path, stream_id: str | None = None) -> VideoStreamRef:
        """Create a stream reference for a video path.

        Parameters
        ----------
        video_path:
            Local path to a decodable video file.
        stream_id:
            Optional explicit stream identifier.
        """
        path = Path(video_path)
        meta = self._probe_capture(path)
        return VideoStreamRef(
            stream_id=stream_id or path.stem,
            video_path=path,
            fps=float(meta["fps"]),
            num_frames=int(meta["num_frames"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
            has_audio=bool(meta["has_audio"]),
            metadata={
                "decoder": str(meta["decoder"]),
                "probed": bool(meta["probed"]),
                "exists": bool(meta["exists"]),
                "opened": True,
            },
        )

    def decode_many(self, video_paths: list[str | Path]) -> list[VideoStreamRef]:
        """Decode a list of videos into stream references."""
        return [self.decode(path, stream_id=f"cam_{idx:02d}") for idx, path in enumerate(video_paths)]

    def _open_stream(self, stream: VideoStreamRef) -> cv2.VideoCapture:
        """Open a stream for frame access."""
        capture = cv2.VideoCapture(str(stream.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open video stream '{stream.video_path}'. "
                "Verify the file exists and OpenCV can decode it."
            )
        return capture

    def get_frame(self, stream: VideoStreamRef, frame_idx: int) -> np.ndarray:
        """Return a single frame as an RGB image array.

        Parameters
        ----------
        stream:
            Video stream reference returned by :meth:`decode`.
        frame_idx:
            Zero-based frame index.
        """
        if frame_idx < 0:
            raise ValueError(f"frame_idx must be non-negative, got {frame_idx}.")

        capture = self._open_stream(stream)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = capture.read()
        finally:
            capture.release()

        if not ok or frame_bgr is None:
            raise IndexError(
                f"Unable to decode frame {frame_idx} from stream '{stream.stream_id}'."
            )

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def iter_frames(self, stream: VideoStreamRef, stride: int = 1) -> Iterator[np.ndarray]:
        """Yield RGB frames from a stream.

        Parameters
        ----------
        stream:
            Video stream reference.
        stride:
            Yield every ``stride`` frames.
        """
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")

        capture = self._open_stream(stream)
        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    break
                if frame_idx % stride == 0:
                    yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_idx += 1
        finally:
            capture.release()

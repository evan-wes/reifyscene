"""Video decoding entrypoints.

This module provides a real OpenCV-backed decoder for local video files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from reifyscene.common.types import VideoStreamRef

# Image extensions recognized when ingesting a frame directory directly (lossless path).
_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _parse_frame_name(stem: str) -> tuple[str | None, int | None]:
    """Parse a frame filename stem into ``(camera_id, frame_idx)``.

    Handles the shape_of_motion naming ``"0_00060"`` → ``("0", 60)`` and a bare
    ``"000060"`` → ``(None, 60)``. Returns ``(None, None)`` when no trailing index
    is present (caller falls back to file position).
    """
    m = re.match(r"^(?P<cam>.+)_(?P<idx>\d+)$", stem)
    if m:
        return m.group("cam"), int(m.group("idx"))
    m = re.search(r"(\d+)$", stem)
    if m:
        return None, int(m.group(1))
    return None, None


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
        """Create a stream reference for a video path **or a frame directory**.

        A **directory** is ingested as a lossless frame sequence (no re-encode) — the
        original frames feed SfM/geometry directly, and the ``stream_id`` + frame indices
        are parsed from the filenames so dataset ground-truth (``{cam}_{idx}.ext``) aligns.
        A **file** is decoded as a video, as before.

        Parameters
        ----------
        video_path:
            Local path to a decodable video file, or a directory of image frames.
        stream_id:
            Optional explicit stream identifier (a frame directory prefers the camera id
            parsed from filenames; this is the fallback when filenames carry no camera).
        """
        # Absolute, so the stream (and every frame path in it) still resolves when a later
        # stage or a resume runs from another working directory.
        path = Path(video_path).expanduser().resolve()
        if path.is_dir():
            return self._decode_frame_dir(path, stream_id)
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

    def _decode_frame_dir(self, path: Path, stream_id: str | None = None) -> VideoStreamRef:
        """Build a lossless frame-sequence stream from a directory of images.

        Frames are keyed by their parsed index (so ``0_00060.png`` is reachable as
        ``frame_idx=60``, matching dataset ground-truth). Single-camera scope: if the
        directory mixes multiple camera prefixes, the target camera is ``stream_id`` (when
        it matches) else the lexicographically-first, and only its frames are used (a
        multi-camera directory → multiple streams is deferred).
        """
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in _FRAME_EXTS)
        if not files:
            raise FileNotFoundError(f"No image frames ({', '.join(_FRAME_EXTS)}) found in '{path}'.")
        parsed = [(*_parse_frame_name(f.stem), f) for f in files]  # (cam, idx, path)
        cams = sorted({cam for cam, _, _ in parsed if cam is not None})
        if len(cams) > 1:
            if stream_id not in cams:
                raise ValueError(
                    f"'{path}' holds frames of {len(cams)} cameras {cams}: say which one "
                    "(stream_id) -- the first alphabetically is not a default")
            target = stream_id
            parsed = [t for t in parsed if t[0] == target]
        elif cams:
            target = cams[0]
        else:
            target = stream_id or "0"  # no camera prefix in filenames
        frame_index: dict[str, str] = {}
        for pos, (_, idx, f) in enumerate(sorted(parsed, key=lambda t: (t[1] if t[1] is not None else 0, t[2].name))):
            frame_index[str(idx if idx is not None else pos)] = str(f)
        probe = cv2.imread(str(files[0]))
        if probe is None:
            raise RuntimeError(f"OpenCV could not read frame '{files[0]}'.")
        height, width = probe.shape[:2]
        return VideoStreamRef(
            stream_id=str(target),
            video_path=path,
            fps=self.default_fps,
            num_frames=len(frame_index),
            width=int(width),
            height=int(height),
            has_audio=False,
            metadata={"decoder": "frames", "source": "frames", "frame_index": frame_index,
                      "num_frames": len(frame_index), "opened": True, "exists": True},
        )

    def decode_many(self, video_paths: list[str | Path], *,
                    frames_camera: str | None = None) -> list[VideoStreamRef]:
        """Decode a list of videos (or frame directories) into stream references.

        ``frames_camera`` picks the camera of a frames folder that holds several (DyCheck's
        ``rgb/1x`` holds ``0_``, ``1_`` and ``2_`` frames); a folder of several cameras without
        it is refused rather than given its first camera.
        """
        return [self.decode(path, stream_id=self.requested_stream_id(idx, frames_camera))
                for idx, path in enumerate(video_paths)]

    @staticmethod
    def requested_stream_id(idx: int, frames_camera: str | None = None) -> str:
        """The stream id ingest asks for the ``idx``-th input: the chosen camera of a frames
        folder, else ``cam_<idx>``. Anything that must agree with ingest calls this."""
        return frames_camera or f"cam_{idx:02d}"

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

        if (stream.metadata or {}).get("source") == "frames":
            return self._read_frame_file(stream, int(frame_idx))

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

    def _read_frame_file(self, stream: VideoStreamRef, frame_idx: int) -> np.ndarray:
        """Load one frame of a frame-sequence stream (lossless) as RGB."""
        frame_index = (stream.metadata or {}).get("frame_index", {})
        file_path = frame_index.get(str(frame_idx))
        if file_path is None:
            raise IndexError(f"Frame {frame_idx} is not present in frame-sequence stream '{stream.stream_id}'.")
        img = cv2.imread(str(file_path))
        if img is None:
            raise IndexError(f"Unable to read frame file '{file_path}' for stream '{stream.stream_id}'.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

        if (stream.metadata or {}).get("source") == "frames":
            ordered = sorted((int(k) for k in stream.metadata.get("frame_index", {})))
            for pos, idx in enumerate(ordered):
                if pos % stride == 0:
                    yield self._read_frame_file(stream, idx)
            return

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

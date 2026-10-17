"""Decode one frame of a stream -- a ladder of backends, heavy ones optional.

A stream's frames come either from a folder of image files or from a video container:

- **images**: ``torchvision.io.decode_image`` (core torchvision), else PIL.
- **video**: torchcodec when installed (frame-accurate by index, fps from the container), else
  PyAV when installed, else OpenCV -- always present, but its index seek
  (``CAP_PROP_POS_FRAMES``) is not reliable on every codec, so the OpenCV rung checks where each
  seek landed and reads forward from the start when it did not land on the frame asked for.
  It never returns a neighbouring frame in place of the one requested.

Every rung returns an RGB ``uint8`` tensor ``(3, H, W)``. :func:`video_backend` says which rung
serves this machine, and a stream records it, so a result can be traced to its decoder.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


class FrameNotFound(IndexError):
    """The stream has no such frame, or the decoder could not produce it."""


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# -- images ---------------------------------------------------------------------------------------

def read_image(path: Path) -> torch.Tensor:
    """An image file as RGB ``uint8`` ``(3, H, W)``."""
    path = Path(path)
    if not path.is_file():
        raise FrameNotFound(f"no image file at {path}")
    try:
        from torchvision.io import ImageReadMode, decode_image, read_file

        return decode_image(read_file(str(path)), mode=ImageReadMode.RGB)
    except (ImportError, RuntimeError):
        from PIL import Image

        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        return torch.from_numpy(arr.copy()).permute(2, 0, 1)


def image_backend() -> str:
    return "torchvision" if _available("torchvision") else "pil"


# -- video ----------------------------------------------------------------------------------------

def video_backend() -> str:
    """The rung this machine decodes video with."""
    for name in ("torchcodec", "av"):
        if _available(name):
            return {"av": "pyav"}.get(name, name)
    return "opencv"


def video_fps(path: Path) -> float | None:
    """The container's frame rate, or None when it does not record one. Never a default."""
    backend = video_backend()
    try:
        if backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder

            fps = VideoDecoder(str(path)).metadata.average_fps
        elif backend == "pyav":
            import av

            with av.open(str(path)) as c:
                rate = c.streams.video[0].average_rate
                fps = float(rate) if rate else None
        else:
            import cv2

            cap = cv2.VideoCapture(str(path))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else None
            cap.release()
    except Exception:  # noqa: BLE001 -- an unreadable rate is an unknown rate
        return None
    return float(fps) if fps and fps > 0 else None


def read_video_frame(path: Path, index: int) -> torch.Tensor:
    """Frame ``index`` (0-based, in decode order) of a video, RGB ``uint8`` ``(3, H, W)``."""
    path = Path(path)
    if not path.is_file():
        raise FrameNotFound(f"no video file at {path}")
    backend = video_backend()
    if backend == "torchcodec":
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(path))
        if not 0 <= index < len(dec):
            raise FrameNotFound(f"{path.name} has {len(dec)} frames; asked for {index}")
        return dec[index]
    if backend == "pyav":
        return _pyav_frame(path, index)
    return _opencv_frame(path, index)


def _pyav_frame(path: Path, index: int) -> torch.Tensor:
    import av

    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i == index:
                return torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
    raise FrameNotFound(f"{path.name} ended before frame {index}")


def _opencv_frame(path: Path, index: int) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FrameNotFound(f"OpenCV cannot open {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, bgr = cap.read()
        if not ok or landed != index:
            # the seek did not land where asked: read forward from the start instead
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bgr = None
            for i in range(index + 1):
                ok, img = cap.read()
                if not ok:
                    raise FrameNotFound(f"{path.name} ended before frame {index}")
                if i == index:
                    bgr = img
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return torch.from_numpy(rgb).permute(2, 0, 1)

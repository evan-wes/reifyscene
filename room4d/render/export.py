"""Scene / video export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class SceneExporter:
    """Export renderable scene artifacts."""

    def export(self, render_artifact, out_path: str) -> None:
        """Write placeholder export output."""
        return None


def write_video(frames: Iterable[np.ndarray], out_path: str | Path, fps: int = 24) -> Path:
    """Encode RGB uint8 frames (H, W, 3) to an mp4 via OpenCV.

    imageio/imageio-ffmpeg aren't in the env; cv2 is. Frames are converted
    RGB->BGR and written with the mp4v codec.
    """
    import cv2

    frames = [np.clip(np.asarray(f), 0, 255).astype(np.uint8) for f in frames]
    if not frames:
        raise ValueError("write_video got no frames.")
    height, width = frames[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out_path

"""Test configuration for source-tree imports and shared helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_test_video(
    path: Path,
    *,
    width: int = 32,
    height: int = 24,
    num_frames: int = 5,
    fps: float = 10.0,
    color_offset: int = 0,
) -> None:
    """Create a tiny MP4 for integration tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to create test video at {path}."
    for idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (idx * 10 + color_offset) % 255
        frame[..., 1] = (50 + color_offset) % 255
        frame[..., 2] = 100
        writer.write(frame)
    writer.release()


def write_ingest(run_dir: Path, *, stream_id: str = "cam_00", video_path: Path | None = None,
                 num_frames: int, width: int = 0, height: int = 0,
                 frame_index: dict[str, str] | None = None, fps: float = 30.0) -> Path:
    """A real ingest artifact for a fixture run, saved the way the ingest stage saves one."""
    from reifyscene.common.types import VideoStreamRef
    from reifyscene.ingest.ingest_artifact import IngestArtifact

    metadata = {"source": "frames", "frame_index": dict(frame_index)} if frame_index else {}
    stream = VideoStreamRef(stream_id=stream_id, video_path=Path(video_path or run_dir / "clip"),
                            fps=fps, num_frames=num_frames, width=width, height=height,
                            has_audio=False, metadata=metadata)
    IngestArtifact(run_id=Path(run_dir).name, stage_name="ingest",
                   save_dir=Path(run_dir) / "ingest", streams=[stream]).save()
    return Path(run_dir) / "ingest"

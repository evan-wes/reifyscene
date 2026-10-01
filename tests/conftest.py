"""Test configuration for source-tree imports and shared helpers."""

from __future__ import annotations

import json
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


def write_manifest(run_dir: Path, config: dict | None = None) -> Path:
    """A run manifest, which is what makes a folder a run (a pipeline run writes it first)."""
    from reifyscene.api.run_outputs import RunOutputs

    # as the run script writes it: nothing records where the run itself is
    config = {k: v for k, v in (config or {}).items() if k != "run_dir"}
    return RunOutputs.create(run_dir).write_manifest({"video_paths": [], "config": config})


def write_checkpoint(run_dir: Path, payload: bytes = b"weights", *, name: str = "render",
                     kind: str = "dynamic", weights: str = "trained_dynamic.npz",
                     recipe: dict | None = None):
    """A checkpoint folder as the render stage leaves one: its weights, and the render record
    naming them and their kind. Returns the run's ``Checkpoint``."""
    from reifyscene.api.run_outputs import RunOutputs

    ckpt_dir = Path(run_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / weights).write_bytes(payload)
    (ckpt_dir / "artifact.json").write_text(
        json.dumps({"scene_state": {"kind": kind, "checkpoint": weights,
                                    **({"recipe": recipe} if recipe is not None else {})}}))
    return RunOutputs.open(run_dir).checkpoint(name)


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
    if not (Path(run_dir) / "run_manifest.json").is_file():
        write_manifest(run_dir)
    IngestArtifact(run_id=Path(run_dir).name, stage_name="ingest",
                   save_dir=Path(run_dir) / "ingest", streams=[stream]).save()
    return Path(run_dir) / "ingest"

"""Frame sampling helpers for calibration mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from reifyscene.common.io import write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.ingest.decoder import VideoDecoder


def _variance_of_laplacian(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


class CalibrationFrameSampler:
    """Sample relatively static frames from one or more input streams."""

    def __init__(self, cfg: dict[str, Any] | None = None, *, decoder: VideoDecoder | None = None) -> None:
        self.cfg = cfg or {}
        self.decoder = decoder or VideoDecoder(default_fps=float(self.cfg.get("default_fps", 30.0)))

    def _candidate_indices(self, stream: VideoStreamRef) -> list[int]:
        spacing_sec = float(self.cfg.get("sample_spacing_sec", 0.75))
        stride = max(1, int(round(stream.fps * max(spacing_sec, 1e-3))))
        indices = list(range(0, max(stream.num_frames, 1), stride))
        max_frames = self.cfg.get("max_frames_per_stream")
        if max_frames is not None and len(indices) > int(max_frames):
            keep = np.linspace(0, len(indices) - 1, int(max_frames), dtype=int)
            indices = [indices[int(i)] for i in keep]
        return indices

    def sample_stream_frames(self, stream: VideoStreamRef) -> list[FrameRef]:
        static_filter = bool(self.cfg.get("prefer_static_frames", True))
        max_static_diff = float(self.cfg.get("static_diff_threshold", 16.0))
        min_sharpness = float(self.cfg.get("min_sharpness", 8.0))
        prev_gray: np.ndarray | None = None
        refs: list[FrameRef] = []
        for frame_idx in self._candidate_indices(stream):
            rgb = self.decoder.get_frame(stream, frame_idx)
            gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            sharpness = _variance_of_laplacian(rgb)
            if sharpness < min_sharpness:
                continue
            if static_filter and prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                if diff > max_static_diff:
                    prev_gray = gray
                    continue
            refs.append(FrameRef(stream_id=stream.stream_id, frame_idx=int(frame_idx), timestamp_sec=float(frame_idx / max(stream.fps, 1e-6))))
            prev_gray = gray
        if not refs and stream.num_frames > 0:
            refs.append(FrameRef(stream_id=stream.stream_id, frame_idx=0, timestamp_sec=0.0))
        return refs

    def sample(self, video_paths: list[str | Path]) -> tuple[list[VideoStreamRef], list[FrameRef]]:
        streams = self.decoder.decode_many(video_paths)
        refs: list[FrameRef] = []
        for stream in streams:
            refs.extend(self.sample_stream_frames(stream))
        refs.sort(key=lambda fr: (fr.stream_id, fr.frame_idx))
        return streams, refs

    def materialize_images(self, streams: list[VideoStreamRef], frame_refs: list[FrameRef], output_dir: str | Path) -> dict[str, str]:
        output_dir = ensure_dir(output_dir)
        stream_lookup = {stream.stream_id: stream for stream in streams}
        manifest: dict[str, dict[str, Any]] = {}
        image_index: dict[str, str] = {}
        for frame_ref in frame_refs:
            stream = stream_lookup[frame_ref.stream_id]
            rgb = self.decoder.get_frame(stream, frame_ref.frame_idx)
            max_dim = self.cfg.get("image_max_dim")
            if max_dim is not None:
                max_dim = int(max_dim)
                h, w = rgb.shape[:2]
                scale = min(1.0, float(max_dim) / float(max(h, w)))
                if scale < 1.0:
                    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                    rgb = np.asarray(Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS), dtype=np.uint8)
            rel = Path(frame_ref.stream_id) / f"frame_{int(frame_ref.frame_idx):06d}.png"
            path = output_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            image_index[f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}"] = str(rel.as_posix())
            manifest[str(rel.as_posix())] = {
                "stream_id": frame_ref.stream_id,
                "frame_idx": int(frame_ref.frame_idx),
                "timestamp_sec": float(frame_ref.timestamp_sec),
            }
        write_json(output_dir / "image_manifest.json", {"images": manifest})
        return image_index

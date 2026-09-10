"""Artifact for ingest and preprocessing outputs.

This artifact is the contract between raw input videos and all downstream stages.
It should stay lightweight and stable even as decoder backends change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.common.types import BaseArtifact, FrameRef, VideoStreamRef


@dataclass
class IngestArtifact(BaseArtifact):
    """Store decoded stream references and lightweight sampling products.

    Implementation notes:
    - This artifact intentionally does *not* store decoded pixel arrays.
    - Downstream stages should keep only references to streams and frames.
    - Cache paths for extracted frames or thumbnails should go into ``metadata``.
    """

    streams: list[VideoStreamRef] = field(default_factory=list)
    keyframes: list[FrameRef] = field(default_factory=list)
    quality_by_stream: dict[str, dict[str, float]] = field(default_factory=dict)
    sampled_windows: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def stream_ids(self) -> list[str]:
        """Return stream identifiers in ingest order."""
        return [stream.stream_id for stream in self.streams]

    def num_streams(self) -> int:
        """Return the number of decoded streams."""
        return len(self.streams)

    def summary(self) -> dict[str, Any]:
        """Return a compact ingest summary for logging and debugging."""
        return {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "num_streams": self.num_streams(),
            "stream_ids": self.stream_ids(),
            "num_keyframes": len(self.keyframes),
            "warnings": self.warnings,
        }

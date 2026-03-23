"""Artifact for temporal synchronization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class SyncArtifact(BaseArtifact):
    """Store pairwise and global time alignment results.

    Convention
    ----------
    ``pairwise_offsets["A|B"]["offset_sec"]`` means:
        ``t_B ~= t_A + offset_sec``
    after the two streams are aligned.
    """

    offsets_by_stream: dict[str, float] = field(default_factory=dict)
    confidence_by_stream: dict[str, float] = field(default_factory=dict)
    pairwise_offsets: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_by_pair: dict[str, float] = field(default_factory=dict)
    anchor_stream_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def stream_ids(self) -> list[str]:
        """Return stream identifiers with solved offsets."""
        return sorted(self.offsets_by_stream.keys())

    def pair_key(self, stream_a: str, stream_b: str) -> str:
        """Return the canonical pair key used in serialized artifacts."""
        return f"{stream_a}|{stream_b}"

    def summary(self) -> dict[str, Any]:
        """Return a compact synchronization summary."""
        return {
            "run_id": self.run_id,
            "anchor_stream_id": self.anchor_stream_id,
            "num_streams": len(self.offsets_by_stream),
            "num_pairs": len(self.pairwise_offsets),
            "streams": self.stream_ids(),
            "notes": self.notes,
        }

"""Artifact for human detections and fitted states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class HumanArtifact(BaseArtifact):
    """Store per-frame and linked human states."""

    humans: list[Any] = field(default_factory=list)
    linked_ids: dict[str, Any] = field(default_factory=dict)

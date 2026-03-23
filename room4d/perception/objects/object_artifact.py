"""Artifact for object detections, masks, and associations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class ObjectArtifact(BaseArtifact):
    """Store object detections, masks, embeddings, and associations."""

    objects: list[Any] = field(default_factory=list)
    linked_ids: dict[str, Any] = field(default_factory=dict)

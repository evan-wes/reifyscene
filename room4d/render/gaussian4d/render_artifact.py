"""Artifact for the fitted renderable scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class RenderArtifact(BaseArtifact):
    """Store a fitted dynamic scene representation."""

    scene_state: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)

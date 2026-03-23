"""Artifact for 3D lifting and partition results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from room4d.common.types import BaseArtifact


@dataclass
class LiftingArtifact(BaseArtifact):
    """Store 3D clusters, proposals, and partition labels."""

    clusters: dict[str, Any] = field(default_factory=dict)
    static_dynamic_labels: dict[str, Any] = field(default_factory=dict)

"""Base interfaces for geometry initializers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from reifyscene.common.types import FrameRef

if TYPE_CHECKING:
    from reifyscene.geometry.geometry_artifact import GeometryArtifact
    from reifyscene.ingest.ingest_artifact import IngestArtifact
    from reifyscene.sync.sync_artifact import SyncArtifact


class GeometryInitializer(ABC):
    """Abstract interface for coarse geometry initialization."""

    @abstractmethod
    def initialize(
        self,
        frames: list[FrameRef],
        *,
        ingest_artifact: IngestArtifact | None = None,
        sync_artifact: SyncArtifact | None = None,
    ) -> GeometryArtifact:
        """Estimate cameras, depth, and pairwise geometry from frames."""
        raise NotImplementedError

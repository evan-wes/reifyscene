"""Base interface for render backends."""

from __future__ import annotations

from typing import Any


class RenderBackend:
    """Base class for pluggable render backends."""

    backend_name = "base"

    def fit(self, scene_artifact, **kwargs):
        """Fit a renderable representation of the scene and export outputs."""
        raise NotImplementedError

    def render(self, scene_artifact, camera_path: dict[str, Any]):
        """Render frames from a camera path."""
        raise NotImplementedError

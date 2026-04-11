"""Base interfaces for renderable scene representations and render backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderSceneModel:
    """Backend-agnostic render-scene contract materialized from a SceneArtifact."""

    cfg: dict[str, Any] = field(default_factory=dict)
    scene_artifact: Any | None = None
    contract_name: str = "scene_contract_v1"
    supported_backends: list[str] = field(default_factory=list)
    quality_modes: list[str] = field(default_factory=lambda: ["realtime", "offline_high_quality"])

    def initialize_from_scene(self, scene_artifact) -> None:
        self.scene_artifact = scene_artifact
        contract = getattr(scene_artifact, "render_contract", {}) or {}
        self.contract_name = str(contract.get("schema", self.contract_name))
        self.supported_backends = [str(v) for v in contract.get("supported_backends", [])]
        if contract.get("quality_modes"):
            self.quality_modes = [str(v) for v in contract.get("quality_modes", [])]

    def summary(self) -> dict[str, Any]:
        contract = getattr(self.scene_artifact, "render_contract", {}) or {}
        static_layer = contract.get("static_layer", {}) or {}
        dynamic_layers = contract.get("dynamic_layers", {}) or {}
        return {
            "contract_name": self.contract_name,
            "supported_backends": list(self.supported_backends),
            "quality_modes": list(self.quality_modes),
            "num_static_gaussians": int(static_layer.get("num_gaussians", 0)),
            "num_dynamic_layers": sum(len(v) for v in dynamic_layers.values()) if isinstance(dynamic_layers, dict) else 0,
        }


class RenderBackend:
    """Base class for pluggable render backends."""

    backend_name = "base"

    def fit(self, scene: RenderSceneModel, **kwargs):
        """Fit or derive a renderable scene representation and export outputs."""
        raise NotImplementedError

    def render(self, scene: RenderSceneModel, camera_path: dict[str, Any]):
        """Render frames from a camera path."""
        raise NotImplementedError

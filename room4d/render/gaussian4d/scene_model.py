"""Renderable layered-scene model materialized as device-resident Gaussians."""

from __future__ import annotations

from typing import Any

import numpy as np

from room4d.render.base import RenderSceneModel
from room4d.render.camera_paths import _to_z_up_points
from room4d.render.gaussian4d.rasterizer import GaussianCloud


def static_cloud_from_artifact(
    scene_artifact, device: str, point_limit: int | None = None
) -> GaussianCloud:
    """Build the static Gaussian set from the scene contract's static layer.

    Falls back to the raw static point cloud (fixed scale/opacity seeds) when
    the contract carries no static layer, matching what the assembler would
    have produced from the same points.
    """
    contract = dict(getattr(scene_artifact, "render_contract", {}) or {})
    static_layer = dict(contract.get("static_layer", {}) or {})
    means = np.asarray(static_layer.get("means_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    scales = np.asarray(static_layer.get("scales_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    colors = np.asarray(static_layer.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
    opacities = np.asarray(static_layer.get("opacities", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if means.shape[0] == 0:
        means = np.asarray(scene_artifact.static_scene.get("points_xyz", np.zeros((0, 3))), dtype=np.float32)
        colors = np.asarray(scene_artifact.static_scene.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        scales = np.full_like(means, 0.03, dtype=np.float32)
        opacities = np.full((means.shape[0],), 0.92, dtype=np.float32)
    if point_limit is not None and means.shape[0] > point_limit:
        stride = max(1, int(np.ceil(means.shape[0] / float(point_limit))))
        means = means[::stride]
        scales = scales[::stride]
        colors = colors[::stride]
        opacities = opacities[::stride]
    return GaussianCloud.from_numpy(
        means=_to_z_up_points(means) if means.size else means.reshape(-1, 3),
        scales=scales,
        colors=colors,
        opacities=opacities,
        device=device,
    )


def entity_cloud(entity: dict[str, Any], device: str) -> GaussianCloud:
    """Template Gaussians for one selected dynamic entity.

    The entity dict comes from the trainer's per-frame selection and already
    carries z-up center/offsets.
    """
    center = np.asarray(entity["center_xyz"], dtype=np.float32)
    offsets = np.asarray(entity.get("template_offsets_xyz", [[0.0, 0.0, 0.0]]), dtype=np.float32)
    means = center[None, :] + offsets
    scales = np.asarray(entity.get("template_scales_xyz", np.full_like(means, 0.1)), dtype=np.float32)
    if scales.shape[0] != means.shape[0]:
        scales = np.repeat(scales[:1], means.shape[0], axis=0)
    opacities = np.asarray(entity.get("template_opacities", np.full((means.shape[0],), 0.8)), dtype=np.float32)
    if opacities.shape[0] != means.shape[0]:
        opacities = np.repeat(opacities[:1], means.shape[0], axis=0)
    colors = np.repeat(np.asarray(entity["color_rgb"], dtype=np.uint8)[None, :], means.shape[0], axis=0)
    return GaussianCloud.from_numpy(
        means=means, scales=scales, colors=colors, opacities=opacities, device=device
    )


class Gaussian4DScene(RenderSceneModel):
    """Layered static/dynamic scene contract backed by GaussianClouds."""

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg=cfg or {}, contract_name="layered_scene_v1")

    def static_gaussians(self, device: str) -> GaussianCloud:
        if self.scene_artifact is None:
            raise ValueError("Scene model has not been initialized from a SceneArtifact.")
        point_limit = self.cfg.get("point_limit")
        return static_cloud_from_artifact(
            self.scene_artifact,
            device=device,
            point_limit=None if point_limit is None else int(point_limit),
        )

"""Seed an (untrained) GaussianCloud by backprojecting GeometryArtifact depth.

This is the bridge from borrowed dataset geometry to a renderable Gaussian set:
each per-frame metric depth map is unprojected through its camera into the
shared world frame, colored by the frame's RGB, and emitted as one
:class:`GaussianCloud`. With no optimization yet, this is a colored point-cloud
splat — exactly what Checkpoint 0 renders to validate coordinate conventions
before any training time is spent. PR 2 turns these seeds into trainable
nn.Parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from room4d.render.gaussian4d.rasterizer import GaussianCloud


def _backproject_view(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Any,
    t_world_cam: np.ndarray,
    *,
    pixel_stride: int,
    depth_near: float,
    depth_far: float | None,
    scale_mult: float,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject one depth map to (world_xyz, colors_rgb, scales) point arrays."""
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    height, width = depth.shape[:2]
    fx, fy, cx, cy = (
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.cx),
        float(intrinsics.cy),
    )

    xs = np.arange(0, width, pixel_stride)
    ys = np.arange(0, height, pixel_stride)
    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu]

    valid = np.isfinite(z) & (z > depth_near)
    if depth_far is not None:
        valid &= z < depth_far
    uu, vv, z = uu[valid], vv[valid], z[valid]
    if z.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.float32)

    x_cam = (uu - cx) / fx * z
    y_cam = (vv - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1).astype(np.float32)

    R = np.asarray(t_world_cam, dtype=np.float32)[:3, :3]
    t = np.asarray(t_world_cam, dtype=np.float32)[:3, 3]
    pts_world = pts_cam @ R.T + t[None, :]

    colors = np.asarray(rgb, dtype=np.uint8)[vv, uu].reshape(-1, 3)

    # Per-point world-space splat radius ~= one (strided) pixel footprint at
    # this depth, so coverage scales with distance and resolution.
    radius = np.clip(scale_mult * pixel_stride * z / fx, scale_min, scale_max)
    scales = np.repeat(radius[:, None], 3, axis=1).astype(np.float32)
    return pts_world.astype(np.float32), colors, scales


def seed_arrays_from_geometry(
    geometry_artifact,
    rgb_by_frame: dict[str, np.ndarray],
    *,
    pixel_stride: int = 1,
    max_points: int | None = 1_500_000,
    depth_near: float = 1e-3,
    depth_far: float | None = None,
    scale_mult: float = 1.5,
    scale_min: float = 1e-5,
    scale_max: float = 0.05,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject every selected frame's depth into (means, colors_uint8, scales).

    Pure numpy (no torch/gsplat) so the scene assembler can build a persistable
    seed without a GPU. ``rgb_by_frame`` maps ``"{stream_id}:{frame_idx}"`` ->
    HxWx3 uint8 image, aligned with ``geometry_artifact.depthmaps_by_frame`` keys.
    """
    state_by_key: dict[tuple[str, float], Any] = {}
    for stream_id, states in geometry_artifact.camera_states_by_stream.items():
        for state in states:
            state_by_key[(str(stream_id), round(float(state.time_sec), 6))] = state

    xyz_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []

    for frame in geometry_artifact.selected_frames:
        key = f"{frame.stream_id}:{int(frame.frame_idx)}"
        depth_payload = geometry_artifact.depthmaps_by_frame.get(key)
        rgb = rgb_by_frame.get(key)
        state = state_by_key.get((str(frame.stream_id), round(float(frame.timestamp_sec), 6)))
        if depth_payload is None or rgb is None or state is None:
            continue
        xyz, colors, scales = _backproject_view(
            depth_payload["depth"],
            rgb,
            state.intrinsics,
            state.pose.T_world_cam,
            pixel_stride=pixel_stride,
            depth_near=depth_near,
            depth_far=depth_far,
            scale_mult=scale_mult,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        if xyz.shape[0]:
            xyz_parts.append(xyz)
            color_parts.append(colors)
            scale_parts.append(scales)

    if not xyz_parts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, 3), dtype=np.float32),
        )

    means = np.concatenate(xyz_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    scales = np.concatenate(scale_parts, axis=0)

    if max_points is not None and means.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(means.shape[0], size=int(max_points), replace=False)
        means, colors, scales = means[idx], colors[idx], scales[idx]

    return means.astype(np.float32), colors.astype(np.uint8), scales.astype(np.float32)


def seed_cloud_from_geometry(
    geometry_artifact,
    rgb_by_frame: dict[str, np.ndarray],
    *,
    device: str = "cuda",
    pixel_stride: int = 1,
    max_points: int | None = 1_500_000,
    depth_near: float = 1e-3,
    depth_far: float | None = None,
    scale_mult: float = 1.5,
    scale_min: float = 1e-5,
    scale_max: float = 0.05,
    opacity: float = 0.9,
    rng_seed: int = 0,
) -> GaussianCloud:
    """Backproject every selected frame's depth into one colored GaussianCloud."""
    means, colors, scales = seed_arrays_from_geometry(
        geometry_artifact,
        rgb_by_frame,
        pixel_stride=pixel_stride,
        max_points=max_points,
        depth_near=depth_near,
        depth_far=depth_far,
        scale_mult=scale_mult,
        scale_min=scale_min,
        scale_max=scale_max,
        rng_seed=rng_seed,
    )
    if means.shape[0] == 0:
        return GaussianCloud.empty(device)
    opacities = np.full((means.shape[0],), float(opacity), dtype=np.float32)
    return GaussianCloud.from_numpy(
        means=means,
        scales=scales,
        colors=colors,
        opacities=opacities,
        device=device,
    )

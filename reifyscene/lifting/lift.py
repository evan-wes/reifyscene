"""The one way a 2D track position becomes a 3D point.

Tracks are measured in the ingest frame's pixels; depth and intrinsics are in the
geometry's (for VGGT a resized, centre-cropped square). :func:`lift_pixels` maps
each position through the geometry's recorded :class:`ImageTransform`, reads the
depth at that geometry pixel and unprojects it with the camera. The tracking
stage's lift and the optimizer's re-lift both call it, so a track has one 3D
position for a given camera, not one per code path.

A position the crop removed, or on a depth hole, or under the confidence
threshold, gets no 3D point -- it is never clamped onto the image border.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reifyscene.common.depth import valid_depth_mask
from reifyscene.geometry.image_transform import ImageTransform

LIFT_FUNCTION = "lift_pixels"


def _intrinsics(K: Any) -> tuple[float, float, float, float]:
    if hasattr(K, "fx"):
        return float(K.fx), float(K.fy), float(K.cx), float(K.cy)
    K = np.asarray(K, dtype=np.float64)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def lift_pixels(
    points_xy: np.ndarray,
    transform: ImageTransform,
    depth: np.ndarray,
    K: Any,
    T_world_cam: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    conf_threshold: float = 0.0,
    from_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) for ingest-frame pixels (N, 2), and which are valid.

    ``transform`` maps ingest-frame pixels (or ``from_size`` pixels, one of its
    intermediate sizes) into the geometry pixels ``depth`` (H, W) and ``K``
    (``CameraIntrinsics`` or 3x3) describe; ``T_world_cam`` is the camera-to-world
    pose. The depth is read at the nearest geometry pixel and the continuous mapped
    position is unprojected (the pipeline's convention). Invalid points are NaN.
    ``confidence`` (H, W) must exceed ``conf_threshold`` where given.
    """
    depth_map = np.asarray(depth, dtype=np.float32)
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    h, w = depth_map.shape[:2]
    if transform.output_size != (w, h):
        raise ValueError(
            f"image transform lands on {transform.output_size[0]}x{transform.output_size[1]} "
            f"but the depth map is {w}x{h}"
        )
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    mapped, valid = transform.map_points(xy, from_size=from_size)
    xyz = np.full((xy.shape[0], 3), np.nan, dtype=np.float32)
    if not valid.any():
        return xyz, valid

    u = np.clip(np.round(mapped[valid, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(mapped[valid, 1]).astype(int), 0, h - 1)
    z = depth_map[v, u].astype(np.float64)
    ok = valid_depth_mask(z)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        sampled = conf[v, u]
        ok &= np.isfinite(sampled) & (sampled > float(conf_threshold))

    fx, fy, cx, cy = _intrinsics(K)
    x_cam = (mapped[valid, 0] - cx) / max(fx, 1e-6) * z
    y_cam = (mapped[valid, 1] - cy) / max(fy, 1e-6) * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1).astype(np.float32)
    T = np.asarray(T_world_cam, dtype=np.float32)
    pts_world = pts_cam @ T[:3, :3].T + T[:3, 3][None, :]
    pts_world[~ok] = np.nan

    idx = np.nonzero(valid)[0]
    xyz[idx] = pts_world
    valid = valid.copy()
    valid[idx[~ok]] = False
    return xyz, valid


def lift_record(
    *, confidence: str | None, conf_threshold: float, transforms: dict[str, Any]
) -> dict[str, Any]:
    """What a lift ran with: recorded on the artifact that holds the lifted points."""
    return {
        "function": LIFT_FUNCTION,
        "source": "depth",
        "confidence": confidence,
        "point_conf_threshold": float(conf_threshold),
        "image_transforms": {str(k): dict(v) for k, v in transforms.items()},
    }

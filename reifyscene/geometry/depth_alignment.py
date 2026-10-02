"""Align a frame's depth map to sparse, multi-view-consistent 3D points.

The geometry backend's depth is consistent within a chunk of frames but drifts in scale across
chunks (VGGT-Long stitches chunks with similarity transforms), and pose refinement moves the
cameras without touching the depth. Measured on lab-coat (2026-10-01): the same static wall
seen 15 frames apart sits at 0.63x the depth. A bundle adjustment's points, by contrast, share
one scale for the whole scene. Fitting each frame's depth to the points that frame observes
makes every frame agree with every other.

The fit is Shape-of-Motion's (``preproc/compute_depth.py: align_monodepth_with_colmap``, MIT,
Copyright (c) 2024 Vickie Ye), in disparity: with each disparity median-centred, ``scale`` is
the median ratio and ``shift`` the median residual; the whole map becomes
``scale * disparity + shift``, and disparities below a small floor are invalid (depth 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthAlignment:
    scale: float          # in disparity
    shift: float          # in disparity
    num_points: int
    residual: float       # median |log(aligned depth / point depth)| at the points, after


def _bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample ``image`` (H, W) at pixel positions ``uv`` (N, 2) (pixel centres at integers)."""
    h, w = image.shape
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
    a, b = u - u0, v - v0
    return ((1 - a) * (1 - b) * image[v0, u0] + a * (1 - b) * image[v0, u1]
            + (1 - a) * b * image[v1, u0] + a * b * image[v1, u1])


def align_depth(depth: np.ndarray, point_depth: np.ndarray, point_uv: np.ndarray
                ) -> tuple[np.ndarray, DepthAlignment]:
    """``depth`` (H, W) aligned to points seen at ``point_uv`` (N, 2) with camera depths
    ``point_depth`` (N,). Returns the aligned depth (0 where invalid) and the fit."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    uv = np.asarray(point_uv, dtype=np.float64)
    z = np.asarray(point_depth, dtype=np.float64)
    keep = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) \
        & (uv[:, 1] <= h - 1)
    uv, z = uv[keep], z[keep]
    disp_map = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-12), 0.0)
    point_disp = 1.0 / np.clip(z, 1e-6, 1e6)
    map_disp = _bilinear(disp_map, uv)
    ok = map_disp > 0
    point_disp, map_disp, z = point_disp[ok], map_disp[ok], z[ok]
    if point_disp.size < 3:
        raise ValueError(f"only {point_disp.size} points land on valid depth: nothing to fit")
    centred_points = point_disp - np.median(point_disp) + 1e-8
    centred_map = map_disp - np.median(map_disp) + 1e-8
    scale = float(np.median(centred_points / centred_map))
    shift = float(np.median(point_disp - scale * map_disp))
    aligned_disp = scale * disp_map + shift
    floor = min(1e-6, float(np.quantile(aligned_disp, 0.01)))
    valid = (aligned_disp >= floor) & (aligned_disp > 0) & (depth > 0)
    aligned = np.where(valid, 1.0 / np.where(valid, aligned_disp, 1.0), 0.0)
    at_points = scale * map_disp + shift
    good = at_points > 0
    residual = float(np.median(np.abs(np.log((1.0 / at_points[good]) / z[good])))) \
        if good.any() else float("inf")
    return aligned.astype(np.float32), DepthAlignment(scale, shift, int(point_disp.size), residual)


def pointmap_from_depth(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """World points (H, W, 3) of a depth map (H, W) through a pinhole camera; NaN where the
    depth is invalid (<= 0)."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(np.asarray(K, np.float64)).T
    cam = rays * depth[..., None]
    T = np.asarray(T_world_cam, dtype=np.float64)
    world = cam @ T[:3, :3].T + T[:3, 3]
    world[depth <= 0] = np.nan
    return world.astype(np.float32)

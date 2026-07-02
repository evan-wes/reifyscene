"""Shared depth-validity + back-projection helpers.

One definition of **valid depth** — finite **and** positive (within an optional far
bound) — so every consumer treats zero / NaN / negative depth the same way. This matters
because **a zero depth is *finite***: a plain ``isfinite`` guard lets a depth hole through
and back-projects it to a spurious point at the camera origin. Ground-truth depth (and
sensor depth generally) carries zeros/NaNs at holes, so producers and consumers alike route
through here.

``unproject_depth_to_world`` is the single dense back-projection convention (world-from-camera
pose, OpenCV pinhole intrinsics) — the same convention VGGT emits, so downstream consumes any
backend's point maps identically. Invalid pixels come back as ``NaN`` (the standard sentinel;
every point-map consumer already filters ``isfinite``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_hw(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    return d


def valid_depth_mask(depth: np.ndarray, *, near: float = 1e-3, far: float | None = None) -> np.ndarray:
    """Boolean mask of usable depth: **finite and > near** (and ``< far`` when given).

    A zero/negative/NaN depth is *not* valid — the guard every depth consumer should use so
    a hole never back-projects to the camera origin.
    """
    d = _as_hw(depth)
    mask = np.isfinite(d) & (d > float(near))
    if far is not None:
        mask &= d < float(far)
    return mask


def unproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: Any,
    T_world_cam: np.ndarray,
    *,
    near: float = 1e-3,
    far: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense back-projection of a metric depth map to world points.

    Parameters
    ----------
    depth : ``HxW`` (or ``HxWx1``) metric depth.
    intrinsics : object with ``fx, fy, cx, cy`` (pixels), at the depth resolution.
    T_world_cam : ``4x4`` world-from-camera pose.

    Returns
    -------
    ``(points, valid)`` — ``points`` is ``HxWx3`` world coordinates with **NaN at invalid
    depth**; ``valid`` is the ``HxW`` boolean mask (usable as a confidence channel).
    """
    d = _as_hw(depth)
    h, w = d.shape
    valid = valid_depth_mask(d, near=near, far=far)
    z = np.where(valid, d, np.nan).astype(np.float32)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx = max(float(intrinsics.fx), 1e-6)
    fy = max(float(intrinsics.fy), 1e-6)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    x_cam = (uu - cx) / fx * z
    y_cam = (vv - cy) / fy * z
    cam = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)  # NaN where invalid
    T = np.asarray(T_world_cam, dtype=np.float32)
    world = cam @ T[:3, :3].T + T[:3, 3][None, :]
    return world.reshape(h, w, 3).astype(np.float32), valid

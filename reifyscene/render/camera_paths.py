"""Camera paths for novel-view playback."""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def orbit_cameras(
    center,
    radius: float,
    num_frames: int,
    *,
    up=(0.0, 0.0, 1.0),
    start_dir=None,
    arc_degrees: float = 90.0,
    height_frac: float = 0.2,
):
    """Arc of look-at cameras around ``center`` in the plane perpendicular to ``up``.

    The sweep is centered on ``start_dir`` (projected into that plane), so the
    path hugs the input trajectory rather than orbiting behind the scene where
    monocular 4D collapses. Returns a list of {"eye", "target", "up"} dicts.
    """
    center = np.asarray(center, dtype=np.float32)
    u = _unit(up)
    if start_dir is not None:
        start = np.asarray(start_dir, dtype=np.float32)
        a = _unit(start - float(np.dot(start, u)) * u)
    else:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        a = _unit(seed - float(np.dot(seed, u)) * u)
    b = _unit(np.cross(u, a))
    half = np.radians(float(arc_degrees)) / 2.0
    cameras = []
    for i in range(int(num_frames)):
        frac = 0.0 if num_frames <= 1 else i / (num_frames - 1)
        theta = -half + frac * 2.0 * half
        direction = np.cos(theta) * a + np.sin(theta) * b
        eye = center + float(radius) * direction + float(height_frac) * float(radius) * u
        cameras.append({"eye": eye.tolist(), "target": center.tolist(), "up": u.tolist()})
    return cameras

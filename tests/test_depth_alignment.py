"""Depth alignment to BA points (reifyscene/geometry/depth_alignment.py).

What these pin: a depth map that is off by a scale and shift in disparity is brought back onto
the points exactly; the fit is robust to a minority of wrong points; it refuses when nothing
lands on valid depth; and a point map re-derived from depth sits where the camera says.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.geometry.depth_alignment import align_depth, pointmap_from_depth

H, W = 40, 60


def _scene():
    v, u = np.mgrid[0:H, 0:W]
    true = 2.0 + 0.05 * u + 0.03 * v                  # a slanted plane, depth 2..6
    return true


def _points(true, n=200, seed=0):
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, W - 1, n), rng.uniform(0, H - 1, n)], 1)
    z = 2.0 + 0.05 * uv[:, 0] + 0.03 * uv[:, 1]         # the plane, exactly
    return uv, z


def test_a_scale_and_shift_in_disparity_are_undone() -> None:
    true = _scene()
    drifted = 1.0 / (0.6 / true + 0.02)               # what a drifting chunk would give
    uv, z = _points(true)
    aligned, fit = align_depth(drifted, z, uv)
    assert np.allclose(aligned, true, rtol=1e-3)
    assert fit.num_points == 200 and fit.residual < 1e-3


def test_a_few_wrong_points_barely_move_the_fit_but_a_cluster_does() -> None:
    """Its robustness is the median's: a few bad tracks are absorbed, a one-sided cluster
    (a person's points while they move) is not -- which is why foreground points are left
    out before fitting."""
    true = _scene()
    uv, z = _points(true)
    few = z.copy()
    few[:10] *= 3.0                                   # 5% bad points
    aligned, _ = align_depth(true * 0.5, few, uv)     # pure scale drift
    assert np.allclose(aligned, true, rtol=0.05)
    many = z.copy()
    many[:30] *= 3.0                                  # 15%, all on one side
    aligned, _ = align_depth(true * 0.5, many, uv)
    assert not np.allclose(aligned, true, rtol=0.05)


def test_no_points_on_valid_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to fit"):
        align_depth(np.zeros((H, W)), np.array([1.0, 2.0]), np.array([[3.0, 3.0], [5.0, 5.0]]))


def test_a_point_map_from_depth_lands_where_the_camera_says() -> None:
    K = np.array([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [1.0, -2.0, 0.5]                        # camera centre in the world
    depth = np.full((H, W), 4.0)
    depth[0, 0] = 0.0
    pts = pointmap_from_depth(depth, K, T)
    assert np.allclose(pts[H // 2, W // 2], [1.0, -2.0, 4.5])   # straight ahead, 4 deep
    assert np.isnan(pts[0, 0]).all()

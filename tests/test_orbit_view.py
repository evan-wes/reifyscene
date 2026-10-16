"""The 3D view's camera (reifyscene/edit/renderer.py ``orbit_w2c``).

What these pin: no motion is the training camera itself; orbiting keeps the pivot centred and at
the zoomed distance; a half turn looks back at the pivot from the far side.
"""

from __future__ import annotations

import numpy as np

from reifyscene.edit.renderer import orbit_w2c


def _camera() -> np.ndarray:
    """A camera at (1, 2, 3) turned 30 degrees about world y."""
    a = np.radians(30)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = r, [1.0, 2.0, 3.0]
    return np.linalg.inv(c2w)


def _pivot(w2c: np.ndarray, depth: float) -> np.ndarray:
    return (np.linalg.inv(w2c) @ np.array([0, 0, depth, 1.0]))[:3]


def _in_view(w2c: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (w2c @ np.append(p, 1.0))[:3]


def test_no_motion_is_the_training_camera() -> None:
    w2c = _camera()
    assert np.allclose(orbit_w2c(w2c, 4.0), w2c)


def test_orbit_keeps_the_pivot_centred_at_the_zoomed_distance() -> None:
    w2c, d = _camera(), 4.0
    pivot = _pivot(w2c, d)
    for yaw, pitch, zoom in [(40, 0, 1), (-70, 25, 0.5), (0, -60, 2.5)]:
        seen = _in_view(orbit_w2c(w2c, d, yaw=yaw, pitch=pitch, zoom=zoom), pivot)
        assert np.allclose(seen, [0, 0, d * zoom]), (yaw, pitch, zoom, seen)


def test_a_half_turn_sees_the_pivot_from_the_far_side() -> None:
    w2c, d = _camera(), 4.0
    back = orbit_w2c(w2c, d, yaw=180)
    eye0 = np.linalg.inv(w2c)[:3, 3]
    assert np.allclose(_in_view(back, eye0), [0, 0, 2 * d])      # the old camera, behind the pivot


def test_pan_moves_the_pivot_sideways_in_the_view() -> None:
    w2c, d = _camera(), 4.0
    panned = orbit_w2c(w2c, d, pan_x=0.5)
    assert np.allclose(_in_view(panned, _pivot(w2c, d)), [-0.5 * d, 0, d])

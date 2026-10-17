"""The body bound (reifyscene/edit/body_volume.py, assignment.bound_to_body).

What these pin: keypoints lift to the depth the scene rendered behind them, pushed back to the
joint's centre; a frame without a whole torso, a joint with no mask pixels or a low score, or
an occluded joint (a depth far from the torso's) is not guessed at; capsules contain what is
inside the body and not what is behind it; the bound keeps a moving person's Gaussians and a
static one's on any frame, and leaves a person with no body untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from reifyscene.edit.body_volume import JOINTS, BodyConfig, BodyVolume, lift_frame
from reifyscene.geometry.image_transform import ImageTransform

W, H, F = 64, 64, 64.0          # raster, focal length
DEPTH = 4.0


def _camera():
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    return SimpleNamespace(K=K, viewmat=np.eye(4))           # camera at the origin, looking +z


def _keypoints(points: dict[str, tuple[float, float]], score: float = 0.9) -> np.ndarray:
    kp = np.zeros((308, 3))
    for name, (u, v) in points.items():
        kp[JOINTS[name]] = (u, v, score)
    return kp


TORSO = {"left_shoulder": (40, 20), "right_shoulder": (24, 20),
         "left_hip": (38, 44), "right_hip": (26, 44)}


def _lift(points, depth=None, mask=None, **cfg):
    depth = np.full((H, W), DEPTH) if depth is None else depth
    mask = np.ones((H, W), bool) if mask is None else mask
    return lift_frame(_keypoints(points), mask, depth, _camera(),
                      ImageTransform.identity((W, H)), (W, H), BodyConfig(**cfg))


def test_a_torso_lifts_to_the_rendered_depth_and_scales_the_body() -> None:
    capsules, length = _lift(TORSO)
    # mid-shoulder (32, 20) to mid-hip (32, 44) is 24 px at depth 4 and focal 64: 1.5 units
    assert np.isclose(length, 24 * DEPTH / F)
    body = BodyVolume({0: capsules}, {0: length})
    torso_centre = np.array([0.0, 0.0, DEPTH + 0.35 * length])     # pushed back off the surface
    assert body.distance(torso_centre, 0)[0] < 0
    assert body.distance(np.array([0.0, 0.0, DEPTH + 3 * length]), 0)[0] > 0   # far behind


def test_no_torso_no_body() -> None:
    assert _lift({k: v for k, v in TORSO.items() if k != "left_hip"}) is None
    assert _lift(TORSO, mask=np.zeros((H, W), bool)) is None        # no person pixels
    assert lift_frame(_keypoints(TORSO, score=0.1), np.ones((H, W), bool),
                      np.full((H, W), DEPTH), _camera(), ImageTransform.identity((W, H)),
                      (W, H), BodyConfig()) is None                  # not seen confidently


def test_an_occluded_joint_is_left_out_not_guessed() -> None:
    depth = np.full((H, W), DEPTH)
    depth[50:62, 26:40] = 0.5                         # something close in front of the knee
    capsules, length = _lift({**TORSO, "left_knee": (32, 55)}, depth=depth)
    ends = [tuple(np.round(a, 6)) for a, b, _ in capsules] + [tuple(np.round(b, 6))
                                                              for _, b, _ in capsules]
    assert not any(e[2] < DEPTH for e in ends)        # no joint at the occluder's depth
    assert len(capsules) == len(_lift(TORSO)[0])      # the thigh was not built


def test_pixels_without_depth_are_not_read_as_depth() -> None:
    depth = np.full((H, W), DEPTH)
    depth[::2] = 0.0                                  # the geometry has no depth on half the rows
    capsules, length = _lift(TORSO, depth=depth)
    assert np.isclose(length, 24 * DEPTH / F)         # the median of the valid pixels only


def test_the_bound_keeps_a_moving_person_and_a_static_one_seen_once() -> None:
    from reifyscene.graph.assignment import bound_to_body

    # one capsule round the origin; frame 0 and frame 1
    body = BodyVolume({0: [(np.zeros(3), np.zeros(3), 1.0)],
                       1: [(np.zeros(3), np.zeros(3), 1.0)]}, {0: 1.0, 1: 1.0})
    fg_at = {0: np.array([[0, 0, 0.5], [0, 0, 5]]), 1: np.array([[0.2, 0, 0], [0, 0, 6]])}
    bg = np.array([[0, 0, 0.9], [0, 0, 9]])

    class Scene:
        fg = SimpleNamespace(num_gaussians=2)

        def cloud_at(self, t):
            import torch
            return SimpleNamespace(means=torch.as_tensor(np.concatenate([fg_at[t], bg])))

    fg, kept_bg, stats = bound_to_body(Scene(), np.array([0, 1]), np.array([0, 1]), body,
                                       {0: 0, 1: 1})
    assert fg.tolist() == [0] and kept_bg.tolist() == [0] and stats["outside_body"] == 2
    empty = BodyVolume()
    fg, kept_bg, stats = bound_to_body(Scene(), np.array([0, 1]), np.array([1]), empty, {0: 0})
    assert fg.tolist() == [0, 1] and kept_bg.tolist() == [1] and stats["body_frames"] == 0

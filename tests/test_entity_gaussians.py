"""Assigning Gaussians to entities (reifyscene/graph/entity_gaussians.py).

The rule is the point-track rule over a different population, so these tests cover the
part that is new: projecting per-frame positions into their views, and the assignment
that follows from it. A synthetic camera looking down -Z makes the expected pixels
arithmetic rather than a guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.graph.entity_gaussians import assign_gaussians_to_entities, project_to_views

WIDTH = HEIGHT = 40


def camera(n_frames: int, *, distance: float = 5.0):
    """n identical cameras at the origin looking down +Z, focal 20, centre (20, 20)."""
    w2c = np.tile(np.eye(4), (n_frames, 1, 1))
    w2c[:, 2, 3] = distance
    K = np.tile(np.asarray([[20.0, 0, 20.0], [0, 20.0, 20.0], [0, 0, 1.0]]), (n_frames, 1, 1))
    return w2c, K


def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_projection_puts_a_point_where_the_arithmetic_says() -> None:
    w2c, K = camera(1)
    # A point 1 unit right of the axis, 5 in front: u = 20 * (1/5) + 20 = 24.
    xy, in_front = project_to_views(np.asarray([[[1.0, 0.0, 0.0]]]), w2c, K)
    assert in_front.all()
    assert np.allclose(xy[0, 0], [24.0, 20.0])


def test_a_point_behind_the_camera_is_not_an_observation() -> None:
    w2c, K = camera(1)
    xy, in_front = project_to_views(np.asarray([[[0.0, 0.0, -10.0]]]), w2c, K)
    assert not in_front.any()
    assert xy.shape == (1, 1, 2)


def test_each_gaussian_goes_to_the_entity_it_sits_inside() -> None:
    frames = [0, 1, 2, 3]
    w2c, K = camera(len(frames))
    # Three Gaussians: one projecting into the left box, one into the right, one outside.
    pos = np.tile(
        np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), (len(frames), 1, 1)
    )
    left, right = box(12, 16, 18, 24), box(22, 16, 28, 24)
    masks = {f: {"human:a": left, "object:b": right} for f in frames}

    support = assign_gaussians_to_entities(
        masks, frame_indices=frames, positions=pos, w2c=w2c, intrinsics=K
    )

    assert support.track_ids_for("human:a") == [0]
    assert support.track_ids_for("object:b") == [1]
    assert support.free_track_ids == [2]  # inside nothing: the background's


def test_a_gaussian_that_moves_between_entities_goes_to_the_one_it_is_in_most() -> None:
    frames = [0, 1, 2, 3, 4]
    w2c, K = camera(len(frames))
    # Two frames inside the left box, three inside the right.
    path = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    pos = np.asarray(path).reshape(len(frames), 1, 3)
    masks = {f: {"human:a": box(12, 16, 18, 24), "object:b": box(22, 16, 28, 24)} for f in frames}

    support = assign_gaussians_to_entities(
        masks, frame_indices=frames, positions=pos, w2c=w2c, intrinsics=K
    )
    assert support.entity_of_track == {0: "object:b"}


def test_mismatched_frame_counts_are_refused() -> None:
    w2c, K = camera(3)
    with pytest.raises(ValueError, match="poses and"):
        project_to_views(np.zeros((4, 2, 3)), w2c, K)
    with pytest.raises(ValueError, match="positions must be"):
        project_to_views(np.zeros((3, 2)), w2c, K)

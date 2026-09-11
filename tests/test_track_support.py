"""Which tracks are evidence for which entity (reifyscene/graph/track_support.py).

Synthetic masks and tracks: a person's box on the left, an object's box on the right,
and tracks placed to exercise each outcome -- supported, ambiguous, background.
"""

from __future__ import annotations

import numpy as np

from reifyscene.graph.track_support import assign_track_support, positions_by_frame

HEIGHT, WIDTH = 40, 60


def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """A boolean mask with one filled rectangle."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


PERSON = box(5, 5, 20, 30)
OBJECT = box(35, 5, 55, 30)


def masks(num_frames: int = 4) -> dict[int, dict[str, np.ndarray]]:
    """Both entities masked on every frame."""
    return {f: {"human:cam0:p1": PERSON, "object:cam0:o1": OBJECT} for f in range(num_frames)}


def tracks(points: list[list[tuple[float, float]]], visibility=None):
    """Build (frame_indices, points_xy, visibility, track_ids) from per-track paths."""
    xy = np.asarray(points, dtype=np.float32).transpose(1, 0, 2)  # (T, N, 2)
    return {
        "frame_indices": np.arange(xy.shape[0]),
        "points_xy": xy,
        "visibility": visibility,
        "track_ids": np.arange(xy.shape[1]),
    }


def test_a_track_inside_one_mask_supports_that_entity() -> None:
    support = assign_track_support(masks(), **tracks([[(10, 10)] * 4]))
    assert support.entity_of_track == {0: "human:cam0:p1"}
    assert support.inside_frames["human:cam0:p1"] == {0: 4}
    assert support.shared_frames["human:cam0:p1"] == {0: 4}
    assert support.free_track_ids == [] and support.ambiguous_track_ids == []


def test_a_track_inside_nothing_is_the_background_s_evidence() -> None:
    support = assign_track_support(masks(), **tracks([[(28, 35)] * 4]))
    assert support.entity_of_track == {}
    assert support.free_track_ids == [0]
    assert support.ambiguous_track_ids == []


def test_a_track_inside_too_rarely_is_ambiguous_not_background() -> None:
    # Inside the person on one frame of four: below both the ratio and the frame count.
    support = assign_track_support(masks(), **tracks([[(10, 10), (28, 35), (28, 35), (28, 35)]]))
    assert support.ambiguous_track_ids == [0]
    assert support.free_track_ids == []
    assert support.entity_of_track == {}


def test_min_frames_rejects_a_perfect_ratio_on_one_frame() -> None:
    # Visible on one frame only, inside it every time: ratio 1.0, but one frame.
    vis = np.zeros((4, 1), dtype=bool)
    vis[0] = True
    support = assign_track_support(masks(), **tracks([[(10, 10)] * 4], visibility=vis))
    assert support.ambiguous_track_ids == [0]
    assert support.entity_of_track == {}


def test_a_track_supports_the_entity_it_is_inside_most_often() -> None:
    # Two frames inside the person, then three inside the object.
    path = [(10, 10), (10, 10), (40, 10), (40, 10), (40, 10)]
    frames = {f: {"human:cam0:p1": PERSON, "object:cam0:o1": OBJECT} for f in range(5)}
    support = assign_track_support(frames, **tracks([path]))
    assert support.entity_of_track == {0: "object:cam0:o1"}
    assert list(support.inside_frames) == ["object:cam0:o1"]
    assert support.inside_frames["object:cam0:o1"] == {0: 3}


def test_invisible_and_off_frame_points_are_not_evidence() -> None:
    vis = np.ones((4, 2), dtype=bool)
    vis[:, 1] = False  # the second track is never visible
    off = [(-5.0, 10.0), (WIDTH + 3.0, 10.0), (10.0, -2.0), (10.0, HEIGHT + 1.0)]
    support = assign_track_support(masks(), **tracks([off, [(10, 10)] * 4], visibility=vis))
    # Off the frame on every frame, and invisible on every frame: no evidence either way.
    assert support.entity_of_track == {}
    assert support.free_track_ids == [0, 1]
    assert support.shared_frames == {}


def test_the_ratio_is_over_the_frames_the_two_share() -> None:
    # The entity is masked on two of four frames; the track is inside both of those.
    frames = {0: {"human:cam0:p1": PERSON}, 2: {"human:cam0:p1": PERSON}}
    support = assign_track_support(frames, **tracks([[(10, 10), (28, 35), (10, 10), (28, 35)]]))
    assert support.entity_of_track == {0: "human:cam0:p1"}
    assert support.shared_frames["human:cam0:p1"] == {0: 2}


def test_a_point_is_tested_at_its_nearest_pixel() -> None:
    # x=4.6 rounds to 5, the first column inside the person's box; 4.4 rounds to 4, outside.
    inside = assign_track_support(masks(), **tracks([[(4.6, 10)] * 4]))
    outside = assign_track_support(masks(), **tracks([[(4.4, 10)] * 4]))
    assert inside.entity_of_track == {0: "human:cam0:p1"}
    assert outside.entity_of_track == {}


def test_positions_are_the_median_of_the_supporting_points() -> None:
    xyz = np.asarray(
        [
            [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [100.0, 0.0, 1.0]],  # frame 0: one outlier
            [[np.nan] * 3, [np.nan] * 3, [np.nan] * 3],  # frame 1: nothing lifted
        ],
        dtype=np.float32,
    )
    valid = np.asarray([[True, True, True], [False, False, False]])
    out = positions_by_frame(
        np.asarray([0, 1, 2]),
        lifted_frame_indices=np.asarray([7, 9]),
        lifted_xyz=xyz,
        lifted_valid=valid,
    )
    assert [p["frame_idx"] for p in out] == [7]
    assert out[0]["xyz"] == [2.0, 0.0, 1.0]
    assert out[0]["num_points"] == 3


def test_positions_use_only_the_entity_s_own_columns() -> None:
    xyz = np.asarray([[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]], dtype=np.float32)
    valid = np.asarray([[True, True]])
    out = positions_by_frame(
        np.asarray([1]), lifted_frame_indices=np.asarray([0]), lifted_xyz=xyz, lifted_valid=valid
    )
    assert out[0]["xyz"] == [10.0, 10.0, 10.0]
    assert out[0]["num_points"] == 1


def test_mismatched_track_arrays_are_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="points_xy must be"):
        assign_track_support(
            masks(),
            frame_indices=np.arange(4),
            points_xy=np.zeros((3, 1, 2), dtype=np.float32),
            visibility=None,
            track_ids=np.arange(1),
        )

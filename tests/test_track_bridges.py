"""Tests for point tracks that span an occlusion gap as re-identification evidence."""

from __future__ import annotations

import numpy as np

from reifyscene.perception.reid.dino_reid import build_cross_occlusion_merge_map
from reifyscene.perception.reid.track_bridges import bridge_counts, tracklet_point_sets


def _mask(h: int, w: int, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- counting the bridges ----------------------------------------------------

def test_a_point_inside_two_tracklets_bridges_them() -> None:
    """One tracked point in A before the gap and in B after it: the same thing moved."""
    masks = {
        0: {"a": _mask(20, 20, (0, 20, 0, 10))},
        1: {"a": _mask(20, 20, (0, 20, 0, 10))},
        8: {"b": _mask(20, 20, (0, 20, 10, 20))},
        9: {"b": _mask(20, 20, (0, 20, 10, 20))},
    }
    frames = np.array([0, 1, 8, 9])
    # one point crosses from the left half to the right; a second stays left the whole time
    xy = np.array([[[2, 5], [2, 15]], [[2, 5], [2, 15]], [[15, 5], [2, 15]], [[15, 5], [2, 15]]],
                  dtype=float)
    sets = tracklet_point_sets(masks, frame_indices=frames, points_xy=xy, visibility=None)
    assert sets["a"] == {0, 1}   # both points sit in the left half while A is masked
    assert sets["b"] == {0}      # only the crossing point reaches B
    assert bridge_counts(sets) == {("a", "b"): 1}


def test_one_frame_inside_is_a_coincidence_not_evidence() -> None:
    masks = {f: {"a": _mask(20, 20, (0, 20, 0, 10))} for f in range(4)}
    frames = np.array([0, 1, 2, 3])
    # the point clips the mask on one frame only
    xy = np.tile(np.array([[[15.0, 5.0]]]), (4, 1, 1))
    xy[1, 0] = [2.0, 5.0]
    sets = tracklet_point_sets(masks, frame_indices=frames, points_xy=xy, visibility=None,
                              min_frames=2)
    assert sets["a"] == set()


def test_an_invisible_point_is_not_inside_anything() -> None:
    masks = {f: {"a": _mask(20, 20, (0, 20, 0, 10))} for f in range(4)}
    frames = np.array([0, 1, 2, 3])
    xy = np.tile(np.array([[[2.0, 5.0]]]), (4, 1, 1))
    vis = np.zeros((4, 1), dtype=bool)
    sets = tracklet_point_sets(masks, frame_indices=frames, points_xy=xy, visibility=vis)
    assert sets["a"] == set()


def test_no_shared_points_is_no_bridge() -> None:
    assert bridge_counts({"a": {0, 1}, "b": {2, 3}}) == {}


# --- how a bridge is allowed to act ------------------------------------------

def _pair():
    # cosine 0.93: below the 0.95 threshold, above it once a bridge relaxes it by 0.05
    vectors = {"a": _unit(1.0, 0.0), "b": _unit(0.93, 0.3675)}
    spans = {"a": ("s", 0, 10), "b": ("s", 20, 30)}
    return vectors, spans


def test_a_bridge_admits_a_pair_appearance_alone_refuses() -> None:
    vectors, spans = _pair()
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.95, ratio_margin=0.0) == {}
    bridged = build_cross_occlusion_merge_map(
        vectors, spans, threshold=0.95, ratio_margin=0.0, bridges={("a", "b"): 3}
    )
    assert bridged == {"b": "a"}  # canonical is the earlier-appearing tracklet


def test_a_single_bridge_is_below_the_bar() -> None:
    vectors, spans = _pair()
    assert build_cross_occlusion_merge_map(
        vectors, spans, threshold=0.95, ratio_margin=0.0, bridges={("a", "b"): 1}, bridge_min=2
    ) == {}


def test_a_bridge_cannot_merge_tracklets_that_look_nothing_alike() -> None:
    """Geometry corroborates; it never overrides. A drifting point must not fuse two people."""
    vectors = {"a": _unit(1.0, 0.0), "b": _unit(0.0, 1.0)}  # cosine 0.0
    spans = {"a": ("s", 0, 10), "b": ("s", 20, 30)}
    assert build_cross_occlusion_merge_map(
        vectors, spans, threshold=0.95, ratio_margin=0.0, bridges={("a", "b"): 50}
    ) == {}


def test_co_visible_tracklets_are_never_bridged_into_one() -> None:
    """Overlapping spans mean both were on screen at once -- not an occlusion gap."""
    vectors, _ = _pair()
    spans = {"a": ("s", 0, 25), "b": ("s", 20, 30)}
    assert build_cross_occlusion_merge_map(
        vectors, spans, threshold=0.5, ratio_margin=0.0, bridges={("a", "b"): 50}
    ) == {}

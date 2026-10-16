"""Tests for merging tracklets that are in the same place at the same time."""

from __future__ import annotations

import numpy as np

from reifyscene.perception.reid.covisible import build_covisible_merge_map


def _at(x: float, frames=range(10), y: float = 0.0, z: float = 0.0):
    return {f: np.array([x, y, z]) for f in frames}


SIZE = np.array([0.05, 0.02, 0.03])  # |extent| ~ 0.062


def test_a_duplicate_mask_is_one_entity() -> None:
    """The measured case: two tracklets 0.2% of their own size apart."""
    pos = {"a": _at(0.0), "b": _at(0.0001)}
    assert build_covisible_merge_map(pos, {"a": SIZE, "b": SIZE}) == {"b": "a"}


def test_two_different_things_are_left_alone() -> None:
    """Measured contrast: genuinely distinct entities sit many times their size apart."""
    pos = {"a": _at(0.0), "b": _at(0.635)}
    assert build_covisible_merge_map(pos, {"a": SIZE, "b": SIZE}) == {}


def test_separation_is_relative_to_size_not_absolute() -> None:
    """The same gap is decisive for a small thing and meaningless for a large one.

    Monocular scale is arbitrary up to a similarity, so an absolute distance here would
    be a constant tuned to one clip.
    """
    gap = {"a": _at(0.0), "b": _at(0.02)}
    small = {"a": SIZE, "b": SIZE}                  # gap is ~32% of size
    large = {"a": SIZE * 20, "b": SIZE * 20}        # same gap, ~1.6% of size
    assert build_covisible_merge_map(gap, small) == {}
    assert build_covisible_merge_map(gap, large) == {"b": "a"}


def test_a_brief_coincidence_is_not_identity() -> None:
    pos = {"a": _at(0.0, frames=range(10)), "b": _at(0.0, frames=[3, 4])}
    assert build_covisible_merge_map(pos, {"a": SIZE, "b": SIZE}, min_shared_frames=3) == {}


def test_tracklets_that_never_overlap_are_not_this_problem() -> None:
    """Disjoint lifetimes are occlusion re-acquisition, which appearance answers."""
    pos = {"a": _at(0.0, frames=range(0, 5)), "b": _at(0.0, frames=range(20, 25))}
    assert build_covisible_merge_map(pos, {"a": SIZE, "b": SIZE}) == {}


def test_three_duplicates_fold_onto_one_id_not_a_chain() -> None:
    pos = {"a": _at(0.0), "b": _at(0.0001), "c": _at(0.0002)}
    ext = {k: SIZE for k in pos}
    assert build_covisible_merge_map(pos, ext) == {"b": "a", "c": "a"}


def test_the_canonical_is_the_earliest_appearing() -> None:
    pos = {"late": _at(0.0, frames=range(5, 15)), "early": _at(0.0001, frames=range(0, 15))}
    assert build_covisible_merge_map(pos, {"late": SIZE, "early": SIZE}) == {"late": "early"}


def test_one_outlier_frame_does_not_decide_it() -> None:
    """A median, so a single badly-placed frame neither joins nor separates two things."""
    a = _at(0.0)
    b = _at(0.0001)
    b[5] = np.array([9.0, 0.0, 0.0])  # one frame placed far away
    assert build_covisible_merge_map({"a": a, "b": b}, {"a": SIZE, "b": SIZE}) == {"b": "a"}


def test_a_track_with_no_extent_is_skipped() -> None:
    pos = {"a": _at(0.0), "b": _at(0.0001)}
    assert build_covisible_merge_map(pos, {"a": SIZE}) == {}

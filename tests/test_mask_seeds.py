"""Tests for placing tracker query points on the entities instead of over the frame."""

from __future__ import annotations

import numpy as np

from reifyscene.tracking.mask_seeds import (
    entity_seed_queries,
    queries_for_window,
    seed_count_for,
    seed_frames_for,
)


def _mask(h: int, w: int, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# --- how many, and when ------------------------------------------------------

def test_seed_count_scales_with_area_between_a_floor_and_a_ceiling() -> None:
    small = seed_count_for(10, min_points=16, per_kilopixel=2.0, max_points=256)
    mid = seed_count_for(20_000, min_points=16, per_kilopixel=2.0, max_points=256)
    huge = seed_count_for(10_000_000, min_points=16, per_kilopixel=2.0, max_points=256)
    assert small == 16          # a tiny object still gets enough to survive losses
    assert 16 < mid < 256       # a person gets more than a handbag
    assert huge == 256          # one big entity cannot take the whole budget


def test_seed_frames_are_the_first_and_every_return_after_a_gap() -> None:
    assert seed_frames_for([0, 1, 2, 3], reseed_gap=4) == [0]
    assert seed_frames_for([0, 1, 20, 21], reseed_gap=4) == [0, 20]
    assert seed_frames_for([5, 6, 8], reseed_gap=4) == [5]      # a short gap is not a return
    assert seed_frames_for([], reseed_gap=4) == []


def test_an_entity_that_appears_late_is_still_seeded() -> None:
    """The failure no grid density can fix: seeding only frame 0 misses later arrivals."""
    masks = {f: {"late": _mask(40, 40, (10, 30, 10, 30))} for f in range(20, 25)}
    q = entity_seed_queries(
        masks, frame_indices=list(range(25)), model_wh=(40, 40), source_wh=(40, 40)
    )
    assert q.shape[0] >= 16
    assert set(q[:, 0].tolist()) == {20.0}  # seeded at its first appearance, not frame 0


# --- placement ---------------------------------------------------------------

def test_queries_land_inside_the_mask_and_in_model_pixels() -> None:
    masks = {0: {"a": _mask(40, 80, (10, 30, 20, 60))}}
    q = entity_seed_queries(
        masks, frame_indices=[0], model_wh=(40, 20), source_wh=(80, 40)  # half scale
    )
    assert q.shape[0] >= 16
    assert (q[:, 1] >= 10).all() and (q[:, 1] <= 30).all()  # x: 20..60 halved
    assert (q[:, 2] >= 5).all() and (q[:, 2] <= 15).all()   # y: 10..30 halved


def test_t_indexes_the_tracked_frames_not_the_video() -> None:
    """The tracker runs on a planned frame list; a query's t must index that list."""
    masks = {10: {"a": _mask(40, 40, (10, 30, 10, 30))}}
    q = entity_seed_queries(
        masks, frame_indices=[0, 5, 10, 15], model_wh=(40, 40), source_wh=(40, 40)
    )
    assert set(q[:, 0].tolist()) == {2.0}  # frame 10 is the third tracked frame


def test_a_frame_the_tracker_skips_is_not_queried() -> None:
    masks = {7: {"a": _mask(40, 40, (10, 30, 10, 30))}}
    q = entity_seed_queries(
        masks, frame_indices=[0, 5, 10], model_wh=(40, 40), source_wh=(40, 40)
    )
    assert q.shape[0] == 0


def test_two_entities_are_seeded_independently() -> None:
    masks = {0: {"a": _mask(40, 40, (2, 18, 2, 18)), "b": _mask(40, 40, (22, 38, 22, 38))}}
    q = entity_seed_queries(
        masks, frame_indices=[0], model_wh=(40, 40), source_wh=(40, 40)
    )
    assert (q[:, 1] < 20).sum() >= 16 and (q[:, 1] > 20).sum() >= 16


def test_seeding_is_deterministic() -> None:
    masks = {0: {"a": _mask(40, 40, (5, 35, 5, 35))}}
    kw = dict(frame_indices=[0], model_wh=(40, 40), source_wh=(40, 40))
    assert np.array_equal(entity_seed_queries(masks, **kw), entity_seed_queries(masks, **kw))


def test_no_masks_is_no_queries_not_a_crash() -> None:
    out = entity_seed_queries({}, frame_indices=[0], model_wh=(4, 4), source_wh=(4, 4))
    assert out.shape == (0, 3)


# --- windowing ---------------------------------------------------------------

def test_a_window_always_gets_the_grid_even_with_no_entity_in_it() -> None:
    """Otherwise a window containing no entity would be tracked with nothing at all."""
    grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = queries_for_window(np.zeros((0, 3), np.float32), start=10, end=20, grid_xy=grid)
    assert out.shape == (2, 3)
    assert (out[:, 0] == 0).all()  # grid seeded at the window's own first frame


def test_window_queries_are_rebased_onto_the_window() -> None:
    grid = np.zeros((1, 2), dtype=np.float32)
    q = np.array([[5.0, 1.0, 1.0], [12.0, 2.0, 2.0], [25.0, 3.0, 3.0]], dtype=np.float32)
    out = queries_for_window(q, start=10, end=20, grid_xy=grid)
    entity_rows = out[1:]
    assert entity_rows.shape[0] == 1          # only the t=12 query is in [10, 20)
    assert entity_rows[0, 0] == 2.0           # rebased to window-local time


# --- the run says how it was seeded ------------------------------------------

def test_the_seeding_note_says_which_points_were_tracked() -> None:
    """A run that cannot report its own seeding is a silent degradation."""
    from reifyscene.tracking.cotracker_wrapper import _seeding_note

    seeded = _seeding_note('cotracker', np.zeros((143, 3), np.float32), 12)
    assert '143 sampled on the entity masks' in seeded

    grid_only = _seeding_note('cotracker', np.zeros((0, 3), np.float32), 12)
    assert 'uniform grid on the first frame only' in grid_only
    assert 'no instance masks' in grid_only

    assert 'heuristic' in _seeding_note('heuristic', None, 12)

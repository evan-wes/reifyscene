"""The edit-damage benchmark's scoring (reifyscene/eval/benchmarks/edit_damage).

What these pin: rest PSNR ignores the object and its ring and reads the cap when nothing else
changed; the change and taken scores divide by what they claim to; matching is one-to-one.
"""

from __future__ import annotations

import numpy as np

from reifyscene.eval.benchmarks.edit_damage import (
    change_score,
    dilate,
    match_objects,
    rest_psnr,
    summarize,
    taken_score,
)


def _box(h=20, w=20, y=(5, 10), x=(5, 10)) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[y[0]:y[1], x[0]:x[1]] = True
    return m


def test_rest_psnr_ignores_the_object_and_its_ring() -> None:
    before = np.full((20, 20, 3), 100, np.uint8)
    after = before.copy()
    obj = _box()
    after[obj] = 0                                   # the object removed
    after[obj.nonzero()[0].max() + 2, 7] = 0         # a smear just past its edge
    keep = ~dilate(obj, 3)
    assert rest_psnr(before, after, keep) == 60.0    # nothing beyond the ring changed
    after[18, 18] = 90                               # the rest touched
    assert rest_psnr(before, after, keep) < 60.0
    assert rest_psnr(before, after, np.zeros((20, 20), bool)) is None


def test_change_and_taken_scores() -> None:
    obj, other = _box(), _box(y=(12, 16), x=(12, 16))
    ch = obj.copy()
    ch[12:14, 12:16] = True                          # half the neighbour changed
    s = change_score(ch, obj, other)
    assert s["entity_removed"] == 1.0 and s["halo_ratio"] == round(8 / 25, 4)
    assert s["others_damaged"] == 0.5
    alpha = np.where(obj, 0.9, 0.0)
    alpha[0, 0] = 0.9                                # one taken pixel off the object
    t = taken_score(alpha, obj)
    assert t["taken_recall"] == 1.0 and t["taken_precision"] == round(25 / 26, 4)


def test_matching_is_one_to_one() -> None:
    obj = _box()
    gt = {0: {1: obj}}
    nodes = {0: {"a": obj, "b": obj}}                # two nodes cover one object
    matched, _, _ = match_objects(gt, nodes, match_iou=0.25)
    assert list(matched) == [1] and matched[1][0] in ("a", "b")


def test_summary_counts_and_medians() -> None:
    rows = [{"objects": [
        {"status": "scored", "rest_psnr_db": 40.0, "taken_recall": 0.9},
        {"status": "scored", "rest_psnr_db": 50.0, "taken_recall": 0.7},
        {"status": "unassignable"}, {"status": "not_proposed"}]}]
    s = summarize(rows)
    assert (s["objects"], s["scored"], s["unassignable"], s["not_proposed"]) == (4, 2, 1, 1)
    assert s["median_rest_psnr_db"] == 45.0 and s["median_taken_recall"] == 0.8

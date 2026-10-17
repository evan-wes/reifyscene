"""Tests for scoring an entity inventory against ground-truth instance masks."""

from __future__ import annotations

import numpy as np

from reifyscene.eval.benchmarks.inventory import SequenceScore, score_sequence_masks


def _mask(h, w, rows):
    m = np.zeros((h, w), bool)
    m[rows[0]:rows[1], :] = True
    return m


def test_a_clean_inventory_recovers_everything() -> None:
    gt = {1: _mask(10, 10, (0, 5)), 2: _mask(10, 10, (5, 10))}
    ours = {"a": gt[1].copy(), "b": gt[2].copy()}
    matches, missed, spurious = score_sequence_masks(gt, ours)
    assert {m["gt_id"] for m in matches} == {1, 2}
    assert missed == [] and spurious == 0


def test_two_nodes_on_one_object_count_once_and_leave_a_spurious() -> None:
    """Over-segmentation must not buy a free second match.

    Matching is one-to-one, so a duplicate mask scores as one recovery plus one node that
    matched nothing -- which is what it is.
    """
    gt = {1: _mask(10, 10, (0, 5))}
    ours = {"a": gt[1].copy(), "b": gt[1].copy()}
    matches, missed, spurious = score_sequence_masks(gt, ours)
    assert len(matches) == 1 and missed == [] and spurious == 1


def test_a_mask_below_the_threshold_does_not_count_as_found() -> None:
    gt = {1: _mask(10, 10, (0, 8))}
    barely = _mask(10, 10, (0, 3))            # IoU 3/8 = 0.375
    assert score_sequence_masks(gt, {"a": barely}, match_iou=0.5)[1] == [1]
    assert score_sequence_masks(gt, {"a": barely}, match_iou=0.3)[1] == []


def test_the_best_match_wins_when_several_overlap() -> None:
    gt = {1: _mask(10, 10, (0, 8))}
    ours = {"poor": _mask(10, 10, (0, 5)), "good": _mask(10, 10, (0, 8))}
    matches, _missed, spurious = score_sequence_masks(gt, ours)
    assert matches[0]["node"] == "good" and matches[0]["iou"] == 1.0
    assert spurious == 1


def test_nothing_found_is_reported_not_crashed() -> None:
    gt = {1: _mask(10, 10, (0, 5))}
    matches, missed, spurious = score_sequence_masks(gt, {})
    assert matches == [] and missed == [1] and spurious == 0


def test_recall_is_over_ground_truth_not_over_our_nodes() -> None:
    """Twenty background fragments must not dilute or inflate the recall of one object."""
    score = SequenceScore(sequence="s", num_gt=2, num_entities=22, recovered=1,
                          missed=1, spurious=21)
    assert score.recall == 0.5

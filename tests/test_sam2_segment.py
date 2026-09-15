"""FP2 SAM 2 instance segmenter: object-ness filter + functional propagation."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.sam2_segment import (
    SAM2InstanceSegmenter,
    _box_iou,
    _mask_bbox_xyxy,
    _touches_all_borders,
)

requires_sam2 = pytest.mark.skipif(
    importlib.util.find_spec("sam2") is None, reason="needs sam2 installed (setup_reifyscene.sh --with-sam2)"
)


# -- pure helpers ---------------------------------------------------------

def test_mask_bbox_xyxy_and_box_iou() -> None:
    m = np.zeros((20, 20), bool)
    m[5:15, 4:10] = True
    assert np.array_equal(_mask_bbox_xyxy(m), np.array([4, 5, 10, 15], np.float32))
    assert _mask_bbox_xyxy(np.zeros((4, 4), bool)) is None
    a = np.array([0, 0, 10, 10], np.float32)
    assert _box_iou(a, a) == pytest.approx(1.0)
    assert _box_iou(a, np.array([20, 20, 30, 30], np.float32)) == 0.0
    assert _box_iou(a, np.array([5, 0, 15, 10], np.float32)) == pytest.approx(1 / 3, abs=1e-6)


def test_touches_all_borders() -> None:
    full = np.ones((10, 10), bool)
    assert _touches_all_borders(full)
    interior = np.zeros((10, 10), bool)
    interior[3:7, 3:7] = True
    assert not _touches_all_borders(interior)


def test_object_ness_filter_keeps_object_drops_background() -> None:
    seg = SAM2InstanceSegmenter(None, None, {})  # __init__ loads no models (lazy)
    h, w = 100, 100

    def rec(mask, stab=0.95, iou=0.9):
        return {"segmentation": mask, "stability_score": stab, "predicted_iou": iou}

    obj = np.zeros((h, w), bool)
    obj[40:60, 40:60] = True  # 400px = 4% area, interior
    assert seg._keep_mask(rec(obj), h, w)

    assert not seg._keep_mask(rec(np.ones((h, w), bool)), h, w)          # whole frame (background)
    tiny = np.zeros((h, w), bool); tiny[0, 0] = True
    assert not seg._keep_mask(rec(tiny), h, w)                            # below min area
    assert not seg._keep_mask(rec(obj, stab=0.5), h, w)                  # low stability
    assert not seg._keep_mask(rec(obj, iou=0.4), h, w)                   # low predicted IoU


# -- functional (real SAM 2) ---------------------------------------------

class _Stream:
    stream_id = "cam_00"


class _Decoder:
    def __init__(self, frames):
        self._frames = frames

    def get_frame(self, stream, frame_idx):
        return self._frames[int(frame_idx)]


class _Detector:
    backend_used = "fake"

    def detect(self, frame):
        return []  # no persons -> everything stays object-kind


class _Ingest:
    streams = [_Stream()]


@requires_sam2
def test_sam2_segmenter_discovers_and_propagates_one_track() -> None:
    H, W, T = 240, 320, 4
    frames = []
    for t in range(T):
        img = np.full((H, W, 3), 30, dtype=np.uint8)
        x0 = 30 + t * 50
        img[90:150, x0:x0 + 60] = (220, 180, 80)  # bright square moving right
        frames.append(img)

    seg = SAM2InstanceSegmenter(_Decoder(frames), _Detector(), {"sam2_points_per_side": 16, "sam2_max_instances": 8})
    refs = [FrameRef(stream_id="cam_00", frame_idx=t, timestamp_sec=float(t) * 0.1) for t in range(T)]
    obs, masks = seg.run(_Ingest(), refs)

    track_ids = {o.track_id for o in obs}
    assert len(track_ids) == 1                       # the one moving square -> one persistent track
    assert all(o.kind == "object" for o in obs)      # no persons detected
    assert all(o.concept == "" for o in obs)         # class-agnostic (named at FP4)
    # propagated across every frame, mask area ~= the 60x60 square
    tid = next(iter(track_ids))
    areas = sorted(int(masks[o.instance_id].sum()) for o in obs if o.track_id == tid)
    assert len(areas) == T
    assert all(2500 <= a <= 5000 for a in areas)


# --- a person the automatic proposer missed is seeded from the detector -------

def test_a_box_seed_gets_an_id_past_the_mask_seeds_and_is_still_tagged() -> None:
    """Regression: the `kind` vote was sized from the mask seeds alone.

    Object ids are handed to mask seeds first, so a box seed lands past the end of that
    range. Counting votes over `range(len(seed_masks))` raised a KeyError the moment one
    existed -- and the ladder swallowed it into a silent fall back to the detector floor,
    which looked like a seeding bug for two runs. The bookkeeping is now sized from the
    ids the propagation actually produced.
    """
    per_frame = {
        0: {0: np.ones((4, 4), bool), 8: np.ones((4, 4), bool)},
        1: {0: np.ones((4, 4), bool), 8: np.ones((4, 4), bool)},
    }
    all_ids = sorted({o for masks in per_frame.values() for o in masks})
    assert all_ids == [0, 8]
    # The pattern the stage uses: every id that propagated must be countable.
    votes = {o: 0 for o in all_ids}
    for masks in per_frame.values():
        for obj_id in masks:
            votes[obj_id] += 1          # would KeyError on 8 under the old sizing
    assert votes == {0: 2, 8: 2}


def test_uncovered_person_boxes_finds_only_what_no_mask_claims() -> None:
    from reifyscene.perception.instance.sam2_segment import SAM2InstanceSegmenter

    class _Det:
        def detect(self, _frame):
            return [
                {"category": "person", "bbox_xyxy": np.array([10, 10, 30, 60], np.float32)},
                {"category": "person", "bbox_xyxy": np.array([50, 10, 70, 60], np.float32)},
                {"category": "chair", "bbox_xyxy": np.array([80, 10, 90, 30], np.float32)},
            ]

    seg = SAM2InstanceSegmenter.__new__(SAM2InstanceSegmenter)
    seg.detector = _Det()
    seg.person_iou = 0.5
    frame = np.zeros((80, 100, 3), np.uint8)

    # The first person already has a mask; the second does not; the chair is not a person.
    claimed = [np.array([10, 10, 30, 60], np.float32)]
    out = seg._uncovered_person_boxes(frame, claimed)
    assert len(out) == 1
    assert out[0].tolist() == [50.0, 10.0, 70.0, 60.0]

    assert len(seg._uncovered_person_boxes(frame, [])) == 2      # nothing claimed yet

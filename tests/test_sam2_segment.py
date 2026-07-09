"""FP2 SAM 2 instance segmenter: object-ness filter + functional propagation."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from room4d.common.types import FrameRef
from room4d.perception.instance.sam2_segment import (
    SAM2InstanceSegmenter,
    _box_iou,
    _mask_bbox_xyxy,
    _touches_all_borders,
)

requires_sam2 = pytest.mark.skipif(
    importlib.util.find_spec("sam2") is None, reason="needs sam2 installed (setup_room4d.sh --with-sam2)"
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

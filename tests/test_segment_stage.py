"""Tests for the floor `segment` stage: greedy-IoU association + assembly (FP1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.perception.instance.segment import FloorInstanceSegmenter, assign_greedy_iou_tracks


def _fr(idx: int, stream: str = 'cam0') -> FrameRef:
    return FrameRef(stream_id=stream, frame_idx=idx, timestamp_sec=float(idx))


def test_greedy_iou_links_same_object_and_splits_distinct() -> None:
    dets = [
        {'frame_ref': _fr(0), 'kind': 'object', 'bbox_xyxy': np.array([0, 0, 10, 10], float), 'confidence': 0.9},
        {'frame_ref': _fr(1), 'kind': 'object', 'bbox_xyxy': np.array([1, 0, 11, 10], float), 'confidence': 0.9},  # overlaps -> same
        {'frame_ref': _fr(1), 'kind': 'object', 'bbox_xyxy': np.array([50, 50, 60, 60], float), 'confidence': 0.8},  # far -> new
    ]
    tids = assign_greedy_iou_tracks(dets, iou_threshold=0.3, max_frame_gap=2)
    assert tids[0] == tids[1]
    assert tids[2] != tids[0]


def test_greedy_iou_separates_kinds_even_when_overlapping() -> None:
    dets = [
        {'frame_ref': _fr(0), 'kind': 'human', 'bbox_xyxy': np.array([0, 0, 10, 10], float), 'confidence': 0.9},
        {'frame_ref': _fr(1), 'kind': 'object', 'bbox_xyxy': np.array([0, 0, 10, 10], float), 'confidence': 0.9},
    ]
    tids = assign_greedy_iou_tracks(dets)
    assert tids[0] != tids[1]


def test_greedy_iou_respects_frame_gap() -> None:
    dets = [
        {'frame_ref': _fr(0), 'kind': 'object', 'bbox_xyxy': np.array([0, 0, 10, 10], float), 'confidence': 0.9},
        {'frame_ref': _fr(5), 'kind': 'object', 'bbox_xyxy': np.array([0, 0, 10, 10], float), 'confidence': 0.9},  # gap 5 > 2
    ]
    tids = assign_greedy_iou_tracks(dets, max_frame_gap=2)
    assert tids[0] != tids[1]


class _Decoder:
    def get_frame(self, stream, idx):
        return np.zeros((20, 20, 3), dtype=np.uint8)


class _Detector:
    backend_used = 'fake'

    def detect(self, frame):
        return [
            {'bbox_xyxy': np.array([0, 0, 10, 10], float), 'score': 0.9, 'category': 'person', 'label_id': 1, 'mask_logits': None},
            {'bbox_xyxy': np.array([12, 0, 20, 10], float), 'score': 0.8, 'category': 'car', 'label_id': 3, 'mask_logits': None},
        ]


class _Segmenter:
    def segment(self, frame, dets):
        for d in dets:
            m = np.zeros((20, 20), dtype=bool)
            x1, y1, x2, y2 = np.asarray(d['bbox_xyxy'], dtype=int).tolist()
            m[y1:y2, x1:x2] = True
            d['mask'] = m
        return dets


def test_floor_segmenter_splits_kinds_assigns_ids_and_masks() -> None:
    seg = FloorInstanceSegmenter(_Decoder(), _Detector(), _Segmenter(), {})
    ingest = IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=Path('.'),
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=30.0, num_frames=3, width=20, height=20, has_audio=False)],
    )
    obs, masks = seg.run(ingest, [_fr(0), _fr(1)])

    assert len(obs) == 4  # 2 detections x 2 frames
    assert {o.kind for o in obs} == {'human', 'object'}
    assert {o.concept for o in obs} == {'person', 'car'}
    # the person (same bbox both frames) stays one track; the car likewise
    assert len({o.track_id for o in obs if o.kind == 'human'}) == 1
    assert len({o.track_id for o in obs if o.kind == 'object'}) == 1
    assert all(o.instance_id in masks for o in obs)
    assert masks[obs[0].instance_id].dtype == bool

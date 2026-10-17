"""Floor-rung instance segmentation: COCO Mask R-CNN + greedy-IoU association.

The deliberately-simple **floor** of the `segment` ladder (design §4A, FP1). It
reuses today's COCO Mask R-CNN detector + segmenter to produce per-frame instance
masks, splits `person` → human-kind / everything else → object-kind, and assigns
**persistent track ids** by greedy IoU matching across consecutive frames per
stream. SAM 3 replaces this rung at FP2; the contract it fills (`InstanceArtifact`:
masks + persistent ids + concept/kind) is stable.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceObservation

_PERSON_CATEGORIES = {'person'}


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def assign_greedy_iou_tracks(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.3,
    max_frame_gap: int = 2,
) -> list[str]:
    """Assign a persistent ``track_id`` to each detection by greedy IoU matching.

    ``detections`` is a flat list of dicts, each with ``frame_ref`` (FrameRef),
    ``kind`` ('human'/'object'), ``bbox_xyxy`` (ndarray), ``confidence`` (float).
    Tracks are matched **within (stream, kind)**, ordered by frame, greedily by
    IoU to the most-recent box of each active track within ``max_frame_gap``.
    Returns a list of track ids aligned with ``detections``. Pure / testable.
    """
    track_ids: list[Optional[str]] = [None] * len(detections)
    next_idx = 0
    # group detection indices by (stream, kind)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, det in enumerate(detections):
        fr: FrameRef = det['frame_ref']
        groups.setdefault((fr.stream_id, str(det.get('kind', 'object'))), []).append(i)

    for (_stream, _kind), idxs in groups.items():
        # active tracks: track_id -> (last_frame_idx, last_bbox)
        active: dict[str, tuple[int, np.ndarray]] = {}
        idxs.sort(key=lambda i: (int(detections[i]['frame_ref'].frame_idx), -float(detections[i].get('confidence', 0.0))))
        for i in idxs:
            det = detections[i]
            frame_idx = int(det['frame_ref'].frame_idx)
            bbox = np.asarray(det['bbox_xyxy'], dtype=np.float32)
            best_tid: Optional[str] = None
            best_iou = iou_threshold
            for tid, (last_frame, last_bbox) in active.items():
                if frame_idx == last_frame or frame_idx - last_frame > max_frame_gap:
                    continue  # one match per frame; respect the gap
                iou = _bbox_iou(bbox, last_bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is None:
                best_tid = f'track_{next_idx:04d}'
                next_idx += 1
            active[best_tid] = (frame_idx, bbox)
            track_ids[i] = best_tid
    return [tid if tid is not None else f'track_{j:04d}' for j, tid in enumerate(track_ids)]


class FloorInstanceSegmenter:
    """Per-frame COCO Mask R-CNN + greedy-IoU → instance observations + masks."""

    def __init__(self, decoder, detector, segmenter, cfg: dict[str, Any] | None = None) -> None:
        self.decoder = decoder
        self.detector = detector
        self.segmenter = segmenter
        cfg = cfg or {}
        self.iou_threshold = float(cfg.get('segment_iou_threshold', 0.3))
        self.max_frame_gap = int(cfg.get('segment_max_frame_gap', 2))

    def run(self, ingest_artifact, selected_frames: list[FrameRef]) -> tuple[list[InstanceObservation], dict[str, np.ndarray]]:
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        raw: list[dict[str, Any]] = []  # per-detection scratch, before track ids
        mask_arrays: dict[str, np.ndarray] = {}
        for frame_ref in selected_frames:
            stream = stream_lookup[frame_ref.stream_id]
            frame = self.decoder.get_frame(stream, frame_ref.frame_idx)
            detections = self.detector.detect(frame)
            detections = self.segmenter.segment(frame, detections)
            for det_idx, det in enumerate(detections):
                category = str(det.get('category', 'object'))
                kind = 'human' if category in _PERSON_CATEGORIES else 'object'
                instance_id = f"inst:{frame_ref.stream_id}:{int(frame_ref.frame_idx)}:{det_idx}"
                mask = det.get('mask')
                if mask is not None:
                    mask_arrays[instance_id] = np.asarray(mask, dtype=bool)
                raw.append({
                    'instance_id': instance_id,
                    'frame_ref': frame_ref,
                    'kind': kind,
                    'concept': category,
                    'bbox_xyxy': np.asarray(det['bbox_xyxy'], dtype=np.float32),
                    'confidence': float(det.get('score', 0.0)),
                    'label_id': int(det.get('label_id', -1)),
                })

        track_ids = assign_greedy_iou_tracks(raw, iou_threshold=self.iou_threshold, max_frame_gap=self.max_frame_gap)
        observations: list[InstanceObservation] = []
        for det, track_id in zip(raw, track_ids):
            observations.append(InstanceObservation(
                instance_id=det['instance_id'],
                track_id=track_id,
                frame_ref=det['frame_ref'],
                concept=det['concept'],
                kind=det['kind'],
                bbox_xyxy=det['bbox_xyxy'],
                confidence=det['confidence'],
                attributes={'label_id': det['label_id'], 'detector_backend': self.detector.backend_used},
            ))
        return observations, mask_arrays


"""Object association across views and time."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from room4d.common.types import ObjectState
from room4d.perception.objects.object_artifact import ObjectArtifact


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


class ObjectAssociator:
    """Associate object instances using tracks and appearance."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_frame_gap = int(self.cfg.get('max_frame_gap', 2))
        self.min_match_score = float(self.cfg.get('min_match_score', 0.55))
        self.track_weight = float(self.cfg.get('track_weight', 2.0))
        self.embed_weight = float(self.cfg.get('embed_weight', 1.0))
        self.iou_weight = float(self.cfg.get('iou_weight', 0.5))
        self.distance_weight = float(self.cfg.get('distance_weight', 0.35))

    @staticmethod
    def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        av = np.asarray(a, dtype=np.float32).reshape(-1)
        bv = np.asarray(b, dtype=np.float32).reshape(-1)
        if av.size == 0 or bv.size == 0 or av.shape != bv.shape:
            return 0.0
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(av, bv) / denom)

    @staticmethod
    def _track_jaccard(a: set[int], b: set[int]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter / max(union, 1))

    @staticmethod
    def _center_distance_norm(a: np.ndarray, b: np.ndarray) -> float:
        ac = np.asarray([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float32)
        bc = np.asarray([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
        diag = max(float(np.linalg.norm(np.asarray([max(a[2], b[2]), max(a[3], b[3])], dtype=np.float32))), 1e-6)
        return float(np.linalg.norm(ac - bc) / diag)

    def _score(self, cur: ObjectState, prev: ObjectState) -> float:
        if cur.category != prev.category:
            return -1.0
        cur_tracks = {int(t) for t in cur.attributes.get('track_ids', [])}
        prev_tracks = {int(t) for t in prev.attributes.get('track_ids', [])}
        track_score = self._track_jaccard(cur_tracks, prev_tracks)
        embed_score = self._cosine(cur.embedding, prev.embedding)
        iou = _bbox_iou(np.asarray(cur.bbox_xyxy), np.asarray(prev.bbox_xyxy))
        dist = self._center_distance_norm(np.asarray(cur.bbox_xyxy), np.asarray(prev.bbox_xyxy))
        return self.track_weight * track_score + self.embed_weight * embed_score + self.iou_weight * iou - self.distance_weight * dist

    def associate(self, object_artifact: ObjectArtifact, geometry_artifact=None, track_artifact=None, sync_artifact=None) -> ObjectArtifact:
        objects = sorted(object_artifact.objects, key=lambda o: (o.frame_ref.stream_id, o.frame_ref.frame_idx, o.confidence), reverse=False)
        next_global_idx = 0
        recent_by_stream: dict[str, dict[str, ObjectState]] = {}
        updated: list[ObjectState] = []
        linked_ids: dict[str, Any] = {'detections_to_global': {}, 'global_to_detections': {}, 'matched_pairs': []}

        for obj in objects:
            stream_recent = recent_by_stream.setdefault(obj.frame_ref.stream_id, {})
            candidates = []
            for global_id, prev in stream_recent.items():
                if abs(int(obj.frame_ref.frame_idx) - int(prev.frame_ref.frame_idx)) > self.max_frame_gap * max(1, int(obj.attributes.get('frame_stride', 1))):
                    continue
                score = self._score(obj, prev)
                if score >= self.min_match_score:
                    candidates.append((score, global_id, prev))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if candidates:
                best_score, global_id, prev = candidates[0]
                new_attrs = dict(obj.attributes)
                new_attrs['global_id'] = global_id
                new_attrs['association_score'] = float(best_score)
                obj = replace(obj, object_id=global_id, attributes=new_attrs)
                linked_ids['matched_pairs'].append({
                    'current_frame': int(obj.frame_ref.frame_idx),
                    'previous_frame': int(prev.frame_ref.frame_idx),
                    'global_id': global_id,
                    'score': float(best_score),
                })
            else:
                global_id = f"obj_{next_global_idx:04d}"
                next_global_idx += 1
                new_attrs = dict(obj.attributes)
                new_attrs['global_id'] = global_id
                new_attrs['association_score'] = None
                obj = replace(obj, object_id=global_id, attributes=new_attrs)
            updated.append(obj)
            stream_recent[obj.object_id] = obj
            linked_ids['detections_to_global'][obj.attributes.get('detection_id', obj.object_id)] = obj.object_id
            linked_ids['global_to_detections'].setdefault(obj.object_id, []).append(obj.attributes.get('detection_id', obj.object_id))

        object_artifact.objects = updated
        object_artifact.linked_ids = linked_ids
        object_artifact.frame_index = object_artifact._build_frame_index(updated)
        return object_artifact

"""Object association across views and time."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    ma = np.asarray(a, dtype=bool)
    mb = np.asarray(b, dtype=bool)
    if ma.shape != mb.shape or ma.size == 0:
        return 0.0
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _TrackRecord:
    track_id: str
    detections: list[ObjectState]
    detection_ids: list[str]
    frame_indices: list[int]
    support_track_ids: set[int]
    embed_mean: np.ndarray | None
    class_hist: dict[str, float]

    @property
    def last(self) -> ObjectState:
        return self.detections[-1]


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


class ObjectAssociator:
    """Associate detections into persistent object tracks and suppress duplicates."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_frame_gap = int(self.cfg.get('max_frame_gap', 2))
        self.min_match_score = float(self.cfg.get('min_match_score', 0.55))
        self.track_weight = float(self.cfg.get('track_weight', 2.0))
        self.embed_weight = float(self.cfg.get('embed_weight', 1.0))
        self.iou_weight = float(self.cfg.get('iou_weight', 0.6))
        self.distance_weight = float(self.cfg.get('distance_weight', 0.35))
        self.class_weight = float(self.cfg.get('class_weight', 0.35))
        self.frame_duplicate_iou = float(self.cfg.get('frame_duplicate_iou', 0.55))
        self.frame_duplicate_mask_iou = float(self.cfg.get('frame_duplicate_mask_iou', 0.35))
        self.frame_duplicate_track_jaccard = float(self.cfg.get('frame_duplicate_track_jaccard', 0.25))
        self.frame_duplicate_center_dist = float(self.cfg.get('frame_duplicate_center_dist', 0.2))
        self.temporal_merge_score = float(self.cfg.get('temporal_merge_score', 1.55))
        self.temporal_merge_gap = int(self.cfg.get('temporal_merge_gap', 6))

    @staticmethod
    def _clone_object(obj: ObjectState) -> ObjectState:
        return replace(
            obj,
            bbox_xyxy=np.asarray(obj.bbox_xyxy, dtype=np.float32).copy(),
            embedding=None if obj.embedding is None else np.asarray(obj.embedding, dtype=np.float32).copy(),
            attributes=dict(obj.attributes),
        )

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

    @staticmethod
    def _rank_detection(obj: ObjectState) -> float:
        support = len({int(t) for t in obj.attributes.get('track_ids', [])})
        area = max(0.0, float(obj.bbox_xyxy[2] - obj.bbox_xyxy[0])) * max(0.0, float(obj.bbox_xyxy[3] - obj.bbox_xyxy[1]))
        return float(obj.confidence) + 0.02 * float(support) + 1e-6 * float(area)

    def _frame_duplicate_score(self, a: ObjectState, b: ObjectState, artifact: ObjectArtifact) -> tuple[float, dict[str, float]]:
        tracks_a = {int(t) for t in a.attributes.get('track_ids', [])}
        tracks_b = {int(t) for t in b.attributes.get('track_ids', [])}
        track_score = self._track_jaccard(tracks_a, tracks_b)
        bbox_iou = _bbox_iou(np.asarray(a.bbox_xyxy), np.asarray(b.bbox_xyxy))
        mask_a = artifact.mask_arrays.get(a.attributes.get('detection_id', a.object_id))
        mask_b = artifact.mask_arrays.get(b.attributes.get('detection_id', b.object_id))
        mask_iou = _mask_iou(mask_a, mask_b)
        dist = self._center_distance_norm(np.asarray(a.bbox_xyxy), np.asarray(b.bbox_xyxy))
        score = 1.5 * track_score + 1.0 * max(mask_iou, bbox_iou) - 0.5 * dist
        return score, {
            'track_jaccard': track_score,
            'bbox_iou': bbox_iou,
            'mask_iou': mask_iou,
            'center_distance': dist,
        }

    def _suppress_per_frame_duplicates(self, artifact: ObjectArtifact, raw_objects: list[ObjectState]) -> tuple[list[ObjectState], dict[str, Any]]:
        by_frame: dict[str, list[ObjectState]] = {}
        for obj in raw_objects:
            by_frame.setdefault(artifact._frame_key(obj.frame_ref), []).append(obj)

        kept: list[ObjectState] = []
        suppression_events: list[dict[str, Any]] = []
        kept_by_detection_id: dict[str, ObjectState] = {}

        for frame_key, frame_objects in sorted(by_frame.items(), key=lambda kv: (kv[0].split(':')[0], int(kv[0].split(':')[1]))):
            frame_kept: list[ObjectState] = []
            for obj in sorted(frame_objects, key=self._rank_detection, reverse=True):
                duplicate_of: ObjectState | None = None
                duplicate_metrics: dict[str, float] | None = None
                for existing in list(frame_kept):
                    score, metrics = self._frame_duplicate_score(obj, existing, artifact)
                    overlap = max(metrics['bbox_iou'], metrics['mask_iou'])
                    close = metrics['center_distance'] <= self.frame_duplicate_center_dist
                    shares_tracks = metrics['track_jaccard'] >= self.frame_duplicate_track_jaccard
                    is_duplicate = (
                        (overlap >= self.frame_duplicate_iou and (shares_tracks or close))
                        or (metrics['mask_iou'] >= self.frame_duplicate_mask_iou and (shares_tracks or close))
                        or metrics['track_jaccard'] >= 0.6
                    )
                    if is_duplicate:
                        duplicate_of = existing
                        duplicate_metrics = dict(metrics)
                        duplicate_metrics['score'] = float(score)
                        break
                if duplicate_of is None:
                    obj.attributes['suppressed'] = False
                    frame_kept.append(obj)
                    kept_by_detection_id[obj.attributes.get('detection_id', obj.object_id)] = obj
                    continue
                if self._rank_detection(obj) > self._rank_detection(duplicate_of):
                    frame_kept.remove(duplicate_of)
                    duplicate_of.attributes.update({
                        'suppressed': True,
                        'suppression_kind': 'frame_duplicate',
                        'suppressed_by_detection_id': obj.attributes.get('detection_id', obj.object_id),
                        'suppressed_by_frame': frame_key,
                        'duplicate_metrics': duplicate_metrics,
                    })
                    suppression_events.append({
                        'kind': 'frame_duplicate',
                        'frame_key': frame_key,
                        'source_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.object_id),
                        'target_detection_id': obj.attributes.get('detection_id', obj.object_id),
                        **(duplicate_metrics or {}),
                    })
                    obj.attributes['suppressed'] = False
                    frame_kept.append(obj)
                    kept_by_detection_id[obj.attributes.get('detection_id', obj.object_id)] = obj
                else:
                    obj.attributes.update({
                        'suppressed': True,
                        'suppression_kind': 'frame_duplicate',
                        'suppressed_by_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.object_id),
                        'suppressed_by_frame': frame_key,
                        'duplicate_metrics': duplicate_metrics,
                    })
                    suppression_events.append({
                        'kind': 'frame_duplicate',
                        'frame_key': frame_key,
                        'source_detection_id': obj.attributes.get('detection_id', obj.object_id),
                        'target_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.object_id),
                        **(duplicate_metrics or {}),
                    })
            kept.extend(sorted(frame_kept, key=lambda o: o.frame_ref.frame_idx))
        return kept, {
            'suppression_events': suppression_events,
            'kept_detection_ids': sorted(kept_by_detection_id.keys()),
        }

    def _class_compatibility(self, category: str, class_hist: dict[str, float]) -> float:
        if not class_hist:
            return 0.0
        total = float(sum(class_hist.values()))
        if total <= 1e-8:
            return 0.0
        p = float(class_hist.get(category, 0.0)) / total
        return 2.0 * p - 1.0

    def _score(self, cur: ObjectState, track: _TrackRecord) -> tuple[float, dict[str, float]]:
        prev = track.last
        cur_tracks = {int(t) for t in cur.attributes.get('track_ids', [])}
        track_score = self._track_jaccard(cur_tracks, track.support_track_ids)
        embed_score = self._cosine(cur.embedding, track.embed_mean)
        iou = _bbox_iou(np.asarray(cur.bbox_xyxy), np.asarray(prev.bbox_xyxy))
        dist = self._center_distance_norm(np.asarray(cur.bbox_xyxy), np.asarray(prev.bbox_xyxy))
        class_score = self._class_compatibility(cur.attributes.get('current_category', cur.category), track.class_hist)
        score = (
            self.track_weight * track_score
            + self.embed_weight * embed_score
            + self.iou_weight * iou
            + self.class_weight * class_score
            - self.distance_weight * dist
        )
        return score, {
            'track_score': track_score,
            'embed_score': embed_score,
            'iou': iou,
            'distance': dist,
            'class_score': class_score,
        }

    @staticmethod
    def _mean_embedding(detections: list[ObjectState]) -> np.ndarray | None:
        feats = [np.asarray(obj.embedding, dtype=np.float32).reshape(-1) for obj in detections if obj.embedding is not None]
        if not feats:
            return None
        arr = np.stack(feats, axis=0)
        mean = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean = mean / norm
        return mean.astype(np.float32)

    def _associate_temporally(self, detections: list[ObjectState]) -> tuple[list[ObjectState], dict[str, _TrackRecord], list[dict[str, Any]]]:
        objects = sorted(detections, key=lambda o: (o.frame_ref.stream_id, int(o.frame_ref.frame_idx), -float(o.confidence)))
        next_track_idx = 0
        recent_by_stream: dict[str, dict[str, _TrackRecord]] = {}
        matched_pairs: list[dict[str, Any]] = []
        assigned: list[ObjectState] = []

        for obj in objects:
            attrs = dict(obj.attributes)
            attrs['current_category'] = str(obj.category)
            obj = replace(obj, attributes=attrs)
            stream_recent = recent_by_stream.setdefault(obj.frame_ref.stream_id, {})
            candidates = []
            for track_id, rec in stream_recent.items():
                if abs(int(obj.frame_ref.frame_idx) - int(rec.last.frame_ref.frame_idx)) > self.max_frame_gap * max(1, int(obj.attributes.get('frame_stride', 1))):
                    continue
                score, parts = self._score(obj, rec)
                if score >= self.min_match_score:
                    candidates.append((score, track_id, rec, parts))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if candidates:
                best_score, track_id, rec, parts = candidates[0]
                rec.detections.append(obj)
                rec.detection_ids.append(obj.attributes.get('detection_id', obj.object_id))
                rec.frame_indices.append(int(obj.frame_ref.frame_idx))
                rec.support_track_ids.update(int(t) for t in obj.attributes.get('track_ids', []))
                if obj.embedding is not None:
                    rec.embed_mean = self._mean_embedding(rec.detections)
                cat = obj.attributes.get('current_category', obj.category)
                rec.class_hist[cat] = rec.class_hist.get(cat, 0.0) + float(obj.confidence)
                obj = replace(obj, object_id=track_id, attributes={**obj.attributes, 'association_score': float(best_score), 'association_parts': parts})
                matched_pairs.append({
                    'current_frame': int(obj.frame_ref.frame_idx),
                    'previous_frame': int(rec.last.frame_ref.frame_idx),
                    'track_id': track_id,
                    'score': float(best_score),
                    **parts,
                })
            else:
                track_id = f'track_{next_track_idx:04d}'
                next_track_idx += 1
                rec = _TrackRecord(
                    track_id=track_id,
                    detections=[obj],
                    detection_ids=[obj.attributes.get('detection_id', obj.object_id)],
                    frame_indices=[int(obj.frame_ref.frame_idx)],
                    support_track_ids={int(t) for t in obj.attributes.get('track_ids', [])},
                    embed_mean=None if obj.embedding is None else np.asarray(obj.embedding, dtype=np.float32).copy(),
                    class_hist={obj.attributes.get('current_category', obj.category): float(obj.confidence)},
                )
                obj = replace(obj, object_id=track_id, attributes={**obj.attributes, 'association_score': None, 'association_parts': None})
                stream_recent[track_id] = rec
            stream_recent[track_id] = rec
            assigned.append(obj)
        return assigned, {tid: rec for stream in recent_by_stream.values() for tid, rec in stream.items()}, matched_pairs

    def _track_merge_score(self, a: _TrackRecord, b: _TrackRecord) -> tuple[float, dict[str, float]]:
        tracks_score = self._track_jaccard(a.support_track_ids, b.support_track_ids)
        embed_score = self._cosine(a.embed_mean, b.embed_mean)
        classes = self._class_compatibility(max(a.class_hist, key=a.class_hist.get), b.class_hist) if a.class_hist and b.class_hist else 0.0
        frames_a = set(a.frame_indices)
        frames_b = set(b.frame_indices)
        overlap_frames = sorted(frames_a & frames_b)
        overlap_iou = 0.0
        overlap_count = 0
        if overlap_frames:
            ious = []
            map_a = {int(det.frame_ref.frame_idx): det for det in a.detections}
            map_b = {int(det.frame_ref.frame_idx): det for det in b.detections}
            for frame in overlap_frames:
                ious.append(_bbox_iou(np.asarray(map_a[frame].bbox_xyxy), np.asarray(map_b[frame].bbox_xyxy)))
            overlap_iou = float(np.mean(ious)) if ious else 0.0
            overlap_count = len(overlap_frames)
        gap = max(0, min(abs(min(b.frame_indices) - max(a.frame_indices)), abs(min(a.frame_indices) - max(b.frame_indices)))) if a.frame_indices and b.frame_indices else 9999
        continuity = 0.0
        if gap <= self.temporal_merge_gap:
            continuity = 1.0 - min(1.0, self._center_distance_norm(np.asarray(a.last.bbox_xyxy), np.asarray(b.last.bbox_xyxy)))
        score = 1.5 * tracks_score + 1.0 * embed_score + 0.5 * classes + 0.8 * overlap_iou + 0.35 * continuity
        return score, {
            'track_jaccard': tracks_score,
            'embed_score': embed_score,
            'class_score': classes,
            'overlap_iou': overlap_iou,
            'overlap_frames': float(overlap_count),
            'gap': float(gap),
            'continuity': continuity,
        }

    def _merge_temporal_duplicates(self, detections: list[ObjectState], tracks: dict[str, _TrackRecord]) -> tuple[list[ObjectState], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        track_ids = sorted(tracks.keys())
        uf = _UnionFind(track_ids)
        merge_events: list[dict[str, Any]] = []
        for i, track_a in enumerate(track_ids):
            for track_b in track_ids[i + 1:]:
                score, parts = self._track_merge_score(tracks[track_a], tracks[track_b])
                if score < self.temporal_merge_score:
                    continue
                if parts['overlap_frames'] > 0 and parts['overlap_iou'] < 0.45 and parts['track_jaccard'] < 0.5:
                    continue
                uf.union(track_a, track_b)
                merge_events.append({'source_track': track_a, 'target_track': track_b, 'score': float(score), **parts})

        groups: dict[str, list[str]] = {}
        for tid in track_ids:
            groups.setdefault(uf.find(tid), []).append(tid)

        merged_objects: list[ObjectState] = []
        summaries: dict[str, dict[str, Any]] = {}
        final_id_map: dict[str, str] = {}
        sorted_groups = sorted(groups.values(), key=lambda tids: min(min(tracks[tid].frame_indices) for tid in tids))
        for group_idx, tids in enumerate(sorted_groups):
            final_id = f'obj_{group_idx:04d}'
            all_dets: list[ObjectState] = []
            class_hist: dict[str, float] = {}
            support_tracks: set[int] = set()
            for tid in tids:
                final_id_map[tid] = final_id
                all_dets.extend(tracks[tid].detections)
                support_tracks.update(tracks[tid].support_track_ids)
                for cat, weight in tracks[tid].class_hist.items():
                    class_hist[cat] = class_hist.get(cat, 0.0) + float(weight)
            consensus = max(class_hist.items(), key=lambda kv: kv[1])[0] if class_hist else 'unknown'
            total_hist = float(sum(class_hist.values())) if class_hist else 0.0
            consensus_conf = float(class_hist.get(consensus, 0.0) / total_hist) if total_hist > 1e-8 else 0.0
            by_frame: dict[int, list[ObjectState]] = {}
            for det in all_dets:
                by_frame.setdefault(int(det.frame_ref.frame_idx), []).append(det)
            kept_ids: list[str] = []
            for frame_idx, frame_dets in sorted(by_frame.items()):
                best = max(frame_dets, key=self._rank_detection)
                best_attrs = dict(best.attributes)
                best_attrs.update({
                    'current_category': best.attributes.get('current_category', best.category),
                    'consensus_category': consensus,
                    'class_histogram': {k: float(v) for k, v in sorted(class_hist.items())},
                    'consensus_score': float(consensus_conf),
                    'support_track_ids': sorted(int(t) for t in support_tracks),
                    'support_track_count': int(len(support_tracks)),
                    'lifespan_start': int(min(by_frame.keys())),
                    'lifespan_end': int(max(by_frame.keys())),
                    'lifespan_length': int(len(by_frame)),
                    'merged_from_tracks': list(sorted(tids)),
                })
                merged_objects.append(replace(best, object_id=final_id, category=consensus, attributes=best_attrs))
                kept_ids.append(best.attributes.get('detection_id', best.object_id))
                for loser in frame_dets:
                    if loser is best:
                        continue
                    loser.attributes.update({
                        'suppressed': True,
                        'suppression_kind': 'temporal_duplicate',
                        'suppressed_by_global_id': final_id,
                        'suppressed_by_detection_id': best.attributes.get('detection_id', best.object_id),
                    })
                    merge_events.append({
                        'kind': 'temporal_duplicate',
                        'frame_idx': int(frame_idx),
                        'source_detection_id': loser.attributes.get('detection_id', loser.object_id),
                        'target_detection_id': best.attributes.get('detection_id', best.object_id),
                        'global_id': final_id,
                    })
            summaries[final_id] = {
                'global_id': final_id,
                'consensus_category': consensus,
                'class_histogram': {k: float(v) for k, v in sorted(class_hist.items())},
                'consensus_score': float(consensus_conf),
                'support_track_ids': sorted(int(t) for t in support_tracks),
                'support_track_count': int(len(support_tracks)),
                'frame_indices': sorted(int(f) for f in by_frame.keys()),
                'lifespan_start': int(min(by_frame.keys())),
                'lifespan_end': int(max(by_frame.keys())),
                'lifespan_length': int(len(by_frame)),
                'observation_count': int(sum(len(v) for v in by_frame.values())),
                'kept_detection_ids': kept_ids,
                'merged_from_tracks': list(sorted(tids)),
            }
        merged_objects = sorted(merged_objects, key=lambda o: (o.frame_ref.stream_id, int(o.frame_ref.frame_idx), o.object_id))
        return merged_objects, merge_events, summaries

    def associate(self, object_artifact: ObjectArtifact, geometry_artifact=None, track_artifact=None, sync_artifact=None) -> ObjectArtifact:
        raw_objects = [self._clone_object(obj) for obj in object_artifact.objects]
        kept_after_frame, frame_meta = self._suppress_per_frame_duplicates(object_artifact, raw_objects)
        associated, tracks, matched_pairs = self._associate_temporally(kept_after_frame)
        merged_objects, temporal_events, track_summaries = self._merge_temporal_duplicates(associated, tracks)

        detections_to_global: dict[str, str] = {}
        global_to_detections: dict[str, list[str]] = {}
        for obj in merged_objects:
            det_id = obj.attributes.get('detection_id', obj.object_id)
            detections_to_global[det_id] = obj.object_id
            global_to_detections.setdefault(obj.object_id, []).append(det_id)
        object_artifact.raw_objects = raw_objects
        object_artifact.objects = merged_objects
        object_artifact.frame_index = object_artifact._build_frame_index(merged_objects)
        object_artifact.linked_ids = {
            'detections_to_global': detections_to_global,
            'global_to_detections': global_to_detections,
            'matched_pairs': matched_pairs,
            'frame_duplicate_events': frame_meta['suppression_events'],
            'temporal_duplicate_events': temporal_events,
            'track_summaries': track_summaries,
            'raw_detection_count': len(raw_objects),
            'kept_detection_count': len(kept_after_frame),
            'final_detection_count': len(merged_objects),
            'suppressed_detection_count': len([obj for obj in raw_objects if obj.attributes.get('suppressed')]),
        }
        object_artifact.notes.extend([
            f"frame_duplicate_events={len(frame_meta['suppression_events'])}",
            f"temporal_duplicate_events={len([e for e in temporal_events if e.get('kind') == 'temporal_duplicate'])}",
            f"persistent_object_tracks={len(track_summaries)}",
        ])
        return object_artifact

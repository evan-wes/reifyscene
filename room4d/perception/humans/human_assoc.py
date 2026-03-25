"""Temporal human association and cleanup for single-video perception."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from room4d.common.types import HumanState
from room4d.perception.humans.human_artifact import HumanArtifact


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


def _track_jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / max(union, 1))


@dataclass
class _HumanTrackRecord:
    human_id: str
    detections: list[HumanState]
    detection_ids: list[str]
    frame_indices: list[int]
    support_track_ids: set[int]

    @property
    def last(self) -> HumanState:
        return self.detections[-1]


class HumanAssociator:
    """Associate human detections across time within a single stream."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_frame_gap = int(self.cfg.get('max_frame_gap', 3))
        self.min_match_score = float(self.cfg.get('min_match_score', 0.35))
        self.track_weight = float(self.cfg.get('track_weight', 2.0))
        self.iou_weight = float(self.cfg.get('iou_weight', 0.7))
        self.center_weight = float(self.cfg.get('center_weight', 0.35))
        self.support_weight = float(self.cfg.get('support_weight', 0.85))
        self.frame_duplicate_iou = float(self.cfg.get('frame_duplicate_iou', 0.55))
        self.frame_duplicate_mask_iou = float(self.cfg.get('frame_duplicate_mask_iou', 0.3))
        self.frame_duplicate_track_jaccard = float(self.cfg.get('frame_duplicate_track_jaccard', 0.25))
        self.frame_duplicate_center_dist = float(self.cfg.get('frame_duplicate_center_dist', 0.2))
        self.smoothing_window = int(self.cfg.get('smoothing_window', 1))

    @staticmethod
    def _frame_key(h: HumanState) -> str:
        return f"{h.frame_ref.stream_id}:{int(h.frame_ref.frame_idx)}"

    @staticmethod
    def _center_xyxy(box: np.ndarray) -> np.ndarray:
        return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

    @staticmethod
    def _center(h: HumanState) -> np.ndarray:
        return HumanAssociator._center_xyxy(np.asarray(h.bbox_xyxy, dtype=np.float32))

    @staticmethod
    def _clone_human(h: HumanState) -> HumanState:
        return replace(
            h,
            bbox_xyxy=np.asarray(h.bbox_xyxy, dtype=np.float32).copy(),
            keypoints_2d=None if h.keypoints_2d is None else np.asarray(h.keypoints_2d, dtype=np.float32).copy(),
            densepose=None if h.densepose is None else dict(h.densepose),
            smpl_params=None if h.smpl_params is None else dict(h.smpl_params),
            attributes=dict(h.attributes),
        )

    @staticmethod
    def _support_quality(attrs: dict[str, Any]) -> tuple[float, dict[str, float]]:
        support = float(attrs.get('support_track_count', len(attrs.get('track_ids', []))))
        mask_support = float(attrs.get('mask_support_count', support))
        bbox_support = float(attrs.get('bbox_support_count', support))
        visible = max(float(attrs.get('visible_track_count', support)), 1.0)
        ratio = support / visible
        mask_ratio = mask_support / max(bbox_support, 1.0)
        score = 0.6 * min(1.0, ratio * 3.0) + 0.4 * min(1.0, mask_ratio)
        return float(score), {
            'support_ratio': float(ratio),
            'mask_support_ratio': float(mask_ratio),
            'support_count': float(support),
        }

    @staticmethod
    def _visible_keypoints(h: HumanState) -> int:
        if h.keypoints_2d is None:
            return 0
        return int(np.sum(np.asarray(h.keypoints_2d, dtype=np.float32)[:, 2] > 0.05))

    @staticmethod
    def _quality_score(confidence: float, support_ratio: float, visible_keypoints: int, dp_count: int, smpl_valid: bool) -> float:
        kp_term = min(1.0, float(visible_keypoints) / 12.0)
        dp_term = min(1.0, float(dp_count) / 128.0)
        smpl_term = 1.0 if smpl_valid else 0.0
        return float(0.25 * confidence + 0.30 * min(1.0, support_ratio * 3.0) + 0.25 * kp_term + 0.10 * dp_term + 0.10 * smpl_term)

    def _frame_duplicate_score(self, a: HumanState, b: HumanState, artifact: HumanArtifact) -> tuple[float, dict[str, float]]:
        tracks_a = {int(t) for t in a.attributes.get('track_ids', [])}
        tracks_b = {int(t) for t in b.attributes.get('track_ids', [])}
        track_score = _track_jaccard(tracks_a, tracks_b)
        bbox_iou = _bbox_iou(np.asarray(a.bbox_xyxy), np.asarray(b.bbox_xyxy))
        mask_a = artifact.mask_arrays.get(a.attributes.get('detection_id', a.human_id))
        mask_b = artifact.mask_arrays.get(b.attributes.get('detection_id', b.human_id))
        mask_iou = _mask_iou(mask_a, mask_b)
        center_dist = float(np.linalg.norm(self._center(a) - self._center(b)))
        diag = max(float(np.linalg.norm(np.asarray([max(a.bbox_xyxy[2], b.bbox_xyxy[2]), max(a.bbox_xyxy[3], b.bbox_xyxy[3])], dtype=np.float32))), 1e-6)
        center_norm = center_dist / diag
        support_a, _ = self._support_quality(a.attributes)
        support_b, _ = self._support_quality(b.attributes)
        score = 1.3 * track_score + 1.1 * max(mask_iou, bbox_iou) + 0.4 * max(support_a, support_b) - 0.5 * center_norm
        return score, {
            'track_jaccard': track_score,
            'bbox_iou': bbox_iou,
            'mask_iou': mask_iou,
            'center_distance': center_norm,
            'support_score_a': support_a,
            'support_score_b': support_b,
        }

    def _suppress_per_frame_duplicates(self, artifact: HumanArtifact, raw_humans: list[HumanState]) -> tuple[list[HumanState], dict[str, Any]]:
        by_frame: dict[str, list[HumanState]] = {}
        for h in raw_humans:
            by_frame.setdefault(self._frame_key(h), []).append(h)

        kept: list[HumanState] = []
        suppression_events: list[dict[str, Any]] = []
        for frame_key, frame_humans in sorted(by_frame.items(), key=lambda kv: (kv[0].split(':')[0], int(kv[0].split(':')[1]))):
            frame_kept: list[HumanState] = []
            ranked = sorted(frame_humans, key=lambda h: (h.confidence, h.attributes.get('support_track_count', 0), self._visible_keypoints(h)), reverse=True)
            for h in ranked:
                duplicate_of: HumanState | None = None
                duplicate_metrics: dict[str, float] | None = None
                for existing in list(frame_kept):
                    score, metrics = self._frame_duplicate_score(h, existing, artifact)
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
                    h.attributes['suppressed'] = False
                    frame_kept.append(h)
                    continue
                rank_h = float(h.confidence) + 0.05 * h.attributes.get('support_track_count', 0) + 0.01 * self._visible_keypoints(h)
                rank_e = float(duplicate_of.confidence) + 0.05 * duplicate_of.attributes.get('support_track_count', 0) + 0.01 * self._visible_keypoints(duplicate_of)
                if rank_h > rank_e:
                    frame_kept.remove(duplicate_of)
                    duplicate_of.attributes.update({
                        'suppressed': True,
                        'suppression_kind': 'frame_duplicate',
                        'suppressed_by_detection_id': h.attributes.get('detection_id', h.human_id),
                        'suppressed_by_frame': frame_key,
                        'duplicate_metrics': duplicate_metrics,
                    })
                    suppression_events.append({
                        'kind': 'frame_duplicate',
                        'frame_key': frame_key,
                        'source_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.human_id),
                        'target_detection_id': h.attributes.get('detection_id', h.human_id),
                        **(duplicate_metrics or {}),
                    })
                    h.attributes['suppressed'] = False
                    frame_kept.append(h)
                else:
                    h.attributes.update({
                        'suppressed': True,
                        'suppression_kind': 'frame_duplicate',
                        'suppressed_by_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.human_id),
                        'suppressed_by_frame': frame_key,
                        'duplicate_metrics': duplicate_metrics,
                    })
                    suppression_events.append({
                        'kind': 'frame_duplicate',
                        'frame_key': frame_key,
                        'source_detection_id': h.attributes.get('detection_id', h.human_id),
                        'target_detection_id': duplicate_of.attributes.get('detection_id', duplicate_of.human_id),
                        **(duplicate_metrics or {}),
                    })
            kept.extend(sorted(frame_kept, key=lambda h: int(h.frame_ref.frame_idx)))
        return kept, {'suppression_events': suppression_events}

    def _score(self, candidate: HumanState, existing: HumanState) -> tuple[float, dict[str, float]]:
        tracks_a = {int(t) for t in candidate.attributes.get('track_ids', [])}
        tracks_b = {int(t) for t in existing.attributes.get('track_ids', [])}
        track_score = _track_jaccard(tracks_a, tracks_b)
        iou = _bbox_iou(np.asarray(candidate.bbox_xyxy, dtype=np.float32), np.asarray(existing.bbox_xyxy, dtype=np.float32))
        center_dist = float(np.linalg.norm(self._center(candidate) - self._center(existing)))
        diag = max(float(np.linalg.norm(np.asarray([existing.bbox_xyxy[2] - existing.bbox_xyxy[0], existing.bbox_xyxy[3] - existing.bbox_xyxy[1]], dtype=np.float32))), 1e-6)
        center_penalty = center_dist / diag
        support_score, support_parts = self._support_quality(candidate.attributes)
        score = self.track_weight * track_score + self.iou_weight * iou + self.support_weight * support_score - self.center_weight * center_penalty
        return score, {'track_score': track_score, 'iou': iou, 'center_penalty': center_penalty, 'support_score': support_score, **support_parts}

    @staticmethod
    def _smooth_boxes(humans: list[HumanState], window: int = 1) -> list[np.ndarray]:
        boxes = [np.asarray(h.bbox_xyxy, dtype=np.float32) for h in humans]
        out: list[np.ndarray] = []
        for idx in range(len(boxes)):
            lo = max(0, idx - window)
            hi = min(len(boxes), idx + window + 1)
            out.append(np.mean(np.stack(boxes[lo:hi], axis=0), axis=0).astype(np.float32))
        return out

    @staticmethod
    def _smooth_keypoints(seq: list[HumanState], idx: int, window: int = 1) -> np.ndarray | None:
        if seq[idx].keypoints_2d is None:
            return None
        lo = max(0, idx - window)
        hi = min(len(seq), idx + window + 1)
        neighbors = [h.keypoints_2d for h in seq[lo:hi] if h.keypoints_2d is not None]
        if not neighbors:
            return None
        kpts = [np.asarray(k, dtype=np.float32) for k in neighbors]
        out = np.asarray(seq[idx].keypoints_2d, dtype=np.float32).copy()
        for j in range(out.shape[0]):
            pts = []
            confs = []
            for arr in kpts:
                if j < arr.shape[0] and float(arr[j, 2]) > 0.05:
                    pts.append(arr[j, :2])
                    confs.append(float(arr[j, 2]))
            if pts:
                pts_arr = np.stack(pts, axis=0)
                out[j, :2] = pts_arr.mean(axis=0)
                out[j, 2] = float(np.mean(confs))
            else:
                out[j, 2] = 0.0
        return out

    @staticmethod
    def _smooth_smpl(seq: list[HumanState], idx: int, window: int = 1) -> dict[str, Any] | None:
        base = seq[idx].smpl_params
        if base is None:
            return None
        lo = max(0, idx - window)
        hi = min(len(seq), idx + window + 1)
        vals = [h.smpl_params for h in seq[lo:hi] if h.smpl_params and h.smpl_params.get('valid', False)]
        if not vals:
            return dict(base)
        out = dict(base)
        for key in ('betas', 'body_pose', 'global_orient', 'transl', 'scale'):
            arrays = [np.asarray(v[key], dtype=np.float32) for v in vals if key in v]
            if arrays:
                out[key] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        out['valid'] = True
        out['smoothed'] = True
        out['valid_ratio'] = float(np.mean([1.0 if v.get('valid', False) else 0.0 for v in vals]))
        return out

    @staticmethod
    def _smooth_densepose(seq: list[HumanState], idx: int, smoothed_box: np.ndarray) -> dict[str, Any] | None:
        base = seq[idx].densepose
        if base is None:
            return None
        out = dict(base)
        uv = np.asarray(base.get('uv', np.zeros((0, 2))), dtype=np.float32)
        part_ids = np.asarray(base.get('part_ids', np.zeros((uv.shape[0],), dtype=np.int32)), dtype=np.int32)
        x1, y1, x2, y2 = smoothed_box.tolist()
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        if uv.size:
            samples_xy = np.stack([x1 + uv[:, 0] * bw, y1 + uv[:, 1] * bh], axis=1).astype(np.float32)
        else:
            samples_xy = np.zeros((0, 2), dtype=np.float32)
        out['samples_xy'] = samples_xy
        out['uv'] = uv
        out['part_ids'] = part_ids
        out['smoothed'] = True
        return out

    def _smooth_track(self, human_id: str, seq: list[HumanState]) -> tuple[list[HumanState], dict[str, Any]]:
        sorted_seq = sorted(seq, key=lambda h: int(h.frame_ref.frame_idx))
        smoothed_boxes = self._smooth_boxes(sorted_seq, window=self.smoothing_window)
        smoothed: list[HumanState] = []
        support_ids = sorted({int(t) for h in sorted_seq for t in h.attributes.get('track_ids', [])})
        support_counts: list[int] = []
        support_density: list[float] = []
        vis_kpts: list[int] = []
        dp_counts: list[int] = []
        smpl_valid: list[float] = []
        confidences: list[float] = []
        duplicate_count = 0
        for idx, h in enumerate(sorted_seq):
            smooth_box = smoothed_boxes[idx]
            attrs = dict(h.attributes)
            attrs['raw_bbox_xyxy'] = np.asarray(h.bbox_xyxy, dtype=np.float32).tolist()
            attrs['smoothed_bbox_xyxy'] = np.asarray(smooth_box, dtype=np.float32).tolist()
            attrs['support_track_ids'] = support_ids
            support_count = int(attrs.get('support_track_count', len(attrs.get('track_ids', []))))
            visible_tracks = max(int(attrs.get('visible_track_count', support_count)), 1)
            support_counts.append(support_count)
            support_density.append(float(support_count) / float(visible_tracks))
            if attrs.get('suppressed'):
                duplicate_count += 1
            smooth_kpts = self._smooth_keypoints(sorted_seq, idx, window=self.smoothing_window)
            visible_kpts = 0 if smooth_kpts is None else int(np.sum(np.asarray(smooth_kpts)[:, 2] > 0.05))
            vis_kpts.append(visible_kpts)
            smooth_smpl = self._smooth_smpl(sorted_seq, idx, window=self.smoothing_window)
            smpl_valid.append(1.0 if (smooth_smpl or {}).get('valid', False) else 0.0)
            smooth_dp = self._smooth_densepose(sorted_seq, idx, smooth_box)
            dp_count = int(np.asarray((smooth_dp or {}).get('samples_xy', np.zeros((0, 2)))).shape[0])
            dp_counts.append(dp_count)
            confidence = float(h.confidence)
            confidences.append(confidence)
            support_quality, support_parts = self._support_quality(attrs)
            attrs.update(support_parts)
            attrs['quality_score'] = self._quality_score(confidence, support_parts['support_ratio'], visible_kpts, dp_count, bool((smooth_smpl or {}).get('valid', False)))
            attrs['support_track_count'] = len(support_ids)
            attrs['support_track_ids'] = support_ids
            smoothed.append(replace(h, human_id=human_id, bbox_xyxy=smooth_box, keypoints_2d=smooth_kpts, densepose=smooth_dp, smpl_params=smooth_smpl, attributes=attrs))
        summary = {
            'lifespan_start': int(sorted_seq[0].frame_ref.frame_idx),
            'lifespan_end': int(sorted_seq[-1].frame_ref.frame_idx),
            'num_observations': len(sorted_seq),
            'support_track_count': len(support_ids),
            'support_track_ids': support_ids,
            'mean_support_track_count': float(np.mean(support_counts)) if support_counts else 0.0,
            'mean_support_density': float(np.mean(support_density)) if support_density else 0.0,
            'mean_visible_keypoints': float(np.mean(vis_kpts)) if vis_kpts else 0.0,
            'mean_densepose_samples': float(np.mean(dp_counts)) if dp_counts else 0.0,
            'smpl_valid_ratio': float(np.mean(smpl_valid)) if smpl_valid else 0.0,
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'duplicate_suppression_count': int(duplicate_count),
            'quality_score': float(np.mean([s.attributes.get('quality_score', 0.0) for s in smoothed])) if smoothed else 0.0,
        }
        return smoothed, summary

    def associate_across_views(self, human_artifact: HumanArtifact, geometry_artifact=None, sync_artifact=None):
        raw = [self._clone_human(h) for h in human_artifact.humans]
        kept_after_frame, frame_meta = self._suppress_per_frame_duplicates(human_artifact, raw)
        by_frame: dict[str, list[HumanState]] = {}
        for h in kept_after_frame:
            by_frame.setdefault(self._frame_key(h), []).append(h)
        track_records: dict[str, list[HumanState]] = {}
        linked: list[HumanState] = []
        matched_pairs: list[dict[str, Any]] = []
        next_id = 0
        for frame_key in sorted(by_frame.keys(), key=lambda fk: (fk.split(':')[0], int(fk.split(':')[1]))):
            humans = by_frame[frame_key]
            for human in humans:
                best_id = None
                best_score = -1e9
                best_parts = None
                for human_id, seq in track_records.items():
                    last = seq[-1]
                    if last.frame_ref.stream_id != human.frame_ref.stream_id:
                        continue
                    if int(human.frame_ref.frame_idx) - int(last.frame_ref.frame_idx) > self.max_frame_gap:
                        continue
                    score, parts = self._score(human, last)
                    if score > best_score:
                        best_score = score
                        best_id = human_id
                        best_parts = parts
                if best_id is None or best_score < self.min_match_score:
                    best_id = f'human_{next_id:03d}'
                    next_id += 1
                    track_records[best_id] = []
                else:
                    matched_pairs.append({
                        'human_id': best_id,
                        'frame_idx': int(human.frame_ref.frame_idx),
                        'score': float(best_score),
                        **(best_parts or {}),
                    })
                human.attributes['association_score'] = float(best_score if np.isfinite(best_score) else 0.0)
                human.attributes['association_parts'] = best_parts
                track_records[best_id].append(human)
                linked.append(replace(human, human_id=best_id))

        smoothed_humans: list[HumanState] = []
        track_summaries: dict[str, Any] = {}
        updated_densepose = dict(human_artifact.densepose_payloads)
        updated_smpl = dict(human_artifact.smpl_payloads)
        for human_id, seq in track_records.items():
            smooth_seq, summary = self._smooth_track(human_id, seq)
            smoothed_humans.extend(smooth_seq)
            track_summaries[human_id] = summary
            for h in smooth_seq:
                det_id = h.attributes.get('detection_id', h.human_id)
                if h.densepose is not None:
                    updated_densepose[det_id] = h.densepose
                if h.smpl_params is not None:
                    updated_smpl[det_id] = h.smpl_params

        artifact = HumanArtifact(
            run_id=human_artifact.run_id,
            stage_name=human_artifact.stage_name,
            save_dir=human_artifact.save_dir,
            humans=sorted(smoothed_humans, key=lambda h: (h.frame_ref.stream_id, int(h.frame_ref.frame_idx), h.human_id)),
            raw_humans=raw,
            linked_ids={
                'track_summaries': track_summaries,
                'frame_duplicate_events': frame_meta['suppression_events'],
                'matched_pairs': matched_pairs,
                'raw_detection_count': len(raw),
                'kept_detection_count': len(kept_after_frame),
                'final_detection_count': len(smoothed_humans),
                'suppressed_detection_count': len([h for h in raw if h.attributes.get('suppressed')]),
            },
            frame_index={},
            mask_arrays=human_artifact.mask_arrays,
            densepose_payloads=updated_densepose,
            smpl_payloads=updated_smpl,
            notes=list(human_artifact.notes),
            metadata=dict(human_artifact.metadata),
        )
        artifact.frame_index = artifact._build_frame_index(artifact.humans)
        artifact.notes.extend([
            f"frame_duplicate_events={len(frame_meta['suppression_events'])}",
            f"persistent_human_tracks={len(track_summaries)}",
        ])
        return artifact

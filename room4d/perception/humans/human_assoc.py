"""Cross-view and temporal human association."""

from __future__ import annotations

from dataclasses import replace
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


def _track_jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / max(union, 1))


class HumanAssociator:
    """Associate human detections across time within a single stream."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.max_frame_gap = int(self.cfg.get('max_frame_gap', 3))
        self.min_match_score = float(self.cfg.get('min_match_score', 0.35))
        self.track_weight = float(self.cfg.get('track_weight', 2.0))
        self.iou_weight = float(self.cfg.get('iou_weight', 0.75))
        self.center_weight = float(self.cfg.get('center_weight', 0.35))

    @staticmethod
    def _frame_key(h: HumanState) -> str:
        return f"{h.frame_ref.stream_id}:{int(h.frame_ref.frame_idx)}"

    @staticmethod
    def _center(h: HumanState) -> np.ndarray:
        return np.asarray([(h.bbox_xyxy[0] + h.bbox_xyxy[2]) * 0.5, (h.bbox_xyxy[1] + h.bbox_xyxy[3]) * 0.5], dtype=np.float32)

    def _score(self, candidate: HumanState, existing: HumanState) -> float:
        tracks_a = {int(t) for t in candidate.attributes.get('track_ids', [])}
        tracks_b = {int(t) for t in existing.attributes.get('track_ids', [])}
        track_score = _track_jaccard(tracks_a, tracks_b)
        iou = _bbox_iou(np.asarray(candidate.bbox_xyxy, dtype=np.float32), np.asarray(existing.bbox_xyxy, dtype=np.float32))
        center_dist = float(np.linalg.norm(self._center(candidate) - self._center(existing)))
        diag = max(float(np.linalg.norm(np.asarray([existing.bbox_xyxy[2] - existing.bbox_xyxy[0], existing.bbox_xyxy[3] - existing.bbox_xyxy[1]], dtype=np.float32))), 1e-6)
        center_penalty = center_dist / diag
        return self.track_weight * track_score + self.iou_weight * iou - self.center_weight * center_penalty

    def associate_across_views(self, human_artifact: HumanArtifact, geometry_artifact=None, sync_artifact=None):
        raw = [replace(h, bbox_xyxy=np.asarray(h.bbox_xyxy, dtype=np.float32).copy(), keypoints_2d=None if h.keypoints_2d is None else np.asarray(h.keypoints_2d, dtype=np.float32).copy(), attributes=dict(h.attributes)) for h in human_artifact.humans]
        by_frame: dict[str, list[HumanState]] = {}
        for h in raw:
            by_frame.setdefault(self._frame_key(h), []).append(h)
        track_records: dict[str, list[HumanState]] = {}
        linked: list[HumanState] = []
        next_id = 0
        for frame_key in sorted(by_frame.keys(), key=lambda fk: (fk.split(':')[0], int(fk.split(':')[1]))):
            humans = by_frame[frame_key]
            for human in humans:
                best_id = None
                best_score = -1e9
                for human_id, seq in track_records.items():
                    last = seq[-1]
                    if last.frame_ref.stream_id != human.frame_ref.stream_id:
                        continue
                    if int(human.frame_ref.frame_idx) - int(last.frame_ref.frame_idx) > self.max_frame_gap:
                        continue
                    score = self._score(human, last)
                    if score > best_score:
                        best_score = score
                        best_id = human_id
                if best_id is None or best_score < self.min_match_score:
                    best_id = f'human_{next_id:03d}'
                    next_id += 1
                    track_records[best_id] = []
                track_records[best_id].append(human)
                human.attributes['association_score'] = float(best_score if np.isfinite(best_score) else 0.0)
                linked.append(replace(human, human_id=best_id))
        track_summaries: dict[str, Any] = {}
        for human_id, seq in track_records.items():
            support_tracks = sorted({int(t) for h in seq for t in h.attributes.get('track_ids', [])})
            visible_kpts = [int(np.sum(np.asarray(h.keypoints_2d)[:, 2] > 0.05)) for h in seq if h.keypoints_2d is not None]
            track_summaries[human_id] = {
                'lifespan_start': int(seq[0].frame_ref.frame_idx),
                'lifespan_end': int(seq[-1].frame_ref.frame_idx),
                'num_observations': len(seq),
                'support_track_count': len(support_tracks),
                'support_track_ids': support_tracks,
                'mean_visible_keypoints': float(np.mean(visible_kpts)) if visible_kpts else 0.0,
            }
        artifact = HumanArtifact(
            run_id=human_artifact.run_id,
            stage_name=human_artifact.stage_name,
            save_dir=human_artifact.save_dir,
            humans=linked,
            raw_humans=raw,
            linked_ids={'track_summaries': track_summaries},
            frame_index={},
            mask_arrays=human_artifact.mask_arrays,
            densepose_payloads=human_artifact.densepose_payloads,
            smpl_payloads=human_artifact.smpl_payloads,
            notes=list(human_artifact.notes),
            metadata=dict(human_artifact.metadata),
        )
        artifact.frame_index = artifact._build_frame_index(artifact.humans)
        return artifact

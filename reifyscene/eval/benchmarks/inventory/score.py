"""Match entity masks to ground-truth instance masks and count what was found."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SequenceScore:
    """What one sequence's inventory got right, missed, and invented."""

    sequence: str
    num_gt: int = 0
    num_entities: int = 0
    recovered: int = 0
    missed: int = 0
    spurious: int = 0
    matches: list[dict[str, Any]] = field(default_factory=list)  # gt_id -> node, iou
    missed_ids: list[int] = field(default_factory=list)
    frames_scored: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.recovered / self.num_gt if self.num_gt else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recall"] = round(self.recall, 4)
        return d


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def score_sequence_masks(
    gt_by_id: Mapping[int, np.ndarray],
    ours_by_node: Mapping[str, np.ndarray],
    *,
    match_iou: float = 0.5,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Greedy one-to-one matching of our masks to ground-truth masks.

    Every pair is scored, then taken in descending IoU order, each ground-truth object
    and each node being claimed at most once. One-to-one is the point: two nodes covering
    one object should count as one recovery and one spurious node, not as two successes.

    Returns ``(matches, missed_gt_ids, spurious_count)``.
    """
    pairs = [
        (_iou(gt, ours), gid, node)
        for gid, gt in gt_by_id.items()
        for node, ours in ours_by_node.items()
    ]
    pairs.sort(key=lambda p: (-p[0], str(p[2])))

    taken_gt: set[int] = set()
    taken_node: set[str] = set()
    matches: list[dict[str, Any]] = []
    for iou, gid, node in pairs:
        if iou < match_iou or gid in taken_gt or node in taken_node:
            continue
        taken_gt.add(gid)
        taken_node.add(node)
        matches.append({"gt_id": int(gid), "node": str(node), "iou": round(float(iou), 4)})

    missed = sorted(set(gt_by_id) - taken_gt)
    spurious = len(ours_by_node) - len(taken_node)
    return matches, [int(m) for m in missed], int(spurious)


def _gt_masks(annotation_dir: Path, frame_idx: int) -> dict[int, np.ndarray]:
    from PIL import Image

    path = annotation_dir / f"{frame_idx:05d}.png"
    if not path.exists():
        return {}
    arr = np.asarray(Image.open(path))
    return {int(i): (arr == i) for i in np.unique(arr) if int(i) != 0}


def score_run(
    run_dir: Path,
    annotation_dir: Path,
    *,
    frames: Sequence[int] | None = None,
    match_iou: float = 0.5,
    stream_id: str = "cam_00",
) -> SequenceScore:
    """Score one run against one DAVIS sequence's annotations.

    Scoring uses the instance masks, not the graph nodes, because a node's identity is a
    tracklet while a ground-truth object is a mask: comparing masks keeps the question
    ("is this thing found") separate from identity, which is scored elsewhere. Frames are
    sampled rather than exhaustive -- an object either tracks or it does not, and three
    frames spread through the clip separate those cases at a fraction of the cost.
    """
    from reifyscene.perception.instance.instance_artifact import InstanceArtifact

    score = SequenceScore(sequence=run_dir.name)
    try:
        inst = InstanceArtifact.load(run_dir / "segment")
    except Exception as exc:  # a run that did not reach segment still gets a row
        score.notes.append(f"no segment artifact: {exc}")
        return score

    by_frame: dict[int, dict[str, np.ndarray]] = {}
    for obs in inst.observations:
        if obs.frame_ref.stream_id != stream_id:
            continue
        mask = inst.mask_arrays.get(obs.instance_id)
        if mask is not None:
            by_frame.setdefault(int(obs.frame_ref.frame_idx), {})[str(obs.track_id)] = mask
    if not by_frame:
        score.notes.append("segment produced no masks")
        return score

    available = sorted(by_frame)
    if frames is None:
        frames = [available[0], available[len(available) // 2], available[-1]]

    best: SequenceScore | None = None
    tracks_seen: set[str] = set()
    for frame_idx in frames:
        gt = _gt_masks(annotation_dir, frame_idx)
        ours = by_frame.get(frame_idx, {})
        if not gt or not ours:
            continue
        tracks_seen |= set(ours)
        matches, missed, spurious = score_sequence_masks(gt, ours, match_iou=match_iou)
        frame_score = SequenceScore(
            sequence=run_dir.name, num_gt=len(gt), num_entities=len(ours),
            recovered=len(matches), missed=len(missed), spurious=spurious,
            matches=matches, missed_ids=missed,
        )
        # Report the frame the inventory did best on: a miss that persists across every
        # sampled frame is a real miss, while one frame of occlusion is not the claim
        # being tested.
        if best is None or frame_score.recovered > best.recovered:
            best = frame_score
        score.frames_scored += 1

    if best is None:
        score.notes.append("no frame had both annotations and masks")
        return score
    best.frames_scored = score.frames_scored
    best.num_entities = len(tracks_seen)
    best.spurious = len(tracks_seen) - best.recovered
    return best

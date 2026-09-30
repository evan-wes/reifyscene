"""Match entity masks to ground-truth instance masks and count what was found."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


GroundTruth = Callable[[int], dict[int, np.ndarray]]


def stream_ground_truth(stream) -> GroundTruth:
    """``frame -> {object id: mask}`` from a stream's source annotations (source pixels, the
    same raster segment masks are in); an unannotated frame gives ``{}``."""
    frames = set(stream.frames)

    def gt(frame_idx: int) -> dict[int, np.ndarray]:
        if int(frame_idx) not in frames:
            return {}
        arr = stream.annotation(int(frame_idx))
        if arr is None:
            return {}
        return {int(i): (arr == i) for i in np.unique(arr) if int(i) != 0}

    return gt


def score_run(
    run_dir: Path,
    *,
    frames: Sequence[int] | None = None,
    match_iou: float = 0.5,
    stream_id: str | None = None,
) -> SequenceScore:
    """Score one run against its input's own annotations (the stream's source: DAVIS).

    Scoring uses the instance masks, not the graph nodes, because a node's identity is a
    tracklet while a ground-truth object is a mask: comparing masks keeps the question
    ("is this thing found") separate from identity, which is scored elsewhere. Frames are
    sampled rather than exhaustive -- an object either tracks or it does not, and three
    frames spread through the clip separate those cases at a fraction of the cost.
    """
    from reifyscene.ingest.streams import RunStreams
    from reifyscene.perception.instance.instance_artifact import InstanceArtifact

    stream = RunStreams.from_run(run_dir).resolve(stream_id)
    stream_id = stream.id
    sequence = stream.source.sequence or run_dir.name
    score = SequenceScore(sequence=sequence)
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

    return score_masks_by_frame(
        by_frame, stream_ground_truth(stream), sequence=sequence, frames=frames,
        match_iou=match_iou, notes=score.notes,
    )


def score_masks_by_frame(
    by_frame: Mapping[int, Mapping[str, np.ndarray]],
    ground_truth: GroundTruth,
    *,
    sequence: str,
    frames: Sequence[int] | None = None,
    match_iou: float = 0.5,
    notes: Sequence[str] | None = None,
) -> SequenceScore:
    """Score any system's per-frame instance masks against one DAVIS sequence.

    This is the whole of the scoring policy -- which frames are sampled, how a frame is
    matched, how the frames are combined -- and it takes plain ``{frame: {id: mask}}`` so
    that a reference implementation and this pipeline traverse *the same code*. A
    comparison in which each side is scored by its own function is not a comparison.

    **An object counts as recovered if some entity matched it on any sampled frame, and the
    denominator is every ground-truth object annotated on any sampled frame.** A miss that
    persists across every sampled frame is a real miss; one frame of occlusion is not the
    claim being tested.

    Both halves of that are deliberate, because an earlier version got them wrong in a way
    that only shows up when two systems are compared. It reported whichever frame the system
    scored best on, so the denominator was *that* frame's object count -- and two systems
    peaking on different frames were then scored against different ground truth (on
    ``lindy-hop``: ours against 8 objects, the reference against 5). It also skipped frames
    where the system produced no masks at all, which quietly excused a system from the
    objects it found nothing on. Both are now frame-set properties rather than system
    properties, so any two systems given the same ``frames`` are measured against the same
    denominator.
    """
    score = SequenceScore(sequence=sequence, notes=list(notes or []))
    available = sorted(by_frame)
    if not available:
        score.notes.append("no masks to score")
        return score
    if frames is None:
        frames = [available[0], available[len(available) // 2], available[-1]]

    gt_ids: set[int] = set()
    matched: dict[int, dict[str, Any]] = {}
    entities: set[str] = set()
    for frame_idx in frames:
        gt = ground_truth(frame_idx)
        if not gt:
            continue
        score.frames_scored += 1
        gt_ids |= set(gt)
        # Note the frame is counted and its objects are in the denominator even when this
        # system produced nothing here.
        ours = by_frame.get(frame_idx, {})
        if not ours:
            continue
        entities |= set(ours)
        for m in score_sequence_masks(gt, ours, match_iou=match_iou)[0]:
            gid = int(m["gt_id"])
            if gid not in matched or m["iou"] > matched[gid]["iou"]:
                matched[gid] = {**m, "frame_idx": int(frame_idx)}

    if not gt_ids:
        score.notes.append("no sampled frame had annotations")
        return score

    score.num_gt = len(gt_ids)
    score.num_entities = len(entities)
    score.recovered = len(matched)
    score.missed_ids = sorted(gt_ids - set(matched))
    score.missed = len(score.missed_ids)
    score.matches = [matched[g] for g in sorted(matched)]
    score.spurious = len(entities) - score.recovered
    return score

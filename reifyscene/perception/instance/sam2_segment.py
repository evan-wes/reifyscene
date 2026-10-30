"""SAM 2 instance segmentation: class-agnostic discovery + video propagation.

The FP2 top rung of the `segment` ladder (design §4A), above the Mask R-CNN +
greedy-IoU floor. Prompt-free and class-agnostic: SAM 2 automatic mask generation
discovers object-like masks on a keyframe (no class list), a tunable object-ness
filter prunes them, and the SAM 2 video predictor propagates each kept mask across
the stream with a **persistent native ID** (replacing the floor's frame-to-frame
IoU stitching). Naming stays downstream (FP4 semantics); this rung emits only
masks + identity. The existing person detector tags `kind='human'` so the
`humans` stage keeps working class-agnostically.

Fills the same `InstanceArtifact` contract the floor does (`(observations,
mask_arrays)`), so it slots into the ladder behind an `importable('sam2')` probe.
SAM 2 is the ungated stand-in until SAM 3 (the eventual top rung) lands.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceObservation

_DEFAULT_CKPT = Path(__file__).resolve().parents[3] / "third_party" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


def _touches_all_borders(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def write_video_frames(tmp_dir: Path, images: list[np.ndarray]) -> None:
    """SAM 2's video predictor reads a directory of integer-named JPEGs."""
    for i, img in enumerate(images):
        Image.fromarray(img).save(tmp_dir / f"{i:05d}.jpg", quality=95)


def vote_human_kind(
    per_frame: dict[int, dict[int, np.ndarray]], num_frames: int, person_boxes_at, person_iou: float,
    obj_ids=None,
) -> dict[int, str]:
    """``{obj_id: 'human' | 'object'}``: human when a track overlaps a person box in at least half the
    frames it appears in.

    ``person_boxes_at(pos)`` returns the person detections on frame ``pos``; it is only called for
    frames where one of ``obj_ids`` (default: every tracked id) is present.
    """
    all_ids = sorted({o for masks in per_frame.values() for o in masks}) if obj_ids is None else sorted(obj_ids)
    wanted = set(all_ids)
    human_votes: dict[int, int] = {o: 0 for o in all_ids}
    present: dict[int, int] = {o: 0 for o in all_ids}
    for pos in range(num_frames):
        obj_masks = {o: m for o, m in per_frame.get(pos, {}).items() if o in wanted and m.any()}
        if not obj_masks:
            continue
        pboxes = person_boxes_at(pos)
        for obj_id, m in obj_masks.items():
            present[obj_id] += 1
            bb = _mask_bbox_xyxy(m)
            if bb is not None and pboxes and max(_box_iou(bb, pb) for pb in pboxes) >= person_iou:
                human_votes[obj_id] += 1
    return {
        o: ("human" if present[o] and human_votes[o] >= (present[o] + 1) // 2 else "object")
        for o in all_ids
    }


class SAM2InstanceSegmenter:
    """SAM 2 automatic discovery + video propagation → instance masks + IDs.

    Constructor mirrors ``FloorInstanceSegmenter`` (``decoder`` for frames,
    ``detector`` reused for person-tagging); ``segmenter`` is unused (SAM 2 owns
    masks). Models are lazily loaded on first ``run`` so constructing the rung is
    cheap and import-safe when SAM 2 is absent (the ladder probe gates execution).
    """

    def __init__(self, decoder, detector, cfg: dict[str, Any] | None = None) -> None:
        self.decoder = decoder
        self.detector = detector
        cfg = cfg or {}
        # Object-ness filter (tunable) — prune SAM 2's everything-segmentation to
        # object-like instances; entity-vs-background is finally settled by motion
        # (dynamic selection) + FP4 semantics.
        self.min_area_frac = float(cfg.get("sam2_min_area_frac", 0.001))
        self.max_area_frac = float(cfg.get("sam2_max_area_frac", 0.5))
        self.min_stability = float(cfg.get("sam2_min_stability", 0.90))
        self.min_predicted_iou = float(cfg.get("sam2_min_predicted_iou", 0.80))
        self.max_instances = int(cfg.get("sam2_max_instances", 20))
        self.points_per_side = int(cfg.get("sam2_points_per_side", 32))
        self.person_iou = float(cfg.get("sam2_person_iou", 0.5))
        # Every Nth frame is checked for a person no track accounts for. 1 is exact
        # and costs a detector pass per frame; the detector already runs on every
        # frame for the `kind` vote, so this shares that cost rather than adding to it.
        self.person_seed_stride = int(cfg.get("sam2_person_seed_stride", 1))
        # Seeding people who arrive after the keyframe is OFF until it is correct. The
        # first attempt compared each detection against boxes claimed on earlier frames,
        # and a walking person's box drifts away from her own earlier one, so she was
        # re-seeded every few frames: davis_lab-coat went from 8 tracks to 54. Catching a
        # late arrival needs re-propagating after each round of seeds, not one pass over
        # stale boxes. The keyframe seeding below is independent of this and is on.
        # OFF by default, on evidence rather than caution. Two things argue against it:
        # DAVIS has no late arrivals to gain from -- 205 of its 206 annotated objects are
        # present on frame 0, and all 63 the inventory misses are -- and enabling it
        # regressed davis_lab-coat, where SAM 2 raises "mat1 and mat2 must have the same
        # dtype" on a second propagation and the ladder falls back to the detector floor.
        # The loop itself is correct and unit-tested; it is the repeat propagation
        # underneath it that is not ready, and there is no measurable upside here to pay
        # for finding out in production.
        self.person_seed_late = bool(cfg.get("sam2_person_seed_late", False))
        self.person_seed_rounds = int(cfg.get("sam2_person_seed_rounds", 3))
        # How many consecutive checked frames a person must go unaccounted for before she
        # is treated as a new arrival. One frame is not evidence: a tracked person whose
        # mask degrades briefly falls below person_iou against her own detection box and
        # looks uncovered, which on lab-coat is exactly what fired the loop when nobody
        # had arrived at all.
        self.person_seed_patience = int(cfg.get("sam2_person_seed_patience", 3))
        self._person_box_cache: dict[int, list[np.ndarray]] = {}
        self.checkpoint = str(cfg.get("sam2_checkpoint", _DEFAULT_CKPT))
        self.config = str(cfg.get("sam2_config", _DEFAULT_CONFIG))
        self.device = str(cfg.get("sam2_device", "cuda"))
        self._amg = None
        self._video_predictor = None

    # -- model loading ----------------------------------------------------
    def _load_models(self) -> None:
        if self._amg is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            model = build_sam2(self.config, self.checkpoint, device=self.device)
            self._amg = SAM2AutomaticMaskGenerator(model, points_per_side=self.points_per_side)
        self._load_video_predictor()

    def _load_video_predictor(self) -> None:
        if self._video_predictor is None:
            from sam2.build_sam import build_sam2_video_predictor

            self._video_predictor = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)

    def track_seeds(self, images: list[np.ndarray], seed_masks: list[np.ndarray]) -> dict[int, dict[int, np.ndarray]]:
        """Track keyframe mask seeds through ``images``: ``{frame position: {seed index: mask}}``.

        The propagation alone, for a rung that finds its own seeds (SAM 3 on the first frame).
        """
        self._load_video_predictor()
        with tempfile.TemporaryDirectory(prefix="sam2_seg_") as tmp:
            tmp_dir = Path(tmp)
            write_video_frames(tmp_dir, images)
            return self._propagate(tmp_dir, seed_masks, {})

    # -- object-ness filter ----------------------------------------------
    def _keep_mask(self, record: dict[str, Any], h: int, w: int) -> bool:
        seg = np.asarray(record["segmentation"], dtype=bool)
        area_frac = float(seg.sum()) / float(h * w)
        if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
            return False
        if float(record.get("stability_score", 0.0)) < self.min_stability:
            return False
        if float(record.get("predicted_iou", 0.0)) < self.min_predicted_iou:
            return False
        if _touches_all_borders(seg):
            return False  # background-like (spans the whole frame)
        return True

    def _discover_keyframe_masks(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Auto-discover + filter object-like masks on the keyframe (largest first)."""
        h, w = frame_rgb.shape[:2]
        records = self._amg.generate(frame_rgb)
        kept = [r for r in records if self._keep_mask(r, h, w)]
        kept.sort(key=lambda r: float(r.get("area", 0.0)), reverse=True)
        return [np.asarray(r["segmentation"], dtype=bool) for r in kept[: self.max_instances]]

    # -- per-stream segmentation -----------------------------------------
    def _decode_stream_frames(self, stream, frames: list[FrameRef]) -> list[np.ndarray]:
        return [np.asarray(self.decoder.get_frame(stream, fr.frame_idx))[..., :3].astype(np.uint8) for fr in frames]

    def cached_person_boxes(self, images: list[np.ndarray]):
        """``pos -> person boxes on images[pos]``, running the detector at most once per frame."""
        def at(pos: int) -> list[np.ndarray]:
            if pos not in self._person_box_cache:
                self._person_box_cache[pos] = self._person_boxes(images[pos])
            return self._person_box_cache[pos]
        return at

    def _person_boxes(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        try:
            dets = self.detector.detect(frame_rgb)
        except Exception:
            return []
        out = []
        for d in dets:
            if str(d.get("category", "")) == "person":
                out.append(np.asarray(d["bbox_xyxy"], dtype=np.float32))
        return out

    def _propagate(
        self, tmp_dir: Path, seed_masks: list[np.ndarray],
        box_seeds: dict[int, list[np.ndarray]],
    ) -> dict[int, dict[int, np.ndarray]]:
        """Track every seed through the clip. Masks seed the keyframe, boxes seed a frame.

        Object ids are assigned to the mask seeds first, so adding a box seed never
        renumbers an existing track.
        """
        state = self._video_predictor.init_state(str(tmp_dir))
        next_id = 0
        for mask in seed_masks:
            self._video_predictor.add_new_mask(
                state, frame_idx=0, obj_id=next_id, mask=mask
            )
            next_id += 1
        for pos in sorted(box_seeds):
            for box in box_seeds[pos]:
                self._video_predictor.add_new_points_or_box(
                    state, frame_idx=int(pos), obj_id=next_id,
                    box=np.asarray(box, dtype=np.float32),
                )
                next_id += 1

        per_frame: dict[int, dict[int, np.ndarray]] = {}
        for frame_pos, obj_ids, mask_logits in self._video_predictor.propagate_in_video(state):
            masks = (mask_logits > 0.0).squeeze(1).detach().cpu().numpy()  # (num_obj, H, W)
            per_frame[int(frame_pos)] = {
                int(o): masks[k].astype(bool) for k, o in enumerate(obj_ids)
            }
        return per_frame

    def _uncovered_person_boxes(
        self, frame_rgb: np.ndarray, existing: list[np.ndarray | None],
        *, boxes: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Person detections on this frame that no existing track accounts for.

        ``boxes`` lets a caller pass detections it already has, so a clip that is checked
        over several rounds runs the detector once per frame rather than once per round.
        """
        if boxes is None:
            boxes = self._person_boxes(frame_rgb)
        if not boxes:
            return []
        known = [b for b in existing if b is not None]
        return [
            box for box in boxes
            if not known or max(_box_iou(box, k) for k in known) < self.person_iou
        ]

    def _earliest_uncovered_people(
        self, images: list[np.ndarray], per_frame: dict[int, dict[int, np.ndarray]]
    ) -> tuple[int, list[np.ndarray]] | None:
        """The first frame holding a person no current track accounts for.

        Only the earliest such frame is returned, and the caller re-propagates before
        asking again. That is what keeps one person from becoming many: seeding every
        uncovered detection across every frame in a single pass re-seeds a walking person
        each time her box drifts from the one already claimed for her, which turned eight
        tracks into fifty-four. After a propagation she is tracked, so she stops being
        uncovered and the next round moves on to someone else.
        """
        streak = 0
        first_seen: tuple[int, list[np.ndarray]] | None = None
        for pos in range(0, len(images), max(1, self.person_seed_stride)):
            if pos not in self._person_box_cache:
                self._person_box_cache[pos] = self._person_boxes(images[pos])
            if not self._person_box_cache[pos]:
                streak, first_seen = 0, None
                continue
            tracked = [
                _mask_bbox_xyxy(m) for m in per_frame.get(pos, {}).values() if m.any()
            ]
            uncovered = self._uncovered_person_boxes(
                images[pos], tracked, boxes=self._person_box_cache[pos]
            )
            if not uncovered:
                streak, first_seen = 0, None
                continue
            streak += 1
            if first_seen is None:
                first_seen = (pos, uncovered)
            if streak >= max(1, self.person_seed_patience):
                return first_seen  # seed where she was first unaccounted for
        return None

    def _seed_late_arrivals(
        self, tmp_dir: Path, images: list[np.ndarray], seed_masks: list[np.ndarray],
        box_seeds: dict[int, list[np.ndarray]],
        per_frame: dict[int, dict[int, np.ndarray]],
    ) -> dict[int, dict[int, np.ndarray]]:
        """Seed people who arrive after the keyframe, re-propagating between rounds.

        Each round costs one propagation and seeds at least one person, so the loop is
        bounded by ``person_seed_rounds``. When nothing arrives late -- the common case --
        the first check finds nothing and no extra propagation happens at all.
        """
        for _ in range(max(0, self.person_seed_rounds)):
            found = self._earliest_uncovered_people(images, per_frame)
            if found is None:
                break
            pos, boxes = found
            box_seeds.setdefault(pos, []).extend(boxes)
            per_frame = self._propagate(tmp_dir, seed_masks, box_seeds)
        return per_frame

    def _segment_stream(
        self, stream, stream_id: str, frames: list[FrameRef],
    ) -> tuple[list[InstanceObservation], dict[str, np.ndarray]]:
        images = self._decode_stream_frames(stream, frames)
        if not images:
            return [], {}
        # Detections are cached by frame position, which only means anything within one
        # stream's frame list.
        self._person_box_cache = {}
        h, w = images[0].shape[:2]
        seed_masks = self._discover_keyframe_masks(images[0])

        # A person the automatic proposer missed on the keyframe is missed for the whole
        # clip: nothing downstream can propose an entity, so re-id, tracking and the graph
        # all inherit the omission. The detector is already being run on every frame to
        # vote on `kind` -- on davis_lab-coat it finds all three people at confidence 1.00
        # while the proposer masks two of them -- so let it seed as well as label.
        box_seeds: dict[int, list[np.ndarray]] = {}
        first_boxes = self._uncovered_person_boxes(
            images[0], [(_mask_bbox_xyxy(m)) for m in seed_masks]
        )
        if first_boxes:
            box_seeds[0] = first_boxes
        if not seed_masks and not box_seeds:
            return [], {}

        with tempfile.TemporaryDirectory(prefix="sam2_seg_") as tmp:
            tmp_dir = Path(tmp)
            write_video_frames(tmp_dir, images)

            per_frame = self._propagate(tmp_dir, seed_masks, box_seeds)
            # Second pass for anyone who arrives after the keyframe. Only run when the
            # first pass actually left a detection unaccounted for, so the common case
            # costs one propagation.
            if self.person_seed_late:
                per_frame = self._seed_late_arrivals(
                    tmp_dir, images, seed_masks, box_seeds, per_frame
                )

        # Person-tag tracks: an object is human if it overlaps a person box in a
        # majority of the frames it appears in (keeps the humans stage working).
        # Sized from the ids the propagation actually produced, not from the mask seeds:
        # a box seed gets an id past the end of that range, and counting on the seed list
        # made the whole rung raise a KeyError the moment one existed -- which the ladder
        # then swallowed into a silent fall back to the detector floor.
        # The late-arrival check already ran the detector on these frames; reuse it so
        # looking for uncovered people costs nothing the kind vote was not paying.
        kind_of = vote_human_kind(per_frame, len(images), self.cached_person_boxes(images), self.person_iou)

        observations: list[InstanceObservation] = []
        mask_arrays: dict[str, np.ndarray] = {}
        for pos, frame_ref in enumerate(frames):
            for obj_id, m in per_frame.get(pos, {}).items():
                if not m.any():
                    continue
                instance_id = f"inst:{stream_id}:{int(frame_ref.frame_idx)}:{obj_id}"
                mask_arrays[instance_id] = m
                bb = _mask_bbox_xyxy(m)
                observations.append(InstanceObservation(
                    instance_id=instance_id,
                    track_id=f"sam2:{stream_id}:{obj_id}",
                    frame_ref=frame_ref,
                    concept="",  # class-agnostic; named at FP4
                    kind=kind_of[obj_id],
                    bbox_xyxy=bb if bb is not None else np.zeros((4,), dtype=np.float32),
                    confidence=1.0,
                    attributes={"segmenter_backend": "sam2", "obj_id": int(obj_id)},
                ))
        return observations, mask_arrays

    # -- entry point ------------------------------------------------------
    def run(self, ingest_artifact, selected_frames: list[FrameRef]) -> tuple[list[InstanceObservation], dict[str, np.ndarray]]:
        self._load_models()
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        by_stream: dict[str, list[FrameRef]] = {}
        for fr in selected_frames:
            by_stream.setdefault(fr.stream_id, []).append(fr)

        observations: list[InstanceObservation] = []
        mask_arrays: dict[str, np.ndarray] = {}
        for stream_id, frames in by_stream.items():
            frames = sorted(frames, key=lambda fr: int(fr.frame_idx))
            obs, masks = self._segment_stream(stream_lookup[stream_id], stream_id, frames)
            observations.extend(obs)
            mask_arrays.update(masks)
        return observations, mask_arrays

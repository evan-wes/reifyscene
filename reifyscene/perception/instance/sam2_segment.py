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
        self.person_seed_late = bool(cfg.get("sam2_person_seed_late", False))
        self.checkpoint = str(cfg.get("sam2_checkpoint", _DEFAULT_CKPT))
        self.config = str(cfg.get("sam2_config", _DEFAULT_CONFIG))
        self.device = str(cfg.get("sam2_device", "cuda"))
        self._amg = None
        self._video_predictor = None

    # -- model loading ----------------------------------------------------
    def _load_models(self) -> None:
        if self._amg is not None and self._video_predictor is not None:
            return
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        model = build_sam2(self.config, self.checkpoint, device=self.device)
        self._amg = SAM2AutomaticMaskGenerator(model, points_per_side=self.points_per_side)
        self._video_predictor = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)

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
        self, frame_rgb: np.ndarray, existing: list[np.ndarray | None]
    ) -> list[np.ndarray]:
        """Person detections on this frame that no existing track accounts for."""
        boxes = self._person_boxes(frame_rgb)
        if not boxes:
            return []
        known = [b for b in existing if b is not None]
        return [
            box for box in boxes
            if not known or max(_box_iou(box, k) for k in known) < self.person_iou
        ]

    def _late_person_seeds(
        self, images: list[np.ndarray], per_frame: dict[int, dict[int, np.ndarray]]
    ) -> dict[int, list[np.ndarray]]:
        """Where a person appears that the keyframe seeds never covered.

        **Known wrong, and off by default** (``sam2_person_seed_late``). The intent is one
        seed per late arrival; the implementation compares each detection against boxes
        claimed on earlier frames, and a walking person's box drifts away from her own
        earlier one, so she is re-seeded again and again -- 8 tracks became 54 on
        davis_lab-coat. Doing this properly means re-propagating after each round of
        seeds so coverage is judged against where the tracks actually are, which is a
        loop of propagations rather than one pass. Kept because the shape is right and
        the fix is known.
        """
        found: dict[int, list[np.ndarray]] = {}
        claimed: list[np.ndarray] = []
        for pos in range(0, len(images), max(1, self.person_seed_stride)):
            tracked = [
                _mask_bbox_xyxy(m) for m in per_frame.get(pos, {}).values() if m.any()
            ]
            for box in self._uncovered_person_boxes(images[pos], tracked + claimed):
                found.setdefault(pos, []).append(box)
                claimed.append(box)
        return found

    def _segment_stream(
        self, stream, stream_id: str, frames: list[FrameRef],
    ) -> tuple[list[InstanceObservation], dict[str, np.ndarray]]:
        images = self._decode_stream_frames(stream, frames)
        if not images:
            return [], {}
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

        # SAM 2's video predictor reads a directory of integer-named JPEGs.
        with tempfile.TemporaryDirectory(prefix="sam2_seg_") as tmp:
            tmp_dir = Path(tmp)
            for i, img in enumerate(images):
                Image.fromarray(img).save(tmp_dir / f"{i:05d}.jpg", quality=95)

            per_frame = self._propagate(tmp_dir, seed_masks, box_seeds)
            # Second pass for anyone who arrives after the keyframe. Only run when the
            # first pass actually left a detection unaccounted for, so the common case
            # costs one propagation.
            late = self._late_person_seeds(images, per_frame) if self.person_seed_late else {}
            if late:
                for pos, boxes in late.items():
                    box_seeds.setdefault(pos, []).extend(boxes)
                per_frame = self._propagate(tmp_dir, seed_masks, box_seeds)

        # Person-tag tracks: an object is human if it overlaps a person box in a
        # majority of the frames it appears in (keeps the humans stage working).
        # Sized from the ids the propagation actually produced, not from the mask seeds:
        # a box seed gets an id past the end of that range, and counting on the seed list
        # made the whole rung raise a KeyError the moment one existed -- which the ladder
        # then swallowed into a silent fall back to the detector floor.
        all_ids = sorted({o for masks in per_frame.values() for o in masks})
        human_votes: dict[int, int] = {o: 0 for o in all_ids}
        present: dict[int, int] = {o: 0 for o in all_ids}
        for pos in range(len(images)):
            obj_masks = per_frame.get(pos, {})
            if not obj_masks:
                continue
            pboxes = self._person_boxes(images[pos])
            for obj_id, m in obj_masks.items():
                if not m.any():
                    continue
                present[obj_id] += 1
                bb = _mask_bbox_xyxy(m)
                if bb is not None and pboxes and max(_box_iou(bb, pb) for pb in pboxes) >= self.person_iou:
                    human_votes[obj_id] += 1
        kind_of = {
            o: ("human" if present[o] and human_votes[o] >= (present[o] + 1) // 2 else "object")
            for o in all_ids
        }

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

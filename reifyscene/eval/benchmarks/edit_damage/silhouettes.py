"""People's silhouettes in photos, independent of any removal -- pseudo ground truth.

A held-out DyCheck view has a real photo and a covisibility mask but no instance masks, so
"where is the person in this view" has to come from the photo itself. Without it, a removal
can only be judged against its own footprint, which is circular: it cannot fail.

Each person in a photo is found by the person detector (the pipeline's own
:class:`~reifyscene.perception.humans.detector.HumanDetector`) and segmented by SAM 2's image
predictor prompted with the detection box. SAM 2 is optional: without it the detector's own
mask is used, and every silhouette records which one produced it. These are model outputs, not
annotation -- reports say so and save the masks as images to be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Silhouette:
    mask: np.ndarray                 # bool (H, W)
    box: np.ndarray                  # xyxy
    score: float
    source: str                      # "sam2 (box prompt)" or "detector mask"


@dataclass
class PersonSegmenter:
    """Detector + SAM 2 box prompt, loaded once, reused for every photo."""

    device: str = "cuda"
    score_threshold: float = 0.5
    _detector: Any = field(default=None, repr=False)
    _predictor: Any = field(default=None, repr=False)
    backend: str = "unloaded"

    def _load(self) -> None:
        if self._detector is not None:
            return
        from reifyscene.perception.humans.detector import HumanDetector

        self._detector = HumanDetector({"backend_mode": "real", "device": self.device,
                                        "score_threshold": self.score_threshold})
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            from reifyscene.perception.instance.sam2_segment import (
                _DEFAULT_CKPT,
                _DEFAULT_CONFIG,
            )

            if not _DEFAULT_CKPT.is_file():
                raise FileNotFoundError(_DEFAULT_CKPT)
            self._predictor = SAM2ImagePredictor(
                build_sam2(_DEFAULT_CONFIG, str(_DEFAULT_CKPT), device=self.device))
            self.backend = "sam2 (box prompt)"
        except (ImportError, FileNotFoundError, RuntimeError):
            self._predictor = None
            self.backend = "detector mask"

    def people(self, rgb: np.ndarray) -> list[Silhouette]:
        """Every person the detector finds in ``rgb`` (uint8 HWC), segmented."""
        self._load()
        h, w = rgb.shape[:2]
        dets = [d for d in self._detector.detect(rgb) if d.get("category") == "person"]
        if not dets:
            return []
        out: list[Silhouette] = []
        if self._predictor is not None:
            import torch

            with torch.inference_mode():
                self._predictor.set_image(rgb)
                for d in dets:
                    masks, _, _ = self._predictor.predict(
                        box=np.asarray(d["bbox_xyxy"], np.float32)[None],
                        multimask_output=False)
                    out.append(Silhouette(np.asarray(masks[0]) > 0.5,
                                          np.asarray(d["bbox_xyxy"]), float(d["score"]),
                                          self.backend))
            return out
        for d in dets:
            logits = d.get("mask_logits")
            if logits is None:
                continue
            m = np.asarray(logits)
            m = m[0] if m.ndim == 3 else m
            if m.shape != (h, w):
                continue                         # never resized to fit
            out.append(Silhouette(m > 0.5, np.asarray(d["bbox_xyxy"]), float(d["score"]),
                                  self.backend))
        return out


def pick_entity(people: list[Silhouette], hint: np.ndarray) -> tuple[Silhouette | None, float]:
    """The person whose silhouette best overlaps ``hint`` (where the entity renders), and that
    IoU. The hint only chooses among independent silhouettes; it never shapes one."""
    best, best_iou = None, 0.0
    for p in people:
        inter = int((p.mask & hint).sum())
        union = int((p.mask | hint).sum())
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best, best_iou = p, iou
    return best, best_iou

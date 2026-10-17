
"""Instance segmentation wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np


class ObjectSegmenter:
    """Generate binary masks for detected object regions."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.mask_threshold = float(self.cfg.get('mask_threshold', 0.5))

    def segment(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        image = np.asarray(frame)
        h, w = image.shape[:2]
        segmented: list[dict[str, Any]] = []
        for det in detections:
            bbox = np.asarray(det['bbox_xyxy'], dtype=np.float32)
            x1, y1, x2, y2 = bbox.tolist()
            xi1 = max(0, min(w - 1, int(np.floor(x1))))
            yi1 = max(0, min(h - 1, int(np.floor(y1))))
            xi2 = max(xi1 + 1, min(w, int(np.ceil(x2))))
            yi2 = max(yi1 + 1, min(h, int(np.ceil(y2))))
            mask_logits = det.get('mask_logits')
            if mask_logits is not None:
                mask = np.asarray(mask_logits, dtype=np.float32) >= self.mask_threshold
                if mask.shape != (h, w):
                    from PIL import Image
                    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), resample=Image.BILINEAR)) > 127
            else:
                mask = np.zeros((h, w), dtype=bool)
                mask[yi1:yi2, xi1:xi2] = True
            det_out = dict(det)
            det_out['mask'] = mask.astype(bool)
            segmented.append(det_out)
        return segmented

"""DensePose wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np


class DensePoseEstimator:
    """Predict dense human surface correspondences.

    The default implementation returns a heuristic dense correspondence set from
    the detected mask and/or keypoints so the pipeline can debug surface-aligned
    reasoning without requiring DensePose weights.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('densepose_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self.max_samples = int(self.cfg.get('densepose_max_samples', 256))
        self._backend_used = 'heuristic'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def _heuristic_densepose(self, frame: np.ndarray, boxes: list[np.ndarray], masks: list[np.ndarray | None], keypoints: list[np.ndarray | None]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        h, w = frame.shape[:2]
        for box, mask, kpts in zip(boxes, masks, keypoints):
            box = np.asarray(box, dtype=np.float32)
            x1, y1, x2, y2 = box.tolist()
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            if mask is None or np.asarray(mask).shape[:2] != (h, w):
                mask_arr = np.zeros((h, w), dtype=bool)
                xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
                xi2, yi2 = min(w, max(xi1 + 1, int(round(x2)))), min(h, max(yi1 + 1, int(round(y2))))
                mask_arr[yi1:yi2, xi1:xi2] = True
            else:
                mask_arr = np.asarray(mask, dtype=bool)
            ys, xs = np.where(mask_arr)
            if xs.size == 0:
                samples_xy = np.zeros((0, 2), dtype=np.float32)
                uv = np.zeros((0, 2), dtype=np.float32)
                part_ids = np.zeros((0,), dtype=np.int32)
            else:
                if xs.size > self.max_samples:
                    step = max(1, xs.size // self.max_samples)
                    xs = xs[::step][: self.max_samples]
                    ys = ys[::step][: self.max_samples]
                samples_xy = np.stack([xs, ys], axis=1).astype(np.float32)
                u = ((samples_xy[:, 0] - x1) / bw).clip(0.0, 1.0)
                v = ((samples_xy[:, 1] - y1) / bh).clip(0.0, 1.0)
                uv = np.stack([u, v], axis=1).astype(np.float32)
                # Simple body-part bucketing from normalized coordinates.
                part_ids = np.ones((samples_xy.shape[0],), dtype=np.int32) * 2  # torso
                part_ids[v < 0.18] = 1  # head
                part_ids[(u < 0.35) & (v >= 0.18) & (v < 0.75)] = 3  # left arm / side
                part_ids[(u > 0.65) & (v >= 0.18) & (v < 0.75)] = 4  # right arm / side
                part_ids[(u < 0.48) & (v >= 0.68)] = 5  # left leg
                part_ids[(u >= 0.48) & (v >= 0.68)] = 6  # right leg
            results.append({
                'backend': 'heuristic',
                'samples_xy': samples_xy,
                'uv': uv,
                'part_ids': part_ids,
                'bbox_xyxy': box.astype(np.float32),
                'keypoints_used': None if kpts is None else int(np.sum(np.asarray(kpts)[:, 2] > 0.05)),
            })
        self._backend_used = 'heuristic'
        return results

    def predict(self, frame: np.ndarray, boxes: list[np.ndarray], masks: list[np.ndarray | None] | None = None, keypoints: list[np.ndarray | None] | None = None) -> list[dict[str, Any]]:
        masks = masks or [None] * len(boxes)
        keypoints = keypoints or [None] * len(boxes)
        # Real DensePose integration can be added later without changing the artifact contract.
        return self._heuristic_densepose(frame, boxes, masks, keypoints)

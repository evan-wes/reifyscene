"""2D human pose estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


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


class HumanPose2DEstimator:
    """Predict 2D body keypoints for detected humans.

    The real backend uses torchvision Keypoint R-CNN and matches the predicted
    person instances to the provided boxes by IoU.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('pose_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.4))
        self._model = None
        self._backend_used = 'uninitialized'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def _load_real_model(self):
        if self._model is not None:
            return self._model
        try:
            from torchvision.models.detection import keypointrcnn_resnet50_fpn
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('torchvision keypoint backend is unavailable') from exc
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _heuristic_keypoints(self, boxes: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for bbox in boxes:
            x1, y1, x2, y2 = map(float, np.asarray(bbox, dtype=np.float32).tolist())
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            pts = np.array([
                [x1 + 0.50 * w, y1 + 0.10 * h, 0.3],  # nose
                [x1 + 0.42 * w, y1 + 0.09 * h, 0.2],
                [x1 + 0.58 * w, y1 + 0.09 * h, 0.2],
                [x1 + 0.35 * w, y1 + 0.12 * h, 0.2],
                [x1 + 0.65 * w, y1 + 0.12 * h, 0.2],
                [x1 + 0.35 * w, y1 + 0.26 * h, 0.5],
                [x1 + 0.65 * w, y1 + 0.26 * h, 0.5],
                [x1 + 0.28 * w, y1 + 0.42 * h, 0.4],
                [x1 + 0.72 * w, y1 + 0.42 * h, 0.4],
                [x1 + 0.22 * w, y1 + 0.58 * h, 0.3],
                [x1 + 0.78 * w, y1 + 0.58 * h, 0.3],
                [x1 + 0.42 * w, y1 + 0.55 * h, 0.5],
                [x1 + 0.58 * w, y1 + 0.55 * h, 0.5],
                [x1 + 0.38 * w, y1 + 0.72 * h, 0.5],
                [x1 + 0.62 * w, y1 + 0.72 * h, 0.5],
                [x1 + 0.36 * w, y1 + 0.92 * h, 0.4],
                [x1 + 0.64 * w, y1 + 0.92 * h, 0.4],
            ], dtype=np.float32)
            out.append(pts)
        self._backend_used = 'heuristic'
        return out

    def _predict_real(self, frame: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        model = self._load_real_model()
        tensor = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = model([tensor])[0]
        pred_boxes = output['boxes'].detach().cpu().numpy().astype(np.float32)
        pred_scores = output['scores'].detach().cpu().numpy().astype(np.float32)
        pred_kpts = output['keypoints'].detach().cpu().numpy().astype(np.float32)
        assigned: list[np.ndarray] = []
        for box in boxes:
            box = np.asarray(box, dtype=np.float32)
            best_idx = -1
            best_iou = -1.0
            for idx, pred_box in enumerate(pred_boxes):
                if float(pred_scores[idx]) < self.score_threshold:
                    continue
                iou = _bbox_iou(box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou > 0.05:
                assigned.append(np.asarray(pred_kpts[best_idx], dtype=np.float32))
            else:
                assigned.extend(self._heuristic_keypoints([box]))
        self._backend_used = 'real'
        return assigned


    def release_resources(self) -> None:
        self._model = None

    def predict(self, frame: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        if self.backend_mode == 'heuristic':
            return self._heuristic_keypoints(boxes)
        if self.backend_mode == 'real':
            return self._predict_real(frame, boxes)
        try:
            return self._predict_real(frame, boxes)
        except Exception:
            return self._heuristic_keypoints(boxes)

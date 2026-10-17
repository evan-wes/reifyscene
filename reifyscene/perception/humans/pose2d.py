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
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('pose_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.4))
        self.max_inference_side = self.cfg.get('pose_max_inference_side', self.cfg.get('max_inference_side'))
        self.oom_retry_on_cpu = bool(self.cfg.get('oom_retry_on_cpu', True))
        self.debug = bool(self.cfg.get('debug', False))
        self._model = None
        self._model_device = None
        self._backend_used = 'uninitialized'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    @staticmethod
    def _resize(frame: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float, float]:
        if max_side is None:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        scale = min(float(max_side) / float(max(h, w)), 1.0)
        if scale >= 1.0:
            return frame, 1.0, 1.0
        import cv2
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), float(w) / nw, float(h) / nh

    def _load_real_model(self, device: str | None = None):
        target = str(device or self.device)
        if self._model is not None and self._model_device == target:
            return self._model
        try:
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        except Exception as exc:
            raise RuntimeError('torchvision keypoint backend is unavailable') from exc
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights)
        model.to(target)
        model.eval()
        self._model = model
        self._model_device = target
        return model

    def _run_real(self, frame: np.ndarray, boxes: list[np.ndarray], device: str, max_side: int | None) -> list[np.ndarray]:
        import torch
        model = self._load_real_model(device)
        image, sx, sy = self._resize(np.asarray(frame), max_side)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            output = model([tensor])[0]
        pred_boxes = output['boxes'].detach().cpu().numpy().astype(np.float32)
        pred_scores = output['scores'].detach().cpu().numpy().astype(np.float32)
        pred_kpts = output['keypoints'].detach().cpu().numpy().astype(np.float32)
        pred_boxes[:, [0, 2]] *= sx
        pred_boxes[:, [1, 3]] *= sy
        pred_kpts[..., 0] *= sx
        pred_kpts[..., 1] *= sy
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
                assigned.append(None)  # honest: no real detection matched this box (never fabricate)
        self._backend_used = 'keypoint_rcnn'
        return assigned

    def _predict_real(self, frame: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        try:
            return self._run_real(frame, boxes, self.device, self.max_inference_side)
        except Exception as exc:
            msg = str(exc).lower()
            if 'out of memory' not in msg or not self.oom_retry_on_cpu or not str(self.device).startswith('cuda'):
                raise
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if self.debug:
                print('[HUMAN DEBUG] CUDA OOM in pose, retrying on CPU.')
            return self._run_real(frame, boxes, 'cpu', min(int(self.max_inference_side or max(frame.shape[:2])), 640))

    def release_resources(self) -> None:
        self._model = None
        self._model_device = None

    def predict(self, frame: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray | None]:
        if self.backend_mode == 'real':
            return self._predict_real(frame, boxes)
        try:
            return self._predict_real(frame, boxes)
        except Exception:
            self._backend_used = 'unavailable'
            return [None] * len(boxes)  # honest: no keypoints (never fabricate a skeleton)

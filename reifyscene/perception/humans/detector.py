"""Human detection and masking module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PERSON_LABEL_ID = 1

@dataclass
class HumanDetectionResult:
    bbox_xyxy: np.ndarray
    score: float
    mask_logits: np.ndarray | None = None


class HumanDetector:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.5))
        self.max_detections = int(self.cfg.get('max_detections', 8))
        self.max_inference_side = self.cfg.get('max_inference_side')
        self.oom_retry_on_cpu = bool(self.cfg.get('oom_retry_on_cpu', True))
        self.debug = bool(self.cfg.get('debug', False))
        self._backend_used = 'uninitialized'
        self._model = None
        self._model_device = None

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
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
        except Exception as exc:
            raise RuntimeError('torchvision detection backend is unavailable') from exc
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        model.to(target)
        model.eval()
        self._model = model
        self._model_device = target
        return model

    def _run_real(self, frame: np.ndarray, *, device: str, max_side: int | None) -> list[HumanDetectionResult]:
        import torch
        model = self._load_real_model(device)
        image, sx, sy = self._resize(np.asarray(frame), max_side)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            output = model([tensor])[0]
        boxes = output['boxes'].detach().cpu().numpy().astype(np.float32)
        scores = output['scores'].detach().cpu().numpy().astype(np.float32)
        labels = output['labels'].detach().cpu().numpy().astype(np.int32)
        masks = output.get('masks')
        mask_arr = masks.detach().cpu().numpy() if masks is not None else None
        results: list[HumanDetectionResult] = []
        for idx in range(len(scores)):
            if int(labels[idx]) != PERSON_LABEL_ID:
                continue
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            box = np.asarray(boxes[idx], dtype=np.float32)
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            mask_logits = None
            if mask_arr is not None:
                mask_logits = np.asarray(mask_arr[idx, 0], dtype=np.float32)
                if sx != 1.0 or sy != 1.0:
                    import cv2
                    mask_logits = cv2.resize(mask_logits, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            results.append(HumanDetectionResult(bbox_xyxy=box, score=score, mask_logits=mask_logits))
            if len(results) >= self.max_detections:
                break
        self._backend_used = 'real'
        return results

    def _detect_real(self, frame: np.ndarray) -> list[HumanDetectionResult]:
        try:
            return self._run_real(frame, device=self.device, max_side=self.max_inference_side)
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
                print('[HUMAN DEBUG] CUDA OOM in detector, retrying on CPU.')
            return self._run_real(frame, device='cpu', max_side=min(int(self.max_inference_side or max(frame.shape[:2])), 768))

    def _detect_heuristic(self, frame: np.ndarray) -> list[HumanDetectionResult]:
        self._backend_used = 'heuristic'
        return []

    def release_resources(self) -> None:
        self._model = None
        self._model_device = None

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.backend_mode == 'heuristic':
            detections = self._detect_heuristic(frame)
        elif self.backend_mode == 'real':
            detections = self._detect_real(frame)
        else:
            try:
                detections = self._detect_real(frame)
            except Exception:
                detections = self._detect_heuristic(frame)
        return [{
            'bbox_xyxy': det.bbox_xyxy,
            'score': det.score,
            'mask_logits': det.mask_logits,
            'category': 'person',
            'label_id': PERSON_LABEL_ID,
        } for det in detections]

"""Human detection and masking module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PERSON_LABEL_ID = 1


@dataclass
class HumanDetectionResult:
    """Normalized human detector output for one instance."""

    bbox_xyxy: np.ndarray
    score: float
    mask_logits: np.ndarray | None = None


class HumanDetector:
    """Detect humans in image frames.

    The real backend uses torchvision Mask R-CNN and filters to the COCO
    ``person`` class. ``auto`` falls back to a no-op heuristic backend when
    torchvision or pretrained weights are not available.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.5))
        self.max_detections = int(self.cfg.get('max_detections', 8))
        self._backend_used = 'uninitialized'
        self._model = None

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def _load_real_model(self):
        if self._model is not None:
            return self._model
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('torchvision detection backend is unavailable') from exc
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _detect_real(self, frame: np.ndarray) -> list[HumanDetectionResult]:
        import torch

        model = self._load_real_model()
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'Expected HxWx3 frame, got shape {image.shape}.')
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = model([tensor])[0]
        boxes = output['boxes'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        masks = output.get('masks')
        mask_arr = masks.detach().cpu().numpy() if masks is not None else None
        results: list[HumanDetectionResult] = []
        for idx in range(len(scores)):
            if int(labels[idx]) != PERSON_LABEL_ID:
                continue
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            results.append(
                HumanDetectionResult(
                    bbox_xyxy=np.asarray(boxes[idx], dtype=np.float32),
                    score=score,
                    mask_logits=None if mask_arr is None else np.asarray(mask_arr[idx, 0], dtype=np.float32),
                )
            )
            if len(results) >= self.max_detections:
                break
        self._backend_used = 'real'
        return results

    def _detect_heuristic(self, frame: np.ndarray) -> list[HumanDetectionResult]:
        self._backend_used = 'heuristic'
        return []


    def release_resources(self) -> None:
        self._model = None

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
        return [
            {
                'bbox_xyxy': det.bbox_xyxy,
                'score': det.score,
                'mask_logits': det.mask_logits,
                'category': 'person',
                'label_id': PERSON_LABEL_ID,
            }
            for det in detections
        ]

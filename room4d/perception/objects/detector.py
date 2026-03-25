
"""Generic object detector wrapper with optional real Mask R-CNN backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COCO_INSTANCE_CATEGORY_NAMES: list[str] = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class DetectionResult:
    """Normalized object detector output for a single instance."""

    bbox_xyxy: np.ndarray
    score: float
    label_id: int
    category: str
    mask_logits: np.ndarray | None = None


class ObjectDetector:
    """Detect objects in image frames.

    The real backend uses torchvision Mask R-CNN. ``auto`` falls back to a
    no-op heuristic backend when torchvision or pretrained weights are not
    available.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.5))
        self.max_detections = int(self.cfg.get('max_detections', 12))
        self.debug = bool(self.cfg.get('debug', False))
        self._model = None
        self._backend_used = 'uninitialized'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def _load_real_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        except Exception as exc:  # pragma: no cover - import availability varies
            raise RuntimeError('torchvision detection backend is unavailable') from exc

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _detect_real(self, frame: np.ndarray) -> list[DetectionResult]:
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
        results: list[DetectionResult] = []
        for idx in range(min(len(scores), self.max_detections)):
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            label_id = int(labels[idx])
            category = COCO_INSTANCE_CATEGORY_NAMES[label_id] if 0 <= label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else f'class_{label_id}'
            if category == 'N/A':
                category = f'class_{label_id}'
            results.append(
                DetectionResult(
                    bbox_xyxy=np.asarray(boxes[idx], dtype=np.float32),
                    score=score,
                    label_id=label_id,
                    category=category,
                    mask_logits=None if mask_arr is None else np.asarray(mask_arr[idx, 0], dtype=np.float32),
                )
            )
        self._backend_used = 'real'
        return results

    def _detect_heuristic(self, frame: np.ndarray) -> list[DetectionResult]:
        self._backend_used = 'heuristic'
        return []

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return normalized detections for a frame.

        Each detection dict contains ``bbox_xyxy``, ``score``, ``label_id``,
        ``category`` and optionally ``mask_logits``.
        """
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
                'label_id': det.label_id,
                'category': det.category,
                'mask_logits': det.mask_logits,
            }
            for det in detections
        ]

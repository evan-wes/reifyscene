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
    bbox_xyxy: np.ndarray
    score: float
    label_id: int
    category: str
    mask_logits: np.ndarray | None = None


class ObjectDetector:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.score_threshold = float(self.cfg.get('score_threshold', 0.5))
        self.max_detections = int(self.cfg.get('max_detections', 12))
        self.max_inference_side = self.cfg.get('max_inference_side')
        self.oom_retry_on_cpu = bool(self.cfg.get('oom_retry_on_cpu', True))
        self.debug = bool(self.cfg.get('debug', False))
        self._model = None
        self._model_device = None
        self._backend_used = 'uninitialized'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    @staticmethod
    def _resize_for_inference(frame: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float, float]:
        if max_side is None:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        scale = min(float(max_side) / float(max(h, w)), 1.0)
        if scale >= 1.0:
            return frame, 1.0, 1.0
        import cv2

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, float(w) / float(new_w), float(h) / float(new_h)

    def _load_real_model(self, device: str | None = None):
        target_device = str(device or self.device)
        if self._model is not None and self._model_device == target_device:
            return self._model
        try:
            import torch
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
        except Exception as exc:
            raise RuntimeError('torchvision detection backend is unavailable') from exc
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        model.to(target_device)
        model.eval()
        self._model = model
        self._model_device = target_device
        return model

    def _run_model(self, frame: np.ndarray, *, device: str, max_side: int | None) -> list[DetectionResult]:
        import torch

        image, sx, sy = self._resize_for_inference(np.asarray(frame), max_side)
        model = self._load_real_model(device)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
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
            box = np.asarray(boxes[idx], dtype=np.float32)
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            mask_logits = None
            if mask_arr is not None:
                mask_logits = np.asarray(mask_arr[idx, 0], dtype=np.float32)
                if sx != 1.0 or sy != 1.0:
                    import cv2
                    mask_logits = cv2.resize(mask_logits, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            results.append(DetectionResult(bbox_xyxy=box, score=score, label_id=label_id, category=category, mask_logits=mask_logits))
        self._backend_used = 'real'
        return results

    def _detect_real(self, frame: np.ndarray) -> list[DetectionResult]:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f'Expected HxWx3 frame, got shape {frame.shape}.')
        try:
            return self._run_model(frame, device=self.device, max_side=self.max_inference_side)
        except Exception as exc:
            msg = str(exc).lower()
            is_oom = 'out of memory' in msg
            if not is_oom or not self.oom_retry_on_cpu or not str(self.device).startswith('cuda'):
                raise
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if self.debug:
                print('[OBJECT DEBUG] CUDA OOM in object detector, retrying on CPU.')
            return self._run_model(frame, device='cpu', max_side=min(int(self.max_inference_side or max(frame.shape[:2])), 768))

    def _detect_heuristic(self, frame: np.ndarray) -> list[DetectionResult]:
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
            'label_id': det.label_id,
            'category': det.category,
            'mask_logits': det.mask_logits,
        } for det in detections]

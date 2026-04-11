"""Instance embedding utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


class InstanceEmbedder:
    """Compute appearance embeddings for object instances with safe fallbacks."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('embedder_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.hist_bins = int(self.cfg.get('hist_bins', 8))
        self.max_inference_side = self.cfg.get('embedder_max_inference_side', self.cfg.get('max_inference_side', 384))
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
        target_device = str(device or self.device)
        if self._model is not None and self._model_device == target_device:
            return self._model
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError('torchvision embedding backend is unavailable') from exc
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)
        model = torch.nn.Sequential(*(list(base.children())[:-1]))
        model.to(target_device)
        model.eval()
        self._model = model
        self._model_device = target_device
        return model

    def _embed_histogram(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        image = np.asarray(frame, dtype=np.uint8)
        feats: list[np.ndarray] = []
        for det in detections:
            mask = np.asarray(det.get('mask'), dtype=bool)
            if mask.shape[:2] != image.shape[:2]:
                mask = np.zeros(image.shape[:2], dtype=bool)
            pixels = image[mask]
            if pixels.size == 0:
                bbox = np.asarray(det['bbox_xyxy'], dtype=np.int32)
                x1, y1, x2, y2 = bbox.tolist()
                pixels = image[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].reshape(-1, 3)
            if pixels.size == 0:
                feats.append(np.zeros((self.hist_bins * 3,), dtype=np.float32))
                continue
            channels = []
            for c in range(3):
                hist, _ = np.histogram(pixels[:, c], bins=self.hist_bins, range=(0, 255), density=True)
                channels.append(hist.astype(np.float32))
            feat = np.concatenate(channels, axis=0)
            norm = float(np.linalg.norm(feat))
            if norm > 1e-8:
                feat = feat / norm
            feats.append(feat.astype(np.float32))
        self._backend_used = 'histogram'
        return np.stack(feats, axis=0) if feats else np.zeros((0, self.hist_bins * 3), dtype=np.float32)

    def _run_real(self, frame: np.ndarray, detections: list[dict[str, Any]], device: str, max_side: int | None) -> np.ndarray:
        import torch
        from PIL import Image
        from torchvision.models import ResNet18_Weights

        model = self._load_real_model(device)
        preprocess = ResNet18_Weights.DEFAULT.transforms()
        image = np.asarray(frame, dtype=np.uint8)
        resized, sx, sy = self._resize(image, max_side)
        crops = []
        for det in detections:
            bbox = np.asarray(det['bbox_xyxy'], dtype=np.float32).copy()
            bbox[[0, 2]] /= sx
            bbox[[1, 3]] /= sy
            x1, y1, x2, y2 = np.asarray(np.round(bbox), dtype=np.int32).tolist()
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(resized.shape[1], max(x1 + 1, x2)); y2 = min(resized.shape[0], max(y1 + 1, y2))
            crop = resized[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
            crops.append(preprocess(Image.fromarray(crop)).to(device))
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = torch.stack(crops, dim=0)
        with torch.no_grad():
            feats = model(batch).flatten(1)
        feats = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._backend_used = 'real'
        return feats.detach().cpu().numpy().astype(np.float32)

    def _embed_real(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        try:
            return self._run_real(frame, detections, self.device, self.max_inference_side)
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
                print('[OBJECT DEBUG] CUDA OOM in embedder, retrying on CPU.')
            return self._run_real(frame, detections, 'cpu', min(int(self.max_inference_side or max(frame.shape[:2])), 384))

    def release_resources(self) -> None:
        self._model = None
        self._model_device = None

    def embed(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        if self.backend_mode == 'histogram':
            return self._embed_histogram(frame, detections)
        if self.backend_mode == 'real':
            return self._embed_real(frame, detections)
        try:
            return self._embed_real(frame, detections)
        except Exception:
            return self._embed_histogram(frame, detections)

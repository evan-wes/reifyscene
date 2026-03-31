
"""Instance embedding utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


class InstanceEmbedder:
    """Compute appearance embeddings for object instances.

    ``auto`` tries a torchvision ResNet feature extractor and falls back to a
    lightweight masked color histogram embedding if unavailable.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cpu'))
        self.hist_bins = int(self.cfg.get('hist_bins', 8))
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
            from torchvision.models import resnet18, ResNet18_Weights
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('torchvision embedding backend is unavailable') from exc
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)
        model = torch.nn.Sequential(*(list(base.children())[:-1]))
        model.to(self.device)
        model.eval()
        self._model = model
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

    def _embed_real(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        import torch
        from PIL import Image
        from torchvision.models import ResNet18_Weights

        model = self._load_real_model()
        preprocess = ResNet18_Weights.DEFAULT.transforms()
        image = np.asarray(frame, dtype=np.uint8)
        crops = []
        for det in detections:
            bbox = np.asarray(det['bbox_xyxy'], dtype=np.int32)
            x1, y1, x2, y2 = bbox.tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], max(x1 + 1, x2))
            y2 = min(image.shape[0], max(y1 + 1, y2))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
            mask = np.asarray(det.get('mask'), dtype=bool)
            if mask.shape[:2] == image.shape[:2]:
                mask_crop = mask[y1:y2, x1:x2]
                crop = crop.copy()
                crop[~mask_crop] = 0
            crops.append(preprocess(Image.fromarray(crop)).to(self.device))
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = torch.stack(crops, dim=0)
        with torch.no_grad():
            feats = model(batch).flatten(1)
        feats = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._backend_used = 'real'
        return feats.detach().cpu().numpy().astype(np.float32)


    def release_resources(self) -> None:
        self._model = None

    def embed(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        if self.backend_mode == 'histogram':
            return self._embed_histogram(frame, detections)
        if self.backend_mode == 'real':
            return self._embed_real(frame, detections)
        try:
            return self._embed_real(frame, detections)
        except Exception:
            return self._embed_histogram(frame, detections)

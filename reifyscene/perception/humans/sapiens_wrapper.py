"""Sapiens human-dense wrapper (swappable seam).

Wraps Meta's Sapiens-lite TorchScript models — 2D pose, body-part segmentation,
depth, and surface normals — behind one swappable estimator, run **top-down on
person crops** (the way Sapiens is trained). Ungated HF checkpoints
(``facebook/sapiens-{task}-{size}-torchscript``), **CC-BY-NC-4.0** (non-commercial
— fine for this no-ship research repo).

Lazy + import-safe: torch + the checkpoints load only on first :meth:`predict`
(and free on :meth:`release_resources`), so constructing the wrapper costs nothing
when Sapiens isn't installed — the humans ladder probes for it and degrades to the
Keypoint-RCNN floor otherwise.

Verified 0.6B I/O (input ``(1,3,1024,768)`` ImageNet-normalized): pose
``(1,308,256,192)`` Goliath heatmaps, seg ``(1,28,512,384)`` part logits, depth
``(1,1,1024,768)``, normal ``(1,3,512,384)``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Sapiens preprocessing: ImageNet mean/std on the 0-1 range, fixed 1024x768 input.
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_H, _INPUT_W = 1024, 768
DEFAULT_TASKS = ('pose', 'seg', 'depth', 'normal')

# (repo, filename) per task, keyed by model size. Only the chosen 0.6B set is
# pinned today; other sizes drop in by adding their filenames here.
_CHECKPOINTS: dict[str, dict[str, tuple[str, str]]] = {
    '0.6b': {
        'pose':   ('facebook/sapiens-pose-0.6b-torchscript',   'sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2'),
        'seg':    ('facebook/sapiens-seg-0.6b-torchscript',    'sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2'),
        'depth':  ('facebook/sapiens-depth-0.6b-torchscript',  'sapiens_0.6b_render_people_epoch_70_torchscript.pt2'),
        'normal': ('facebook/sapiens-normal-0.6b-torchscript', 'sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2'),
    },
}


def sapiens_cached(size: str = '0.6b') -> Optional[str]:
    """Ladder probe: ``None`` if torch + the Sapiens ``size`` checkpoints are present, else a reason.

    Mirrors how SAM2's checkpoint presence gates that rung — after
    ``setup_reifyscene.sh --with-sapiens`` caches the weights, the rung activates.
    """
    import importlib.util

    if importlib.util.find_spec('torch') is None:
        return 'not-installed: torch'
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return 'not-installed: huggingface_hub'
    ckpts = _CHECKPOINTS.get(size)
    if not ckpts:
        return f'unknown sapiens size: {size}'
    for repo, filename in ckpts.values():
        if not isinstance(try_to_load_from_cache(repo, filename), str):
            return f'sapiens-{size} not cached (setup_reifyscene.sh --with-sapiens)'
    return None


class SapiensEstimator:
    """Top-rung human-dense estimator: Sapiens pose + seg + depth + normals."""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = dict(cfg or {})
        self.size: str = str(cfg.get('sapiens_size', '0.6b'))
        self.tasks: tuple[str, ...] = tuple(cfg.get('sapiens_tasks', DEFAULT_TASKS))
        self.device: Optional[str] = cfg.get('device')
        self._models: dict[str, Any] = {}
        self._device_resolved: Optional[str] = None
        self._backend_used = 'uninitialized'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    # --- model (lazy; import-safe when torch / checkpoints absent) -----------
    def _load_models(self) -> None:
        if self._models:
            return
        import torch
        from huggingface_hub import hf_hub_download

        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpts = _CHECKPOINTS[self.size]
        for task in self.tasks:
            repo, filename = ckpts[task]
            path = hf_hub_download(repo, filename)
            self._models[task] = torch.jit.load(path, map_location=device).eval()
        self._device_resolved = device

    def release_resources(self) -> None:
        self._models = {}
        self._device_resolved = None

    # --- preprocessing ------------------------------------------------------
    def _crop_box(self, frame: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(float(c))) for c in np.asarray(box).ravel()[:4])
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        return np.ascontiguousarray(frame[y1:y2, x1:x2]), (x1, y1, x2, y2)

    def _preprocess(self, crop: np.ndarray):
        """Aspect-pad to the model's 3:4 (W:H), resize, ImageNet-normalize.

        Sapiens dense heads (seg especially) collapse to background if the person
        crop is stretched, so center-pad to 768:1024 before the resize and undo the
        pad when decoding outputs back to crop space. Returns ``(tensor, meta)``.
        """
        import cv2
        import torch

        ch, cw = crop.shape[:2]
        target = _INPUT_W / _INPUT_H  # 0.75
        if cw / ch > target:
            ph, pw = int(round(cw / target)), cw
        else:
            ph, pw = ch, int(round(ch * target))
        pad_top, pad_left = (ph - ch) // 2, (pw - cw) // 2
        padded = cv2.copyMakeBorder(crop, pad_top, ph - ch - pad_top, pad_left, pw - cw - pad_left, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(padded, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device_resolved)
        return tensor, {'pad_top': pad_top, 'pad_left': pad_left, 'ph': ph, 'pw': pw, 'ch': ch, 'cw': cw}

    # --- per-task decoding (heatmap/logits/maps -> crop-then-image space) ----
    @staticmethod
    def _decode_pose(heatmaps: np.ndarray, meta: dict, origin: tuple[int, int]) -> np.ndarray:
        """(K, hh, hw) heatmaps -> (K, 3) [x, y, conf] in original-image coords."""
        k, hh, hw = heatmaps.shape
        x1, y1 = origin
        flat = heatmaps.reshape(k, -1)
        idx = flat.argmax(axis=1)
        conf = flat.max(axis=1)
        ys, xs = np.divmod(idx, hw)
        kpts = np.zeros((k, 3), dtype=np.float32)
        # heatmap -> padded-crop -> crop -> image
        kpts[:, 0] = xs * (meta['pw'] / hw) - meta['pad_left'] + x1
        kpts[:, 1] = ys * (meta['ph'] / hh) - meta['pad_top'] + y1
        kpts[:, 2] = conf
        return kpts

    @staticmethod
    def _resize_and_unpad(arr: np.ndarray, meta: dict, *, nearest: bool = False):
        """Resize a model output to padded-crop size, then slice off the padding."""
        import cv2
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        resized = cv2.resize(np.asarray(arr, dtype=np.float32), (meta['pw'], meta['ph']), interpolation=interp)
        return resized[meta['pad_top']:meta['pad_top'] + meta['ch'], meta['pad_left']:meta['pad_left'] + meta['cw']]

    def _decode_seg(self, logits: np.ndarray, meta: dict) -> np.ndarray:
        parts = logits.argmax(axis=0).astype(np.float32)  # (sh, sw) class ids
        return self._resize_and_unpad(parts, meta, nearest=True).astype(np.uint8)

    def _decode_normal(self, normals: np.ndarray, meta: dict) -> np.ndarray:
        n = self._resize_and_unpad(np.transpose(normals, (1, 2, 0)), meta)  # (ch, cw, 3)
        return (n / np.clip(np.linalg.norm(n, axis=2, keepdims=True), 1e-6, None)).astype(np.float32)

    # --- public -------------------------------------------------------------
    def predict(self, frame: np.ndarray, boxes: list) -> list[Optional[dict[str, Any]]]:
        """Per person box -> ``{keypoints_2d, seg, depth, normals, backend}`` (or None)."""
        if not boxes:
            return []
        self._load_models()
        import torch

        results: list[Optional[dict[str, Any]]] = []
        for box in boxes:
            crop, (x1, y1, x2, y2) = self._crop_box(frame, box)
            tensor, meta = self._preprocess(crop)
            out: dict[str, Any] = {'backend': f'sapiens_{self.size}'}
            with torch.no_grad():
                if 'pose' in self._models:
                    hm = self._models['pose'](tensor)[0].float().cpu().numpy()
                    out['keypoints_2d'] = self._decode_pose(hm, meta, (x1, y1))
                if 'seg' in self._models:
                    seg = self._models['seg'](tensor)[0].float().cpu().numpy()
                    out['seg'] = self._decode_seg(seg, meta)
                if 'depth' in self._models:
                    depth = self._models['depth'](tensor)[0, 0].float().cpu().numpy()
                    out['depth'] = self._resize_and_unpad(depth, meta)
                if 'normal' in self._models:
                    nrm = self._models['normal'](tensor)[0].float().cpu().numpy()
                    out['normals'] = self._decode_normal(nrm, meta)
            results.append(out)
        self._backend_used = f'sapiens_{self.size}'
        return results

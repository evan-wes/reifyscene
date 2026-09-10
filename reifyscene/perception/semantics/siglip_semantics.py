"""Semantics rungs: SigLIP2 open-vocab category (top) + COCO-80 concept (floor).

The ``semantics`` ladder (design §4A, FP4) names each canonical tracklet:

- :class:`SiglipSemantics` (top, FP4a) embeds the tracklet's masked crops with a
  frozen SigLIP2 ViT, mean-pools, and zero-shot matches against a candidate text
  vocabulary -> an open-vocab category + score + a SigLIP image embedding (the
  graph's semantic handle). Qwen2.5-VL free-form attributes/caption arrive at FP4b.
- :class:`FloorSemantics` (floor) carries the detector's COCO-80 ``concept``
  string already on the instances, degrading to ``"unknown"`` -- the schema-valid
  graceful rung that needs no model.

Both key their output by **canonical** (reid-merged) track id, so one entity
gets one label. The zero-shot decision is the pure, model-free
:func:`score_zero_shot`, so the matching logic is unit-testable without weights.
``model_id`` is a config knob (default the ungated ``google/siglip2-*``).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.perception.objects.detector import COCO_INSTANCE_CATEGORY_NAMES
from reifyscene.perception.semantics.semantics_artifact import UNKNOWN_CATEGORY, SemanticRecord

DEFAULT_MODEL_ID = 'google/siglip2-so400m-patch14-384'  # ungated; fixed-resolution variant
DEFAULT_PROMPT = 'This is a photo of a {}.'  # SigLIP zero-shot template
DEFAULT_TOPK = 5
# Default candidate vocabulary = COCO-80 (drops the torchvision placeholders), so
# the SigLIP rung is comparable to the detector floor; override via cfg['vocab'].
DEFAULT_VOCAB = [c for c in COCO_INSTANCE_CATEGORY_NAMES if c not in ('__background__', 'N/A')]


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def _pooled(out):
    """Pooled embedding tensor from a transformers feature call.

    ``get_image_features``/``get_text_features`` return a plain tensor on some
    transformers versions and a ``BaseModelOutputWithPooling`` on others (5.x for
    SigLIP2); handle both, falling back to a mean over the token dim.
    """
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, 'last_hidden_state'):
        return out.last_hidden_state.mean(dim=1)
    return out


def score_zero_shot(
    image_emb: np.ndarray,
    text_embs: np.ndarray,
    labels: list[str],
    *,
    topk: int = DEFAULT_TOPK,
    min_score: float = float('-inf'),
) -> tuple[str, float, list[tuple[str, float]]]:
    """Zero-shot category for one tracklet (pure: no torch / no model).

    Args:
        image_emb: ``(D,)`` tracklet image embedding (L2-normalized recommended).
        text_embs: ``(M, D)`` per-label text embeddings (rows L2-normalized).
        labels: the ``M`` candidate label strings.
        topk: how many ranked ``(label, score)`` pairs to return.
        min_score: if the best cosine is below this, abstain to ``"unknown"``.

    Returns ``(category, best_score, ranked_topk)``. Score is cosine similarity.
    """
    if text_embs is None or len(labels) == 0 or np.asarray(text_embs).size == 0:
        return UNKNOWN_CATEGORY, 0.0, []
    img = _l2norm(image_emb)
    txt = np.asarray(text_embs, dtype=np.float32)
    sims = txt @ img  # (M,) cosine if rows are unit-norm
    order = np.argsort(-sims)[: max(1, int(topk))]
    ranked = [(str(labels[i]), float(sims[i])) for i in order]
    best_label, best_score = ranked[0]
    if best_score < min_score:
        return UNKNOWN_CATEGORY, best_score, ranked
    return best_label, best_score, ranked


def _group_by_canonical(instance_artifact: InstanceArtifact, canonical) -> dict[str, list]:
    groups: dict[str, list] = {}
    for o in instance_artifact.observations:
        groups.setdefault(canonical(o.track_id), []).append(o)
    return groups


class FloorSemantics:
    """Floor rung: carry the detector's COCO-80 ``concept`` per canonical track."""

    def __init__(self, reid_artifact=None) -> None:
        self.reid = reid_artifact

    def _canonical(self, track_id: str) -> str:
        return self.reid.canonical(track_id) if self.reid is not None else track_id

    def run(self, ingest_artifact, instance_artifact: InstanceArtifact):
        """Return ``({canonical: SemanticRecord}, {})`` -- no embeddings at the floor."""
        records: dict[str, SemanticRecord] = {}
        for canonical, observations in _group_by_canonical(instance_artifact, self._canonical).items():
            concepts = [str(o.concept) for o in observations if str(o.concept)]
            if concepts:
                category, count = Counter(concepts).most_common(1)[0]
                score = count / len(observations)
            else:
                category, score = UNKNOWN_CATEGORY, 0.0
            records[canonical] = SemanticRecord(
                category=category, score=float(score), category_topk=[(category, float(score))],
                attributes={'source': 'coco_concept'},
            )
        return records, {}


class SiglipSemantics:
    """Top rung: SigLIP2 open-vocab category + image embedding per canonical track."""

    def __init__(self, decoder, reid_artifact=None, cfg: Optional[dict] = None) -> None:
        cfg = dict(cfg or {})
        self.decoder = decoder
        self.reid = reid_artifact
        self.model_id: str = cfg.get('model_id', DEFAULT_MODEL_ID)
        self.vocab: list[str] = list(cfg.get('vocab', DEFAULT_VOCAB))
        self.prompt: str = cfg.get('prompt', DEFAULT_PROMPT)
        self.topk: int = int(cfg.get('topk', DEFAULT_TOPK))
        self.min_score: float = float(cfg.get('min_score', float('-inf')))
        self.device: Optional[str] = cfg.get('device')
        self.use_mask: bool = bool(cfg.get('use_mask', True))
        self._model = None
        self._processor = None
        self._device_resolved: Optional[str] = None
        self._text_embs: Optional[np.ndarray] = None  # (M, D), cached

    def _canonical(self, track_id: str) -> str:
        return self.reid.canonical(track_id) if self.reid is not None else track_id

    # --- model (lazy, import-safe when transformers absent) -----------------
    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if device == 'cuda' else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype).to(device).eval()
        self._device_resolved = device
        self._text_embs = self._encode_text([self.prompt.format(lbl) for lbl in self.vocab])

    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        import torch

        inputs = self._processor(
            text=prompts, padding='max_length', max_length=64, truncation=True, return_tensors='pt',
        ).to(self._device_resolved)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        arr = _pooled(feats).float().cpu().numpy()
        return np.stack([_l2norm(row) for row in arr], axis=0)

    def _crop(self, frame: np.ndarray, bbox_xyxy, mask) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (int(round(float(c))) for c in np.asarray(bbox_xyxy).ravel()[:4])
        x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
        x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
        crop = frame[y0:y1, x0:x1]
        if self.use_mask and mask is not None:
            m = np.asarray(mask)
            if m.shape[:2] == frame.shape[:2]:
                m = m[y0:y1, x0:x1]
            if m.shape[:2] == crop.shape[:2]:
                crop = crop * (m[..., None] > 0)
        return np.ascontiguousarray(crop.astype(np.uint8))

    def _embed_image_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        inputs = self._processor(images=crops, return_tensors='pt').to(self._device_resolved)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return _pooled(feats).float().cpu().numpy()

    def run(self, ingest_artifact, instance_artifact: InstanceArtifact):
        """Return ``({canonical: SemanticRecord}, {canonical: SigLIP embedding})``.

        Frames are decoded once; all instances in a frame are cropped + embedded
        together. Per-canonical-track image embeddings are mean-pooled + L2
        normalized, then zero-shot matched against the vocabulary.
        """
        self._load_model()
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}

        by_frame: dict[str, list] = {}
        for obs in instance_artifact.observations:
            by_frame.setdefault(InstanceArtifact._frame_key(obs.frame_ref), []).append(obs)

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for _frame_key, observations in by_frame.items():
            fr: FrameRef = observations[0].frame_ref
            stream = stream_lookup.get(fr.stream_id)
            if stream is None:
                continue
            frame = self.decoder.get_frame(stream, fr.frame_idx)
            crops = [self._crop(frame, o.bbox_xyxy, instance_artifact.mask_arrays.get(o.instance_id)) for o in observations]
            feats = self._embed_image_crops(crops)
            for o, vec in zip(observations, np.asarray(feats)):
                key = self._canonical(o.track_id)
                vec = _l2norm(vec)
                sums[key] = sums[key] + vec if key in sums else vec.copy()
                counts[key] = counts.get(key, 0) + 1

        records: dict[str, SemanticRecord] = {}
        embeddings: dict[str, np.ndarray] = {}
        for key, total in sums.items():
            emb = _l2norm(total / max(counts[key], 1))
            embeddings[key] = emb
            category, score, topk = score_zero_shot(
                emb, self._text_embs, self.vocab, topk=self.topk, min_score=self.min_score,
            )
            records[key] = SemanticRecord(
                category=category, score=float(score), category_topk=topk,
                attributes={'source': 'siglip2', 'model_id': self.model_id},
            )
        return records, embeddings

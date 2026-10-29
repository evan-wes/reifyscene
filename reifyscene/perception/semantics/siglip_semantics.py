"""Semantics rungs: SigLIP2 open-vocab category (top) + COCO-80 concept (floor).

The ``semantics`` ladder (design §4A, FP4) names each canonical tracklet:

- :class:`SiglipSemantics` (top, FP4a) embeds the tracklet's masked crops with a
  frozen SigLIP2 ViT, mean-pools, and zero-shot matches against a candidate text
  vocabulary -> an open-vocab category + score + a SigLIP image embedding (the
  graph's semantic handle). Qwen2.5-VL free-form attributes/caption arrive at FP4b.
  It also says whether a track is backdrop (floor, sky, a crowd) rather than a thing
  (``SemanticRecord.background``; :mod:`reifyscene.perception.semantics.vocab`).
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
from typing import NamedTuple, Optional

import numpy as np

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.perception.semantics import vocab
from reifyscene.perception.semantics.semantics_artifact import UNKNOWN_CATEGORY, SemanticRecord

DEFAULT_MODEL_ID = 'google/siglip2-so400m-patch14-384'  # ungated; fixed-resolution variant
DEFAULT_PROMPT = 'This is a photo of a {}.'  # SigLIP zero-shot template
DEFAULT_TOPK = 5
# Abstention is OFF by default. A raw top1-top2 cosine gap is not a quantity a
# constant can be set against: it shrinks as the candidate list grows, because more
# near-neighbours crowd the runner-up. A threshold tuned on COCO-80 would
# over-abstain on a 500-word open vocabulary and under-abstain on a 5-word scene
# vocabulary -- the same defect as handing an open-vocab model a closed list.
#
# When abstention IS enabled, it is enabled on the z-score (cfg['min_margin_z']):
# how many standard deviations above the typical candidate the winner sits, over the
# whole vocabulary. That statistic stays comparable as the vocabulary changes size,
# which is the property a default needs. Both statistics are recorded on every
# record either way, so the threshold can be chosen from data rather than guessed.
DEFAULT_MIN_MARGIN_Z: float | None = None
# The candidate labels (:mod:`reifyscene.perception.semantics.vocab`): COCO-80 and LVIS's
# frequent tier for things, plus background classes; a person among COCO-80 and the background
# classes. Override via cfg['vocab'], cfg['person_vocab'], cfg['background_vocab'] and
# cfg['extended_vocab'].
DEFAULT_VOCAB = list(vocab.DEFAULT_VOCAB)
DEFAULT_PERSON_VOCAB = list(vocab.PERSON_VOCAB)
DEFAULT_BACKGROUND_VOCAB = list(vocab.BACKGROUND_VOCAB)
DEFAULT_EXTENDED_VOCAB = list(vocab.LVIS_FREQUENT)
# Backdrop or thing (``SiglipSemantics._backdrop``): a track is backdrop when a background class
# beats every core thing (COCO-80) and no extended thing (LVIS's frequent tier) beats it by this
# much (cosine, this model and prompt). The extended tier must win clearly because its long tail
# scores close to any backdrop -- "paper towel" for a sidewalk, "tape" for a ceiling -- while a
# real thing COCO cannot name beats the water around it. On the ten DAVIS slice runs
# (2026-10-17), of the tracks a background class won against COCO: every backdrop piece's best
# extended thing led by +0.040 or less; the gold-fish clip's four fish ("fish") by +0.049 to
# +0.061, and a black bin bag ("plastic bag") by +0.050.
DEFAULT_BACKGROUND_MARGIN = 0.045


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


class ZeroShotResult(NamedTuple):
    """One zero-shot match, including a refusal and the evidence behind it."""

    category: str
    score: float
    topk: list[tuple[str, float]]
    margin: float | None  # top1 - top2 cosine; None when there is no runner-up
    margin_z: float | None  # (top1 - mean) / sd over the whole vocabulary
    abstain_reason: str | None  # None when the match stands

    @property
    def abstained(self) -> bool:
        """Whether the rung was asked and declined to name the track."""
        return self.abstain_reason is not None


def score_zero_shot(
    image_emb: np.ndarray,
    text_embs: np.ndarray,
    labels: list[str],
    *,
    topk: int = DEFAULT_TOPK,
    min_score: float = float('-inf'),
    min_margin_z: float | None = DEFAULT_MIN_MARGIN_Z,
) -> ZeroShotResult:
    """Zero-shot category for one tracklet (pure: no torch / no model).

    Args:
        image_emb: ``(D,)`` tracklet image embedding (L2-normalized recommended).
        text_embs: ``(M, D)`` per-label text embeddings (rows L2-normalized).
        labels: the ``M`` candidate label strings.
        topk: how many ranked ``(label, score)`` pairs to return.
        min_score: if the best cosine is below this, abstain to ``"unknown"``.
        min_margin_z: if the winner's z-score is below this, abstain to ``"unknown"``.
            ``None`` (the default) never abstains on separation.

    **Absolute score does not say whether a label is right.** When the vocabulary does
    not contain the right answer the model still returns a winner, and that winner can
    outscore the correct labels elsewhere in the run: on ``davis_lab-coat_full`` the
    junk label ``handbag`` scored 0.1170, above both correctly-named people. What
    separates them is *separation* -- the two people led their runner-up by 0.045 and
    0.033, every junk label by 0.0105 or less.

    **But a raw gap is not a quantity a threshold can be set against**, because it
    shrinks as the candidate list grows. ``margin_z`` is the same evidence made
    scale-free: how many standard deviations above the typical candidate the winner
    sits, over the whole vocabulary. That is what the gate uses, and it is off unless
    a caller asks for it. Both statistics are reported regardless, so a threshold can
    be chosen from measurements instead of guessed.

    A single-label vocabulary has nothing to be separated from, so both statistics are
    ``None`` and the gate does not fire.

    **Known limitation of the z-score.** It places the winner against the *spread* of
    the candidates, so a bulk of near-identical candidates collapses that spread and a
    negligible gap reads as many sigma. Real vocabularies fan out, which is why the
    statistic works on them -- but a tiny or highly redundant candidate list breaks it,
    and a threshold is not transferable between vocabularies of different shape. Pinned
    by ``test_margin_z_is_inflated_by_a_degenerate_bulk``.

    Returns a :class:`ZeroShotResult`. The rejected best label and its evidence are
    still reported when it abstains, so the call can be audited downstream.
    """
    if text_embs is None or len(labels) == 0 or np.asarray(text_embs).size == 0:
        return ZeroShotResult(UNKNOWN_CATEGORY, 0.0, [], None, None, 'no_labels')
    img = _l2norm(image_emb)
    txt = np.asarray(text_embs, dtype=np.float32)
    sims = txt @ img  # (M,) cosine if rows are unit-norm
    order = np.argsort(-sims)[: max(1, int(topk))]
    ranked = [(str(labels[i]), float(sims[i])) for i in order]
    best_label, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else None
    # Over the WHOLE vocabulary, not the top-k: the point is to place the winner
    # against the bulk of candidates, which is what stays stable as the list grows.
    # Needs a real distribution to sit in: with two candidates the z-score is exactly
    # 1.0 whatever the scores are, which is a constant masquerading as evidence.
    sd = float(np.std(sims))
    margin_z = (best_score - float(np.mean(sims))) / sd if sims.size > 2 and sd > 0 else None
    if best_score < min_score:
        return ZeroShotResult(UNKNOWN_CATEGORY, best_score, ranked, margin, margin_z, 'low_score')
    if min_margin_z is not None and margin_z is not None and margin_z < min_margin_z:
        return ZeroShotResult(
            UNKNOWN_CATEGORY, best_score, ranked, margin, margin_z, 'low_separation'
        )
    return ZeroShotResult(best_label, best_score, ranked, margin, margin_z, None)


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
                reason = None
            else:
                # SAM2 is class-agnostic and emits concept=''. No label was ever
                # offered, so this is a refusal with a reason, not a silent blank.
                category, score, reason = UNKNOWN_CATEGORY, 0.0, 'no_concept'
            records[canonical] = SemanticRecord(
                category=category, score=float(score), category_topk=[(category, float(score))],
                abstained=reason is not None, abstain_reason=reason,
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
        self.person_vocab: list[str] = list(cfg.get('person_vocab', DEFAULT_PERSON_VOCAB))
        self.background_vocab: set[str] = set(cfg.get('background_vocab',
                                                      DEFAULT_BACKGROUND_VOCAB))
        self.extended_vocab: set[str] = set(cfg.get('extended_vocab', DEFAULT_EXTENDED_VOCAB))
        self.background_margin: float = float(cfg.get('background_margin',
                                                      DEFAULT_BACKGROUND_MARGIN))
        # Every label either vocabulary holds, encoded once: rows of ``_text_embs``.
        self._labels: list[str] = list(dict.fromkeys(self.vocab + self.person_vocab))
        self.prompt: str = cfg.get('prompt', DEFAULT_PROMPT)
        self.topk: int = int(cfg.get('topk', DEFAULT_TOPK))
        self.min_score: float = float(cfg.get('min_score', float('-inf')))
        raw_z = cfg.get('min_margin_z', DEFAULT_MIN_MARGIN_Z)
        self.min_margin_z: float | None = None if raw_z is None else float(raw_z)
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
        self._text_embs = self._encode_text([self.prompt.format(lbl) for lbl in self._labels])

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

        A track SAM 3 found is named by the phrase that found it instead: SAM 3 matched
        that phrase against the pixels, where this stage would be choosing among 453
        labels from a crop. SigLIP's answer is kept beside it (``siglip_category``), so a
        disagreement between the two stays visible.
        """
        self._load_model()
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}

        by_frame: dict[str, list] = {}
        for obs in instance_artifact.observations:
            by_frame.setdefault(InstanceArtifact._frame_key(obs.frame_ref), []).append(obs)

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        humans: dict[str, int] = {}                     # observations the segmenter called a person
        phrases: dict[str, dict[str, list[float]]] = {}  # SAM 3's phrase -> its scores, per track
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
                humans[key] = humans.get(key, 0) + (o.kind == 'human')
                if (o.attributes or {}).get('concept_source') == 'sam3' and str(o.concept):
                    phrases.setdefault(key, {}).setdefault(str(o.concept), []).append(float(o.confidence))

        row = {lbl: i for i, lbl in enumerate(self._labels)}
        records: dict[str, SemanticRecord] = {}
        embeddings: dict[str, np.ndarray] = {}
        for key, total in sums.items():
            emb = _l2norm(total / max(counts[key], 1))
            embeddings[key] = emb
            person = 2 * humans[key] > counts[key]
            labels = self.person_vocab if person else self.vocab
            text = self._text_embs[[row[lbl] for lbl in labels]]
            result = score_zero_shot(
                emb, text, labels,
                topk=self.topk, min_score=self.min_score, min_margin_z=self.min_margin_z,
            )
            category, score = result.category, float(result.score)
            attributes = {'source': 'siglip2', 'model_id': self.model_id}
            background = False
            if not person and not result.abstained:
                background, best_bg, leads = self._backdrop(emb, text, labels)
                attributes.update(leads)
                if background:
                    category, score = best_bg
            if key in phrases:
                records[key] = self._sam3_record(phrases[key], category, score)
                continue
            records[key] = SemanticRecord(
                category=category,
                score=score,
                category_topk=result.topk,
                margin=result.margin,
                margin_z=result.margin_z,
                abstained=result.abstained,
                abstain_reason=result.abstain_reason,
                background=background,
                attributes=attributes,
            )
        return records, embeddings

    def _sam3_record(self, scores_by_phrase: dict[str, list[float]], siglip_category: str,
                     siglip_score: float) -> SemanticRecord:
        """The record of a track SAM 3 named: the phrase with the most total score over its frames
        (a re-identified track can join several SAM 3 tracks), scored by its mean.

        A backdrop phrase ("tree", "palm trees", "crowd of people") marks the track as background,
        the same rule the zero-shot labels follow.
        """
        from reifyscene.perception.instance.sam3_segment import head_noun

        ranked = sorted(scores_by_phrase.items(), key=lambda kv: (-sum(kv[1]), kv[0]))
        topk = [(phrase, float(np.mean(s))) for phrase, s in ranked[: max(1, self.topk)]]
        category, score = topk[0]
        noun = head_noun(category)
        backdrop = {category, noun, noun[:-1] if noun.endswith('s') else noun}
        return SemanticRecord(
            category=category,
            score=score,
            category_topk=topk,
            background=bool(backdrop & self.background_vocab),
            attributes={'source': 'sam3_phrase', 'siglip_category': siglip_category,
                        'siglip_score': float(siglip_score), 'model_id': self.model_id},
        )

    def _backdrop(self, emb: np.ndarray, text: np.ndarray, labels: list[str]
                  ) -> tuple[bool, tuple[str, float] | None, dict[str, float]]:
        """``(background, (best background label, its score), leads)``: whether the track is
        backdrop -- a background class beats every core thing, and no extended thing beats it
        by ``background_margin`` -- and how far the best core and extended things led the best
        background class (``core_lead``, ``extended_lead``). A person is never asked."""
        sims = text @ emb
        is_bg = np.array([lbl in self.background_vocab for lbl in labels])
        is_ext = np.array([lbl in self.extended_vocab for lbl in labels]) & ~is_bg
        is_core = ~is_bg & ~is_ext
        if not is_bg.any():
            return False, None, {}
        b = int(np.flatnonzero(is_bg)[np.argmax(sims[is_bg])])
        leads = {}
        if is_core.any():
            leads['core_lead'] = float(sims[is_core].max() - sims[b])
        if is_ext.any():
            leads['extended_lead'] = float(sims[is_ext].max() - sims[b])
        background = (leads.get('core_lead', -np.inf) < 0
                      and leads.get('extended_lead', -np.inf) < self.background_margin)
        return background, (labels[b], float(sims[b])), leads

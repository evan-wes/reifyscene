"""Top-rung re-ID: DINO appearance features + cross-occlusion gap merges.

The DINO rung of the `reid` ladder (design §4A, FP3). It sits above the
ResNet18 floor (:class:`reifyscene.perception.reid.reid_stage.FloorReid`) and adds
the thing the floor leaves empty: a **merge map** that re-acquires identity
*across occlusion gaps* — two tracklets the segmenter gave separate ids but that
are the same entity (it left, came back) get folded onto one canonical id.

Two pieces, kept separable:

- :class:`DinoReid` embeds each tracklet with a frozen DINO ViT (mean-pooled,
  L2-normalized) and records its temporal span, then calls
- :func:`build_cross_occlusion_merge_map`, a **pure** function (vectors + spans
  in, merge map out — no model, no torch) so the merge logic is unit-testable
  without weights.

Backend choice is a config knob, not a code change: ``model_id`` defaults to the
ungated **DINOv2-L** (``facebook/dinov2-large``, Apache) and swaps to the gated
**DINOv3-L** (``facebook/dinov3-vitl16-pretrain-lvd1689m``) by id alone once HF
access lands — both load through the same ``transformers.AutoModel`` call. The
model is imported + built lazily inside :meth:`DinoReid.run`, so constructing the
rung is import-safe when ``transformers`` is absent (the ladder probe gates it).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.perception.reid.track_bridges import bridge_counts, tracklet_point_sets

# Default (ungated, Apache) appearance backbone. DINOv3-L is a drop-in id swap.
DEFAULT_MODEL_ID = 'facebook/dinov2-large'
DINOV3L_MODEL_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'  # gated; swap in when granted
# Cosine-similarity floor for folding two disjoint tracklets onto one identity.
# High enough to stay precise (a false merge corrupts an entity node); tuned on
# the synthetic occlusion validation.
DEFAULT_MERGE_THRESHOLD = 0.6
# Ratio-test margin: a tracklet's best candidate must beat its second-best by at
# least this (cosine) margin to count as unambiguous (Lowe's ratio test, additive
# form). Off by default (0.0): on the windmill validation the mutual-NN rule alone
# gave zero false merges, and a non-zero margin only cost true merges. Kept as a
# tunable knob for scenes where near-tied multi-candidate ambiguity is a problem.
DEFAULT_RATIO_MARGIN = 0.0


# Point tracks spanning a gap admit a pair that appearance alone would refuse. Two is
# the smallest count that is not a single point clipping a mask in passing; the relax is
# deliberately small, because geometry here is corroboration, not proof.
DEFAULT_BRIDGE_MIN = 2
DEFAULT_BRIDGE_RELAX = 0.05


def build_cross_occlusion_merge_map(
    vectors: dict[str, np.ndarray],
    spans: dict[str, tuple[str, int, int]],
    *,
    threshold: float = DEFAULT_MERGE_THRESHOLD,
    ratio_margin: float = DEFAULT_RATIO_MARGIN,
    bridges: dict[tuple[str, str], int] | None = None,
    bridge_min: int = DEFAULT_BRIDGE_MIN,
    bridge_relax: float = DEFAULT_BRIDGE_RELAX,
) -> dict[str, str]:
    """Fold appearance-similar, time-disjoint tracklets onto one canonical id.

    Pure (no torch / no model): the unit-testable core of FP3's cross-occlusion
    re-acquisition.

    Args:
        vectors: ``track_id -> L2-normalized appearance vector``.
        spans: ``track_id -> (stream_id, frame_min, frame_max)`` (inclusive).
        threshold: minimum cosine similarity for a pair to be a merge candidate.
        ratio_margin: each side's best candidate must beat its second-best by at
            least this margin (abstain on near-ties).
        bridges: ``{(id_a, id_b): shared point tracks}`` from
            :mod:`~reifyscene.perception.reid.track_bridges`, sorted-pair keyed.
        bridge_min: how many shared point tracks make a pair bridged.
        bridge_relax: how far a bridged pair's appearance threshold is lowered.

    Returns:
        ``{track_id: canonical_id}`` for every *non-canonical* member of a merged
        pair (canonical ids are omitted — :meth:`ReidArtifact.canonical` treats a
        missing id as its own canonical). The canonical is the earlier-appearing
        member (smallest ``frame_min``, then ``stream_id``, then id).

    A pair merges only if it is a **mutual nearest neighbour** (each is the
    other's most-similar valid candidate) *and* passes the **ratio test** (the
    best candidate clearly beats the second-best on both sides). This is far more
    precise than greedy thresholding on near-identical instances: a tracklet that
    looks slightly more like a *different* entity than its own other half is
    correctly left unmerged rather than confidently mis-joined.

    **Geometry admits candidates; it never merges them.** A pair bridged by point
    tracks that sat inside both tracklets' masks has its appearance threshold lowered by
    ``bridge_relax`` -- so the same thing observed continuously through a gap can be
    considered even when it does not look quite like itself. It still has to win
    mutual-nearest-neighbour and the ratio test. A point track can drift off one entity
    onto another, and that failure must not be able to fuse two people on its own.

    Candidates are restricted to **same-stream**, **span-disjoint** pairs — two
    tracklets sharing any frame are co-visible (e.g. over-segmented parts), a
    different problem than occlusion re-acquisition. Mutual-best matching yields a
    *matching* (each tracklet folds onto at most one partner), so this merges
    pairs, not multi-gap chains; chaining across several gaps is a deliberate
    follow-up. Cross-stream identity matching is out of FP3 scope (design §4A:
    surfaces at fusion).
    """
    tracks = [t for t in vectors if t in spans]
    # Ranked valid candidates per tracklet: same stream, disjoint spans, sim >= threshold.
    cand: dict[str, list[tuple[float, str]]] = {t: [] for t in tracks}
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            ti, tj = tracks[a], tracks[b]
            si, mi, xi = spans[ti][0], int(spans[ti][1]), int(spans[ti][2])
            sj, mj, xj = spans[tj][0], int(spans[tj][1]), int(spans[tj][2])
            if si != sj:
                continue
            if not (xi < mj or xj < mi):  # spans touch/overlap -> co-visible, skip
                continue
            sim = float(np.dot(np.asarray(vectors[ti]).ravel(), np.asarray(vectors[tj]).ravel()))
            shared = 0 if not bridges else int(bridges.get(tuple(sorted((ti, tj))), 0))
            limit = threshold - bridge_relax if shared >= int(bridge_min) else threshold
            if sim >= limit:
                cand[ti].append((sim, tj))
                cand[tj].append((sim, ti))
    for t in tracks:
        cand[t].sort(reverse=True)

    def second_sim(t: str) -> float:
        return cand[t][1][0] if len(cand[t]) >= 2 else float('-inf')

    merge_map: dict[str, str] = {}
    for t in tracks:
        if not cand[t]:
            continue
        sim_t, u = cand[t][0]  # t's best candidate
        if not cand[u] or cand[u][0][1] != t:
            continue  # not a mutual nearest neighbour
        sim_u = cand[u][0][0]
        # Ratio test: best must clearly beat second-best on both sides.
        if sim_t - second_sim(t) < ratio_margin or sim_u - second_sim(u) < ratio_margin:
            continue
        canonical, other = sorted((t, u), key=lambda x: (int(spans[x][1]), spans[x][0], x))
        merge_map[other] = canonical
    return merge_map


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32) if norm > 1e-8 else v.astype(np.float32)


class DinoReid:
    """Per-tracklet DINO appearance vectors + a cross-occlusion merge map.

    Mirrors :class:`FloorReid`'s decode-once / batch-embed loop, but computes its
    own features with a frozen DINO ViT instead of the ResNet18 embedder, and
    fills the merge map the floor leaves empty.
    """

    def __init__(self, decoder, cfg: Optional[dict] = None) -> None:
        cfg = dict(cfg or {})
        self.decoder = decoder
        self.model_id: str = cfg.get('model_id', DEFAULT_MODEL_ID)
        self.merge_threshold: float = float(cfg.get('merge_threshold', DEFAULT_MERGE_THRESHOLD))
        self.ratio_margin: float = float(cfg.get('ratio_margin', DEFAULT_RATIO_MARGIN))
        self.device: Optional[str] = cfg.get('device')
        self.use_mask: bool = bool(cfg.get('use_mask', True))
        self._model = None
        self._processor = None
        self._device_resolved: Optional[str] = None

    # --- model (lazy, import-safe when transformers absent) -----------------
    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if device == 'cuda' else torch.float32
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype).to(device).eval()
        self._device_resolved = device

    def _crop(self, frame: np.ndarray, bbox_xyxy, mask) -> np.ndarray:
        """Crop the instance box (background masked to 0 when a mask is given)."""
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

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Run a frame's crops through DINO -> (N, D) pooled features (cpu numpy)."""
        import torch

        inputs = self._processor(images=crops, return_tensors='pt').to(self._device_resolved)
        with torch.no_grad():
            out = self._model(**inputs)
        feats = getattr(out, 'pooler_output', None)
        if feats is None:  # fall back to the CLS token of the last hidden state
            feats = out.last_hidden_state[:, 0]
        return feats.float().cpu().numpy()

    def run(
        self, ingest_artifact, instance_artifact: InstanceArtifact, track_artifact=None
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        """Return ``({track_id: mean DINO vector}, merge_map)``.

        Frames are decoded once; all instances in a frame are cropped + embedded
        together. Per-track vectors are mean-pooled then L2-normalized; the merge
        map is built from those vectors + each track's temporal span.

        ``track_artifact`` adds geometric corroboration: point tracks that sat inside
        both tracklets' masks bridge the gap between them, and a bridged pair's
        appearance threshold is relaxed (:mod:`reifyscene.perception.reid.track_bridges`).
        It cannot merge anything on its own, and without it the merge is exactly the
        appearance-only one.
        """
        self._load_model()
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}

        by_frame: dict[str, list] = {}
        for obs in instance_artifact.observations:
            by_frame.setdefault(InstanceArtifact._frame_key(obs.frame_ref), []).append(obs)

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        spans: dict[str, list] = {}  # track_id -> [stream_id, frame_min, frame_max]
        for _frame_key, observations in by_frame.items():
            fr: FrameRef = observations[0].frame_ref
            stream = stream_lookup.get(fr.stream_id)
            if stream is None:
                continue
            frame = self.decoder.get_frame(stream, fr.frame_idx)
            crops = [self._crop(frame, o.bbox_xyxy, instance_artifact.mask_arrays.get(o.instance_id)) for o in observations]
            feats = self._embed_crops(crops)
            for o, vec in zip(observations, np.asarray(feats)):
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                sums[o.track_id] = sums[o.track_id] + vec if o.track_id in sums else vec.copy()
                counts[o.track_id] = counts.get(o.track_id, 0) + 1
                span = spans.get(o.track_id)
                if span is None:
                    spans[o.track_id] = [o.frame_ref.stream_id, int(o.frame_ref.frame_idx), int(o.frame_ref.frame_idx)]
                else:
                    span[1] = min(span[1], int(o.frame_ref.frame_idx))
                    span[2] = max(span[2], int(o.frame_ref.frame_idx))

        embeddings = {t: _l2norm(total / max(counts[t], 1)) for t, total in sums.items()}
        span_tuples = {t: (s[0], s[1], s[2]) for t, s in spans.items()}
        bridges = self._bridges(instance_artifact, track_artifact)
        merge_map = build_cross_occlusion_merge_map(
            embeddings, span_tuples, threshold=self.merge_threshold,
            ratio_margin=self.ratio_margin, bridges=bridges,
        )
        return embeddings, merge_map

    @staticmethod
    def _bridges(instance_artifact, track_artifact) -> dict[tuple[str, str], int]:
        """Point tracks shared between tracklets, per stream. Empty when unavailable."""
        if track_artifact is None:
            return {}
        payloads = getattr(track_artifact, 'track_payloads_by_stream', {}) or {}
        if not payloads:
            return {}
        by_stream: dict[str, dict[int, dict[str, Any]]] = {}
        for obs in instance_artifact.observations:
            mask = instance_artifact.mask_arrays.get(obs.instance_id)
            if mask is None:
                continue
            stream = obs.frame_ref.stream_id
            by_stream.setdefault(stream, {}).setdefault(int(obs.frame_ref.frame_idx), {})[
                str(obs.track_id)
            ] = mask
        out: dict[tuple[str, str], int] = {}
        for stream_id, masks_by_frame in by_stream.items():
            payload = payloads.get(stream_id) or {}
            if not payload:
                continue
            point_sets = tracklet_point_sets(
                masks_by_frame,
                frame_indices=np.asarray(payload.get('frame_indices', []), dtype=int),
                points_xy=np.asarray(
                    payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32))
                ),
                visibility=payload.get('visibility'),
            )
            out.update(bridge_counts(point_sets))
        return out

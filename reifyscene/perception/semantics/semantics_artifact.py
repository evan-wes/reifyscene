"""Artifact for the ``semantics`` stage: per-tracklet open-vocab labels.

The ``semantics`` stage (design §4A) consumes the :class:`InstanceArtifact`
masks + the ``reid`` canonical ids and adds an open-vocab **category** (and, at
FP4b, free-form **attributes/caption**) per canonical tracklet — the
``"open-vocab semantic handle (embedding + caption)"`` the graph wants on its
entity nodes (design §5a). It replaces the COCO-80 detector string the floor
carries today.

Per-track records (category + score + top-k + attributes + caption) persist as
JSON; the SigLIP image embeddings persist as compressed ``.npz`` (one per track)
with an index in ``artifact.json``, mirroring :class:`ReidArtifact`. The ladder
manifest lives under ``metadata['capability']``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reifyscene.common.backend_ladder import CapabilityManifest
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact

UNKNOWN_CATEGORY = 'unknown'


@dataclass
class SemanticRecord:
    """Open-vocab semantics for one canonical tracklet.

    ``category`` is ``"unknown"`` whenever the rung declined to name the track.
    An abstention is a *value*, not a missing field: ``abstained`` says the rung
    was asked and refused, ``abstain_reason`` says why, and ``score``/``margin``/
    ``category_topk`` still carry the (rejected) evidence so a consumer can audit
    the call. A record with ``abstained=False`` and ``category="unknown"`` was
    never asked -- a different thing entirely.
    """

    category: str = UNKNOWN_CATEGORY
    score: float = 0.0
    category_topk: list[tuple[str, float]] = field(default_factory=list)  # [(label, score), ...]
    # top1 - top2 cosine, or None when there was no runner-up to measure against.
    # The discriminative signal: an off-domain vocabulary produces a high-scoring
    # but near-flat ranking, so absolute score does not separate a real hit from a
    # forced one and this does. Measurements in `score_zero_shot`.
    margin: float | None = None
    abstained: bool = False
    abstain_reason: str | None = None  # 'low_margin' | 'low_score' | 'no_labels' | 'no_concept'
    attributes: dict[str, Any] = field(default_factory=dict)
    caption: Optional[str] = None  # free-form; filled by Qwen-VL at FP4b

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['category_topk'] = [[str(lbl), float(s)] for lbl, s in self.category_topk]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'SemanticRecord':
        topk = [(str(lbl), float(s)) for lbl, s in raw.get('category_topk', [])]
        return cls(
            category=str(raw.get('category', UNKNOWN_CATEGORY)),
            score=float(raw.get('score', 0.0)),
            category_topk=topk,
            margin=None if raw.get('margin') is None else float(raw['margin']),
            abstained=bool(raw.get('abstained', False)),
            abstain_reason=raw.get('abstain_reason'),
            attributes=dict(raw.get('attributes', {})),
            caption=raw.get('caption'),
        )


@dataclass
class SemanticsArtifact(BaseArtifact):
    """Per-canonical-tracklet open-vocab category (+ embedding, attributes, caption)."""

    records: dict[str, SemanticRecord] = field(default_factory=dict)  # canonical track_id -> record
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)   # canonical track_id -> SigLIP vector
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    # --- accessor -----------------------------------------------------------
    def category_for(self, track_id: str) -> str:
        """Open-vocab category for a (canonical) track id, or ``"unknown"``."""
        rec = self.records.get(track_id)
        return rec.category if rec is not None else UNKNOWN_CATEGORY

    # --- capability manifest ------------------------------------------------
    def attach_manifest(self, manifest: CapabilityManifest) -> None:
        self.metadata['capability'] = manifest.to_dict()

    @property
    def capability(self) -> Optional[CapabilityManifest]:
        raw = self.metadata.get('capability')
        return CapabilityManifest.from_dict(raw) if raw else None

    # --- persistence (records → json, embeddings → npz + index) -------------
    def save(self) -> None:
        ensure_dir(self.save_dir)
        emb_dir = ensure_dir(self.save_dir / 'embeddings')
        emb_index: dict[str, str] = {}
        for track_id, emb in self.embeddings.items():
            filename = f"{self._sanitize_key(track_id)}.npz"
            np.savez_compressed(emb_dir / filename, embedding=np.asarray(emb, dtype=np.float32))
            emb_index[track_id] = filename
        payload = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'save_dir': str(self.save_dir),
            'metadata': dict(self.metadata),
            'records': {tid: rec.to_dict() for tid, rec in self.records.items()},
            'embedding_index': emb_index,
            'notes': list(self.notes),
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> 'SemanticsArtifact':
        raw = read_json(save_dir / 'artifact.json')
        embeddings: dict[str, np.ndarray] = {}
        for track_id, filename in raw.get('embedding_index', {}).items():
            path = require_file(save_dir / 'embeddings' / filename, "semantics embedding")
            with np.load(path, allow_pickle=True) as npz:
                embeddings[track_id] = np.asarray(npz['embedding'], dtype=np.float32)
        records = {tid: SemanticRecord.from_dict(r) for tid, r in raw.get('records', {}).items()}
        return cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw.get('metadata', {}),
            records=records,
            embeddings=embeddings,
            notes=raw.get('notes', []),
        )

    def summary(self) -> dict[str, Any]:
        cap = self.capability
        named = sum(1 for r in self.records.values() if r.category != UNKNOWN_CATEGORY)
        return {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'num_tracks': len(self.records),
            'num_named': named,
            'num_embeddings': len(self.embeddings),
            'backend_used': None if cap is None else cap.backend_used,
            'fidelity': None if cap is None else cap.fidelity,
        }

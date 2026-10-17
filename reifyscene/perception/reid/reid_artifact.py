"""Artifact for the ``reid`` stage: tracklet appearance features + gap merges.

The ``reid`` stage (design §4A) consumes the :class:`InstanceArtifact` track
ids and adds two things the identity backbone alone does not solve:

- a **per-tracklet appearance embedding** (DINOv3-L at the top of the ladder,
  down to ResNet18 / color-histogram at the floor); and
- a **merge map** that re-acquires identity *across occlusion gaps* — two
  tracklets that SAM 3 / the floor associator gave separate ids but that are the
  same entity get folded onto one canonical id.

It enriches identity rather than owning masks: ``objects`` / ``humans`` read the
merge map to canonicalize track ids before building their nodes. Embeddings are
persisted as compressed ``.npz`` (one per track) with an index in
``artifact.json``; the ladder manifest lives under ``metadata['capability']``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reifyscene.common.backend_ladder import CapabilityManifest
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact


@dataclass
class ReidArtifact(BaseArtifact):
    """Per-tracklet embeddings + a cross-occlusion identity merge map."""

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # track_id -> vector
    merge_map: dict[str, str] = field(default_factory=dict)  # track_id -> canonical id (gaps folded)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    # --- merge map ----------------------------------------------------------
    def canonical(self, track_id: str) -> str:
        """Follow the merge map to the canonical id (path-compressed, cycle-safe)."""
        seen: set[str] = set()
        current = track_id
        while current in self.merge_map and self.merge_map[current] != current:
            if current in seen:  # defensive: never loop on a malformed map
                break
            seen.add(current)
            current = self.merge_map[current]
        return current

    def merged_groups(self) -> dict[str, list[str]]:
        """Inverse of the merge map: canonical id -> [original track ids]."""
        groups: dict[str, list[str]] = {}
        ids = set(self.embeddings) | set(self.merge_map)
        for track_id in sorted(ids):
            groups.setdefault(self.canonical(track_id), []).append(track_id)
        return groups

    # --- capability manifest ------------------------------------------------
    def attach_manifest(self, manifest: CapabilityManifest) -> None:
        self.metadata['capability'] = manifest.to_dict()

    @property
    def capability(self) -> Optional[CapabilityManifest]:
        raw = self.metadata.get('capability')
        return CapabilityManifest.from_dict(raw) if raw else None

    # --- persistence (embeddings → npz + index, merge map → json) -----------
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
            'metadata': dict(self.metadata),
            'merge_map': dict(self.merge_map),
            'embedding_index': emb_index,
            'notes': list(self.notes),
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> 'ReidArtifact':
        raw = read_json(save_dir / cls.FILE)
        embeddings: dict[str, np.ndarray] = {}
        for track_id, filename in raw['embedding_index'].items():
            path = require_file(save_dir / 'embeddings' / filename, "reid embedding")
            with np.load(path, allow_pickle=True) as npz:
                embeddings[track_id] = np.asarray(npz['embedding'], dtype=np.float32)
        return cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw['metadata'],
            embeddings=embeddings,
            merge_map=raw['merge_map'],
            notes=raw['notes'],
        )

    def summary(self) -> dict[str, Any]:
        cap = self.capability
        groups = self.merged_groups()
        return {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'num_embeddings': len(self.embeddings),
            'num_input_tracks': len(set(self.embeddings) | set(self.merge_map)),
            'num_canonical_tracks': len(groups),
            'backend_used': None if cap is None else cap.backend_used,
            'fidelity': None if cap is None else cap.fidelity,
        }

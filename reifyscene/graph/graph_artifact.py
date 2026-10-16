"""Artifact for dynamic scene graphs.

Node records persist as JSON; the per-entity semantic embeddings persist as compressed
``.npz`` (one per node) with an index in ``artifact.json``, mirroring
:class:`~reifyscene.perception.semantics.semantics_artifact.SemanticsArtifact`. They are
kept out of the JSON because a node's identity vector is a thousand floats and the graph
is a document a person reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, GraphEdge, GraphNode


@dataclass
class GraphArtifact(BaseArtifact):
    """Store scene-graph nodes, edges, and temporal state summaries."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # node_id -> vector
    node_histories: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    def save(self) -> None:
        """Write the graph, with each node's embedding beside it as its own ``.npz``."""
        ensure_dir(self.save_dir)
        payload = self.to_payload()
        index: dict[str, str] = {}
        if self.embeddings:
            emb_dir = ensure_dir(self.save_dir / 'embeddings')
            for node_id, emb in self.embeddings.items():
                filename = f"{self._sanitize_key(node_id)}.npz"
                np.savez_compressed(
                    emb_dir / filename, embedding=np.asarray(emb, dtype=np.float32)
                )
                index[node_id] = filename
        payload.pop('embeddings', None)  # the vectors live beside the JSON, not inside it
        payload['embedding_index'] = index
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> GraphArtifact:
        """Read the graph back, embeddings included."""
        raw = read_json(Path(save_dir) / cls.FILE)
        index = raw.pop('embedding_index')
        # save() moves the vectors out of the JSON into embedding_index; the field
        # is rebuilt from those files below.
        raw['embeddings'] = {}
        artifact = cls.from_payload(raw, save_dir)
        for node_id, filename in index.items():
            emb_path = Path(save_dir) / 'embeddings' / filename
            artifact.embeddings[str(node_id)] = np.load(emb_path)['embedding']
        return artifact

    def node_ids(self) -> list[str]:
        """Return all node ids in insertion order."""
        return [node.node_id for node in self.nodes]

    def summary(self) -> dict[str, Any]:
        """Return a compact graph summary."""
        return {
            "run_id": self.run_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "notes": self.notes,
        }

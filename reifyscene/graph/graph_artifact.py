"""Artifact for dynamic scene graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reifyscene.common.types import BaseArtifact, GraphEdge, GraphNode


@dataclass
class GraphArtifact(BaseArtifact):
    """Store scene-graph nodes, edges, and temporal state summaries."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    node_histories: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

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

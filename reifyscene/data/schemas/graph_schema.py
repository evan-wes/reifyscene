"""Schemas for scene-graph metadata."""

from dataclasses import dataclass


@dataclass
class GraphSchema:
    """Serializable graph description."""
    graph_id: str

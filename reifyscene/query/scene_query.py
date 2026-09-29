"""Structured queries over a finished run's scene graph -- no language model.

The query contract's rung 3 (``docs/design/2026-09-12-back-half-ladders-and-graph-contract.md``):
fixed templates, each returning entity ids with the evidence behind them. A natural-language
parser (rungs 1-2) is later work; it is meant to produce the same ``{template, args}`` this
takes, so everything downstream stays as it is.

Templates:

- ``select(category, frame=None)`` -- entities of a category (observed on that frame, if given)
- ``count(category, frame=None)`` -- how many
- ``when(entity)`` -- presence span, observed frames and gaps
- ``relate(entity, predicate)`` -- the graph has no relation edges yet; this says so rather than
  inventing an answer (relations are adopted from VLA-3D next)

A category matches the entity's semantic ``category`` or its label (not its top-k alternatives,
which are runners-up).
``unknown`` (the semantics stage abstained) is a category like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.graph.presence import Presence, presence_from_frames

ENTITY_TYPES = ("human", "object")
TEMPLATES = ("select", "count", "when", "relate")


class QueryError(ValueError):
    """A query the graph cannot answer as asked (bad template, unknown entity)."""


@dataclass
class Entity:
    node_id: str
    node_type: str
    label: str
    category: str
    category_topk: list[str]
    caption: str
    confidence: float
    abstained: bool
    presence: Presence
    positions: dict[int, list[float]] = field(default_factory=dict)
    extent: list[float] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def matches(self, category: str) -> bool:
        c = category.strip().lower()
        # Not the top-k alternatives: they are runners-up, and "toothbrush" is someone's fourth
        # guess for every person on lab-coat.
        return c in {self.category.lower(), self.label.lower()}

    def summary(self) -> dict[str, Any]:
        p = self.presence
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "category": self.category,
            "category_topk": self.category_topk,
            "caption": self.caption,
            "abstained": self.abstained,
            "confidence": self.confidence,
            "first": p.first,
            "last": p.last,
            "num_observed": len(p.observed),
            "gaps": [{"start": g.start, "end": g.end, "kind": g.kind} for g in p.gaps],
            "extent": self.extent,
        }


def _entity(node) -> Entity:
    a = dict(node.attrs or {})
    topk = a.get("category_topk") or []
    topk = [t[0] if isinstance(t, (list, tuple)) else (t.get("category") if isinstance(t, dict)
                                                         else t) for t in topk]
    positions = {int(p["frame_idx"]): list(p["xyz"]) for p in a.get("positions") or []
                 if "frame_idx" in p and "xyz" in p}
    return Entity(
        node_id=node.node_id,
        node_type=str(node.node_type),
        label=str(node.label or ""),
        category=str(a.get("category") or node.label or "unknown"),
        category_topk=[str(t) for t in topk if t],
        caption=str(a.get("caption") or ""),
        confidence=float(node.confidence or 0.0),
        abstained=bool(a.get("abstained", False)),
        presence=presence_from_frames(a.get("frame_indices") or []),
        positions=positions,
        extent=a.get("extent"),
        attrs=a,
    )


class SceneQuery:
    """The structured query surface over one run's graph."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.entities: dict[str, Entity] = {
            n.node_id: _entity(n) for n in graph.nodes if str(n.node_type) in ENTITY_TYPES
        }

    @classmethod
    def from_run(cls, run_dir: Path) -> SceneQuery:
        from reifyscene.graph.graph_artifact import GraphArtifact

        return cls(GraphArtifact.load(Path(run_dir) / "graph"))

    # -- lookups ------------------------------------------------------------------------------

    def entity(self, ref: str) -> Entity:
        """By node id, or by its trailing tracklet number (``8`` for ``...:sam2:cam_00:8``)."""
        if ref in self.entities:
            return self.entities[ref]
        hits = [e for e in self.entities.values() if e.node_id.rsplit(":", 1)[-1] == str(ref)]
        if len(hits) == 1:
            return hits[0]
        raise QueryError(f"no entity {ref!r}" if not hits else f"{ref!r} is ambiguous")

    def categories(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entities.values():
            out[e.category] = out.get(e.category, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    # -- templates ----------------------------------------------------------------------------

    def select(self, category: str, frame: int | None = None) -> list[Entity]:
        hits = [e for e in self.entities.values() if e.matches(category)]
        if frame is not None:
            hits = [e for e in hits if e.presence.present_at(int(frame))]
        return sorted(hits, key=lambda e: e.node_id)

    def count(self, category: str, frame: int | None = None) -> int:
        return len(self.select(category, frame))

    def when(self, ref: str) -> Presence:
        return self.entity(ref).presence

    def relate(self, ref: str, predicate: str) -> dict[str, Any]:
        self.entity(ref)
        rel = [e for e in self.graph.edges if str(e.edge_type) not in ("observed_by",)]
        if not rel:
            return {"available": False,
                    "reason": "the graph has no relation edges yet; relations (VLA-3D's "
                              "predicates) are the next piece of the graph to build"}
        raise QueryError("relation edges exist but relate() does not read them yet")

    # -- one entry point for CLI and web ------------------------------------------------------

    def run(self, template: str, args: dict[str, Any]) -> dict[str, Any]:
        """``{template, args}`` -> a JSON-able answer with evidence."""
        t = template.strip().lower()
        frame = args.get("frame")
        frame = None if frame in (None, "") else int(frame)
        if t == "select":
            hits = self.select(str(args["category"]), frame)
            return {"template": t, "args": args, "entities": [e.summary() for e in hits],
                    "count": len(hits)}
        if t == "count":
            return {"template": t, "args": args, "count": self.count(str(args["category"]), frame)}
        if t == "when":
            e = self.entity(str(args["entity"]))
            return {"template": t, "args": args, "entity": e.node_id,
                    "presence": e.presence.to_dict()}
        if t == "relate":
            return {"template": t, "args": args,
                    **self.relate(str(args["entity"]), str(args.get("predicate", "")))}
        raise QueryError(f"unknown template {template!r}; have {', '.join(TEMPLATES)}")

"""Structured queries over a finished run's scene graph -- no language model.

The query contract's rung 3 (``docs/design/2026-09-12-back-half-ladders-and-graph-contract.md``):
fixed templates, each returning entity ids with the evidence behind them. A natural-language
parser (rungs 1-2) is later work; it is meant to produce the same ``{template, args}`` this
takes, so everything downstream stays as it is.

Templates:

- ``select(category, frame=None)`` -- entities of a category (observed on that frame, if given)
- ``count(category, frame=None)`` -- how many
- ``when(entity)`` -- presence span, observed frames and gaps
- ``relate(entity, predicate=None, frame=None)`` -- the entity's relation edges (VLA-3D's
  predicates, reifyscene/graph/relations.py) in either direction, each with its validity
  intervals; at a frame, only those holding then. A graph built without relations says so.

A category matches the entity's semantic ``category`` or its label (not its top-k alternatives,
which are runners-up).
``unknown`` (the semantics stage abstained) is a category like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.graph.node_ids import tracklet_number
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
        from reifyscene.api.run_outputs import RunOutputs

        return cls(RunOutputs.open(run_dir).artifact("graph"))

    # -- lookups ------------------------------------------------------------------------------

    def entity(self, ref: str) -> Entity:
        """By node id, or by its trailing tracklet number (``8`` for ``...:sam2:cam_00:8``)."""
        if ref in self.entities:
            return self.entities[ref]
        hits = [e for e in self.entities.values() if tracklet_number(e.node_id) == str(ref)]
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

    def relations(self) -> list:
        from reifyscene.graph.relations import PREDICATES

        return [e for e in self.graph.edges if str(e.edge_type) in PREDICATES]

    def relate(self, ref: str, predicate: str | None = None,
               frame: int | None = None) -> dict[str, Any]:
        """Relations involving an entity, as sentences with their intervals."""
        e = self.entity(ref)
        rels = self.relations()
        if not rels and not (self.graph.metadata or {}).get("relations"):
            return {"available": False,
                    "reason": "this run's graph has no relations: the relation computer "
                              "(VLA-3D) was not available when the graph was built"}
        out = []
        for r in rels:
            also = (r.attrs or {}).get("and")
            if e.node_id not in (r.src, r.dst, also):
                continue
            pred, src, dst = self._from_side_of(e.node_id, str(r.edge_type), r.src, r.dst)
            if predicate and pred != predicate:
                continue
            iv = (r.attrs or {}).get("intervals") or []
            if frame is not None and not any(a <= frame <= b for a, b in iv):
                continue
            out.append({"predicate": pred, "src": src, "dst": dst, "and": also,
                        "stored_as": [str(r.edge_type), r.src, r.dst],
                        "intervals": iv, "confidence": r.confidence,
                        "sentence": self._sentence(pred, src, dst, also)})
        out.sort(key=lambda x: (x["predicate"], -x["confidence"]))
        return {"available": True, "entity": e.node_id, "relations": out, "count": len(out)}

    @staticmethod
    def _from_side_of(node: str, pred: str, src: str, dst: str) -> tuple[str, str, str]:
        """Phrase a relation with ``node`` as its subject where that is a true rephrasing.

        Each vertical fact is stored once (``B above A``; the mirrored ``A below B`` is dropped
        when the graph is built), so asking about A reads it backwards: "A is below B". ``near``
        is symmetric. ``on``, ``in``, ``between`` and the rankings keep their stored subject --
        their converses ("has X on it") are not predicates of the graph.
        """
        if node == dst and pred in ("above", "below"):
            return ("below" if pred == "above" else "above"), dst, src
        if node == dst and pred == "near":
            return pred, dst, src
        return pred, src, dst

    def _name(self, node_id: str) -> str:
        e = self.entities.get(node_id)
        return f"{e.category} {tracklet_number(node_id)}" if e else node_id

    def _sentence(self, pred: str, src: str, dst: str, also: str | None) -> str:
        if pred == "between":
            return f"{self._name(src)} is between {self._name(dst)} and {self._name(also or '')}"
        if pred in ("closest", "farthest"):
            cat = self.entities[src].category if src in self.entities else "entity"
            return f"{self._name(src)} is the {pred} {cat} to {self._name(dst)}"
        verb = {"near": "is near", "above": "is above", "below": "is below", "on": "is on",
                "in": "is in"}.get(pred, pred)
        return f"{self._name(src)} {verb} {self._name(dst)}"

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
            pred = str(args.get("predicate") or "") or None
            return {"template": t, "args": args,
                    **self.relate(str(args["entity"]), None if pred in (None, "any") else pred,
                                  frame)}
        raise QueryError(f"unknown template {template!r}; have {', '.join(TEMPLATES)}")

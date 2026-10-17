"""The ``gaussians`` layer: which of the trained scene's Gaussians each entity is.

The graph is built before the scene is trained, so the graph stage cannot say what an entity is
made of in the render. This layer adds it afterwards, following the graph's layer rule
(``docs/design/2026-09-30-graph-layers.md``): a later stage adds attributes under its own key,
stamped with what it was computed from, and never changes what an earlier stage wrote.

What it adds:

- on each entity node, ``attrs["gaussians"]``: ``status`` (``grounded`` when any Gaussian was
  assigned, ``unassignable`` when none was), the counts in the moving (fg) and static (bg)
  fields, the fraction of the scene assigned, and what the floater filter dropped;
- the index arrays themselves in ``<graph>/layers/gaussians.npz`` (a person is ~10^5 indices,
  too many for a document a person reads), keyed by node;
- ``metadata["layers"]["gaussians"]``: the checkpoint it was computed against (its folder and a
  content digest), how the Gaussians were selected (``selection``: ``owner`` on a checkpoint
  trained with owners, else ``flashsplat`` with its parameters), and the ``renderable`` grade.

Indices are only meaningful for the checkpoint they came from: a retrained scene has other
Gaussians in another order. :func:`load_handles` therefore refuses a checkpoint whose digest
differs, rather than handing back indices that would remove the wrong parts of the scene.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reifyscene.common.types import BaseArtifact

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import Checkpoint, RunOutputs

LAYER = "gaussians"
LAYER_FILE = Path("layers") / "gaussians.npz"
ENTITY_TYPES = ("human", "object")


@dataclass
class HandlesArtifact(BaseArtifact):
    """The ``handles`` stage's own record: what it wrote onto the graph, and from what.

    The substance is on the graph (the ``gaussians`` layer); this is what resume copies and
    fingerprints, so it carries the grade, the checkpoint digest and why a grade is absent.
    """

    result: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "stage_name": self.stage_name, **self.result}


class StaleHandles(RuntimeError):
    """The graph's handles were computed against a different checkpoint."""


def _key(node_id: str, part: str) -> str:
    return f"{node_id.replace(':', '__').replace('/', '_')}--{part}"


def compute_handles(
    editor,
    node_ids: list[str],
    *,
    method: str = "flashsplat",
    slackness: float = 0.0,
    filtered: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """``{node_id: (fg_idx, bg_idx, stats)}`` for every entity: by owner, or by FlashSplat
    with one evidence pass for all.

    ``editor`` is a :class:`~reifyscene.edit.remove.RemoveEditor` (it holds the checkpoint on
    the GPU and caches the evidence). An entity the id maps do not carry -- never segmented in
    any training frame -- gets empty indices and ``stats["reason"]``.
    """
    if method == "owner":
        from reifyscene.graph.renderable_handle import owner_handles

        known = set(owner_handles(editor.bundle))
    else:
        editor.evidence()
        known = set(n for n in editor._node_ids if n is not None)
    out: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    empty = np.array([], dtype=np.int64)
    for node in node_ids:
        if node not in known:
            out[node] = (empty, empty, {"reason": "not in the id maps"})
            continue
        fg, bg, stats = editor.select(node, method=method, slackness=slackness,
                                      filtered=filtered)
        stats["coverage_deg"] = editor.view_span(node, fg, bg)
        out[node] = (np.asarray(fg, np.int64), np.asarray(bg, np.int64), stats)
    return out


def write_layer(
    graph,
    handles: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    checkpoint: Checkpoint,
    num_gaussians: int,
    selection: str = "flashsplat",
    slackness: float | None = None,
    filtered: bool | None = None,
    body_bound: bool = False,
    stage: str = "handles",
) -> str:
    """Put the layer on ``graph`` (node attrs, index file, metadata) and save it.

    Replaces an earlier ``gaussians`` layer wholesale; touches nothing else. Returns the
    ``renderable`` grade: grounded (every entity has Gaussians), degraded (some do), absent.
    """
    arrays: dict[str, np.ndarray] = {}
    grounded = 0
    entities = [n for n in graph.nodes if n.node_type in ENTITY_TYPES]
    for node in entities:
        fg, bg, stats = handles.get(node.node_id, (np.array([], np.int64),) * 2 + ({},))
        taken = int(len(fg) + len(bg))
        grounded += taken > 0
        arrays[_key(node.node_id, "fg")], arrays[_key(node.node_id, "bg")] = fg, bg
        node.attrs[LAYER] = {
            "status": "grounded" if taken else "unassignable",
            "fg": int(len(fg)), "bg": int(len(bg)),
            "assigned_fraction": round(taken / max(num_gaussians, 1), 6),
            **{k: v for k, v in stats.items()
               if k in ("filter_dropped", "reason", "outside_body", "body_frames",
                        "coverage_deg")},
        }
    grade = ("grounded" if entities and grounded == len(entities)
             else "degraded" if grounded else "absent")
    path = Path(graph.save_dir) / LAYER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    layers = graph.metadata.setdefault("layers", {})
    layers[LAYER] = {
        "stage": stage,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": {"path": checkpoint.name, "digest": checkpoint.digest()},
        "selection": selection,
        "method": ("ownership: owned from seeding, trained to the owner's pixels"
                   if selection == "owner" else
                   "FlashSplat evidence majority" + (" + floater filter" if filtered else "")
                   + (" + body bound (people)" if body_bound else "")),
        "body_bound": bool(body_bound),
        "slackness": slackness,
        "file": str(LAYER_FILE),
        "provides": {"renderable": grade},
        "counts": {"entities": len(entities), "grounded": grounded},
    }
    graph.save()
    return grade


def layer_info(graph) -> dict[str, Any] | None:
    """The layer's metadata, or None when the graph has no ``gaussians`` layer."""
    return (graph.metadata or {}).get("layers", {}).get(LAYER)


def load_handles(run: RunOutputs, node_id: str, *,
                 checkpoint: Checkpoint) -> tuple[np.ndarray, np.ndarray]:
    """``(fg_idx, bg_idx)`` of one entity, refused unless computed against this checkpoint.

    Raises ``KeyError`` when there is no layer or no such entity in it, ``StaleHandles`` when
    the layer came from another checkpoint.
    """
    meta = run.record("graph")["metadata"]
    info = meta.get("layers", {}).get(LAYER)
    if info is None:
        raise KeyError("the graph has no gaussians layer")
    want = checkpoint.digest()
    if info["checkpoint"]["digest"] != want:
        raise StaleHandles(f"handles are for checkpoint {info['checkpoint']['path']} "
                           f"({info['checkpoint']['digest']}), not {want}")
    with np.load(run.stage_dir("graph") / info["file"]) as z:
        fk, bk = _key(node_id, "fg"), _key(node_id, "bg")
        if fk not in z.files:
            raise KeyError(f"no handles for {node_id}")
        return z[fk], z[bk]


def run_handles(graph, render, *, run: RunOutputs, device: str, slackness: float = 0.0,
                filtered: bool = True, body_bound: bool = False) -> dict[str, Any]:
    """The ``handles`` stage: bind every entity to its Gaussians and write the layer.

    Absent, with the reason, when there is nothing to bind: no entities, or no trained dynamic
    checkpoint (a static or empty render has no per-entity motion to separate).
    """
    entities = [n.node_id for n in graph.nodes if n.node_type in ENTITY_TYPES]
    state = getattr(render, "scene_state", {}) or {}
    ckpt = state.get("checkpoint")
    if not entities:
        return {"renderable": "absent", "reason": "the graph has no entities"}
    if state.get("kind") != "dynamic" or not ckpt:
        return {"renderable": "absent",
                "reason": f"no trained dynamic checkpoint (render kind {state.get('kind')!r})"}
    from reifyscene.edit.remove import RemoveEditor

    checkpoint = run.checkpoint(Path(render.save_dir).name)   # the saved render, as recorded
    editor = RemoveEditor(checkpoint, device=device, body_bound=body_bound and filtered)
    editor.use_layer = False          # this is what writes the layer: never read it back
    method = editor.default_method    # owners when the checkpoint was trained with them
    if method == "flashsplat":
        editor.evidence(refresh=True)     # the id maps and evidence of THIS graph and checkpoint
    else:
        slackness, filtered, body_bound = None, None, False
    handles = compute_handles(editor, entities, method=method, slackness=slackness,
                              filtered=filtered)
    grade = write_layer(graph, handles, checkpoint=checkpoint,
                        num_gaussians=int(editor.renderer.scene.num_gaussians),
                        selection=method, slackness=slackness, filtered=filtered,
                        body_bound=bool(body_bound and filtered))
    info = layer_info(graph)
    return {"renderable": grade, "layer": LAYER, "checkpoint": info["checkpoint"],
            "counts": info["counts"],
            "unassignable": sorted(n for n, (fg, bg, _) in handles.items()
                                   if len(fg) + len(bg) == 0)}

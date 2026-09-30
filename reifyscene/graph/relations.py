"""Relations between entities, per frame, from VLA-3D's relation functions.

The graph contract (§3.2 of ``docs/design/2026-09-12-back-half-ladders-and-graph-contract.md``)
adopts VLA-3D's eight view-independent predicates -- ``above``, ``below``, ``on``, ``in``,
``near``, ``between``, ``closest``, ``farthest`` -- and this module calls **their code**
(``third_party/VLA-3D/scene_graph``, MIT, pinned by the setup script, two patches in
``third_party_patches/vla-3d/``) rather than re-deriving it. What it adds is the adaptation to our
scenes, which differ from theirs in four ways:

- **Time.** VLA-3D relates one static snapshot. We call it on every frame, over the entities
  present on that frame, then group each relation's frames into *validity intervals*. An interval
  never spans a frame where either endpoint is absent, so a relation ends at an occlusion gap --
  the contract's rule that absence is not evidence the relation was false. Runs shorter than
  ``min_frames`` are dropped as flicker.
- **Units.** Their thresholds are metres; our scale is arbitrary. Rather than convert their
  thresholds, our boxes are converted **into metres** and their functions run with their own
  defaults. The scale comes from people where a scene has them (a person's median box height is
  taken as 1.7 m); otherwise the scene's extent is taken as a typical room's. The source used is
  recorded in the graph's metadata.
- **Boxes.** Theirs are oriented 3D boxes. Ours are axis-aligned in a gravity-aligned world:
  the entity's centre, ±2 of its robust per-axis spread (a MAD-sigma). Our positions come from
  camera-facing surfaces, so boxes are thin in depth, which weakens tests that need horizontal
  overlap (``above``, ``below``, ``on``).
- **Classes.** Their class rules (containers for ``in``, furniture that supports ``on``, walls
  that never do) are keyed by NYU40 ids. The few COCO categories those rules concern are mapped
  (``NYU_OF``); every other category gets its own id outside their lists, so only their default
  rules apply to it, and closest/farthest still group by our category.

Vertical predicates (``above``, ``below``, ``on``, ``in``) are emitted only when the world frame
is gravity-aligned, as the contract requires; otherwise they are not computed in camera axes.

Edge direction: ``src <predicate> dst`` reads as a sentence -- ``cup on table``,
``person near dog``, ``bird between cup and handbag`` (the second anchor in ``attrs["and"]``),
``dog closest person`` (the closest dog to that person).
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VLA3D_COMMIT = "a0c023c9662bd875f0b3aeff4ff60adc1991e449"
PREDICATES = ("above", "below", "on", "in", "near", "between", "closest", "farthest")
VERTICAL = ("above", "below", "on", "in")
PERSON_HEIGHT_M = 1.7
FALLBACK_SCENE_EXTENT_M = 5.0
BOX_SIGMAS = 2.0

# VLA-3D's argparse defaults (scene_graph/generate_scene_info.py), used unchanged.
VLA3D_DEFAULTS: dict[str, float] = {
    "between_iom": 0.5, "vertical_iom": 0.5, "near_thres": 0.01, "overlap_thres": 0.3,
    "symmetry_thres": 0.5, "distance_thres": 1.0, "anchor_size_thres": 1.5, "on_thres": 0.01,
    "in_thres": 0.1, "under_thres": 0.01, "ordered_thres": 0.2,
}

# COCO category -> NYU40 id, for the categories VLA-3D's class rules concern.
NYU_OF: dict[str, int] = {
    "chair": 5, "couch": 83, "sofa": 83, "bed": 157, "bench": 204, "dining table": 19,
    "table": 19, "refrigerator": 17, "sink": 24, "microwave": 13, "oven": 238,
    "toaster": 275,
}
_SYNTHETIC_ID_BASE = 100_000


class RelationsUnavailable(RuntimeError):
    """VLA-3D is not checked out (or its dependencies are missing)."""


def vla3d_dir() -> Path:
    env = os.environ.get("REIFYSCENE_VLA3D_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "third_party" / "VLA-3D"


def load_vla3d():
    """Import VLA-3D's relation module from the pinned checkout."""
    scene_graph = vla3d_dir() / "scene_graph"
    if not (scene_graph / "generate_scene_info.py").is_file():
        raise RelationsUnavailable(
            f"VLA-3D is not checked out at {vla3d_dir()}; run scripts/setup_reifyscene.sh "
            "(it clones it by default) or set REIFYSCENE_VLA3D_DIR")
    if str(scene_graph) not in sys.path:
        sys.path.insert(0, str(scene_graph))
    try:
        return importlib.import_module("generate_scene_info")
    except ImportError as exc:
        raise RelationsUnavailable(f"VLA-3D's relation code cannot be imported: {exc}") from exc


# -- boxes and scale --------------------------------------------------------------------------

def box_corners(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(8, 3) corners in VLA-3D's order: the top four, then the bottom four."""
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * half[0]
    y = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * half[1]
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * half[2]
    return np.stack([x, y, z], axis=1) + center


@dataclass
class EntityTrack:
    node_id: str
    category: str
    # frame -> (centre, half-size)
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def entity_tracks(nodes) -> list[EntityTrack]:
    """Per-frame boxes from the graph's entity nodes (their ``positions``)."""
    out = []
    for n in nodes:
        if str(n.node_type) not in ("human", "object"):
            continue
        a = n.attrs or {}
        fallback = a.get("extent")
        t = EntityTrack(n.node_id, str(a.get("category") or n.label or "unknown"))
        for p in a.get("positions") or []:
            ext = p.get("extent") or fallback
            if "xyz" not in p or ext is None:
                continue
            t.boxes[int(p["frame_idx"])] = (np.asarray(p["xyz"], float),
                                            BOX_SIGMAS * np.abs(np.asarray(ext, float)))
        if t.boxes:
            out.append(t)
    return out


def metric_scale(tracks: list[EntityTrack]) -> tuple[float, dict[str, Any]]:
    """Scene units per metre, and how it was decided."""
    heights = [2 * half[2] for t in tracks if t.category == "person"
               for _c, half in t.boxes.values() if half[2] > 0]
    if heights:
        h = float(np.median(heights))
        return h / PERSON_HEIGHT_M, {"source": "person_height", "person_height_m": PERSON_HEIGHT_M,
                                     "median_person_height_units": round(h, 6),
                                     "samples": len(heights)}
    pts = np.array([c for t in tracks for c, _h in t.boxes.values()])
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) > 1 else 1.0
    return max(diag, 1e-9) / FALLBACK_SCENE_EXTENT_M, {
        "source": "scene_extent", "assumed_extent_m": FALLBACK_SCENE_EXTENT_M,
        "extent_units": round(diag, 6)}


# -- per frame --------------------------------------------------------------------------------

def _objects(tracks: list[EntityTrack], frame: int, to_m: float, class_id: dict[str, int],
             min_half_m: float) -> list[dict[str, Any]]:
    objs = []
    for t in tracks:
        if frame not in t.boxes:
            continue
        c, half = t.boxes[frame]
        c_m, half_m = c * to_m, np.maximum(half * to_m, min_half_m)
        objs.append({"object_id": t.node_id, "nyu_id": class_id[t.category], "center": c_m.tolist(),
                     "size": (2 * half_m).tolist(), "bbox": box_corners(c_m, half_m).tolist()})
    return objs


Relation = tuple[str, str, str, str | None]


def relations_on_frame(g, objects: list[dict[str, Any]], *, vertical: bool,
                       th: dict[str, float] = VLA3D_DEFAULTS) -> list[Relation]:
    """``(predicate, src, dst, and)`` holding on one frame, from VLA-3D's functions."""
    if len(objects) < 2:
        return []
    pts = np.array([p for o in objects for p in o["bbox"]])
    lo, hi = pts.min(0), pts.max(0)
    region = box_corners((lo + hi) / 2, (hi - lo) / 2).tolist()
    same_class = g.group_by_class(objects)
    out: list[tuple[str, str, str, str | None]] = []
    for i, anchor in enumerate(objects):
        a = anchor["object_id"]
        if vertical:
            for tgt in g.relate_above(th["vertical_iom"], th["on_thres"], i, objects):
                out.append(("above", tgt, a, None))
            for tgt in g.relate_below(th["vertical_iom"], th["under_thres"], i, objects):
                out.append(("below", tgt, a, None))
            for tgt in g.relate_on(th["vertical_iom"], th["on_thres"], th["under_thres"],
                                   th["in_thres"], i, objects):
                out.append(("on", tgt, a, None))
            for tgt in g.relate_in(i, objects, th["in_thres"]):
                out.append(("in", tgt, a, None))
        for tgt in g.relate_near(th["near_thres"], i, objects, region):
            s, d = sorted((tgt, a))                      # symmetric: one edge per pair
            out.append(("near", s, d, None))
        for first, second in g.relate_between(th["between_iom"], i, objects, th["overlap_thres"],
                                              th["symmetry_thres"], th["distance_thres"],
                                              th["anchor_size_thres"]):
            p, q = sorted((first, second))
            out.append(("between", a, p, q))
        closest, farthest = g.relate_ordered(i, objects, same_class, th["ordered_thres"])
        for ranked in closest:
            out.append(("closest", ranked[0], a, None))
        for ranked in farthest:
            out.append(("farthest", ranked[0], a, None))
    return sorted(set(out), key=lambda r: (r[0], r[1], r[2], r[3] or ""))


def _intervals(frames: list[int], present_both: list[int], min_frames: int) -> list[list[int]]:
    """Runs of consecutive both-present frames where the relation held."""
    held = set(frames)
    runs, cur = [], []
    for f in present_both:
        if f in held and (not cur or f == cur[-1] + 1):
            cur.append(f)
            continue
        if len(cur) >= min_frames:
            runs.append([cur[0], cur[-1]])
        cur = [f] if f in held else []
    if len(cur) >= min_frames:
        runs.append([cur[0], cur[-1]])
    return runs


def compute_relations(graph, *, min_frames: int = 2) -> tuple[list, dict[str, Any]]:
    """``(GraphEdge list, metadata)`` for every predicate over every frame."""
    from reifyscene.common.types import GraphEdge

    g = load_vla3d()
    tracks = entity_tracks(graph.nodes)
    world = (graph.metadata or {}).get("world_frame") or {}
    vertical = bool(world.get("gravity_aligned"))
    meta: dict[str, Any] = {"source": f"VLA-3D @ {VLA3D_COMMIT[:12]}", "thresholds": VLA3D_DEFAULTS,
                            "min_frames": min_frames, "box": f"centre ± {BOX_SIGMAS:g} sigma",
                            "gravity_aligned": vertical,
                            "predicates": [p for p in PREDICATES if vertical or p not in VERTICAL]}
    if len(tracks) < 2:
        meta.update(counts={}, metric_scale={"source": "none"})
        return [], meta
    units_per_m, scale_meta = metric_scale(tracks)
    to_m = 1.0 / units_per_m
    cats = sorted({t.category for t in tracks})
    class_id = {c: NYU_OF.get(c.lower(), _SYNTHETIC_ID_BASE + i) for i, c in enumerate(cats)}
    halves = [h for t in tracks for _c, half in t.boxes.values() for h in half * to_m if h > 0]
    min_half_m = 0.01 * float(np.median(halves)) if halves else 1e-4

    frames = sorted({f for t in tracks for f in t.boxes})
    held: dict[tuple[str, str, str, str | None], list[int]] = defaultdict(list)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for f in frames:
            objs = _objects(tracks, f, to_m, class_id, min_half_m)
            for rel in relations_on_frame(g, objs, vertical=vertical):
                held[rel].append(f)

    present = {t.node_id: set(t.boxes) for t in tracks}
    edges, counts = [], defaultdict(int)
    for (pred, src, dst, also), fs in sorted(held.items(), key=lambda kv: kv[0][:3]):
        ids = [src, dst] + ([also] if also else [])
        both = sorted(set.intersection(*(present[i] for i in ids)))
        runs = _intervals(fs, both, min_frames)
        if not runs:
            continue
        n_held = sum(b - a + 1 for a, b in runs)
        attrs: dict[str, Any] = {"intervals": runs, "num_frames": n_held,
                                 "frames_both_present": len(both), "source": "vla3d"}
        if also:
            attrs["and"] = also
        edges.append(GraphEdge(src=src, dst=dst, edge_type=pred,
                               confidence=round(n_held / max(len(both), 1), 4), attrs=attrs))
        counts[pred] += 1
    meta.update(counts=dict(sorted(counts.items())), metric_scale=scale_meta | {
        "units_per_metre": round(units_per_m, 6)}, class_ids={c: class_id[c] for c in cats
                                                               if c.lower() in NYU_OF})
    return edges, meta


def add_relations(graph, *, min_frames: int = 2) -> dict[str, Any]:
    """Replace the graph's relation edges with freshly computed ones; return the metadata."""
    keep = [e for e in graph.edges if str(e.edge_type) not in PREDICATES]
    edges, meta = compute_relations(graph, min_frames=min_frames)
    graph.edges = keep + edges
    graph.metadata = dict(graph.metadata or {})
    graph.metadata["relations"] = meta
    return meta

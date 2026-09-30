"""Relations from VLA-3D's functions, adapted to our scenes (reifyscene/graph/relations.py).

These run VLA-3D's actual code (skipped when the checkout is absent). What they pin is the
adaptation, not VLA-3D's rules: boxes reach their functions in the corner order they expect and in
metres; a relation's intervals never span a frame where an endpoint is missing; vertical
predicates are not emitted without gravity; symmetric ``near`` is one edge per pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.common.types import GraphNode
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.relations import (
    _intervals,
    add_relations,
    box_corners,
    compute_relations,
    entity_tracks,
    metric_scale,
    vla3d_dir,
)

requires_vla3d = pytest.mark.skipif(
    not (vla3d_dir() / "scene_graph" / "generate_scene_info.py").is_file(),
    reason="VLA-3D is not checked out (scripts/setup_reifyscene.sh clones it)")

UNITS_PER_M = 0.1          # a scene one tenth of metric scale, as ours typically are


def _node(nid: str, category: str, boxes: dict[int, tuple[list, list]]) -> GraphNode:
    """``boxes``: frame -> (centre, full size) in METRES; stored in scene units as sigma extents."""
    pos = [{"frame_idx": f, "xyz": list(np.array(c) * UNITS_PER_M),
            "extent": list(np.array(sz) * UNITS_PER_M / 4)} for f, (c, sz) in boxes.items()]
    ntype = "human" if category == "person" else "object"
    return GraphNode(node_id=f"{ntype}:cam_00:sam2:cam_00:{nid}", node_type=ntype, label=category,
                     confidence=1.0, attrs={"category": category, "positions": pos,
                                            "frame_indices": sorted(boxes)})


def _graph(nodes, gravity=True) -> GraphArtifact:
    return GraphArtifact(run_id="t", stage_name="graph", save_dir="/tmp/x", nodes=nodes,
                         metadata={"world_frame": {"gravity_aligned": gravity}})


FRAMES = range(6)
TABLE = {f: ([0, 0, 0.75], [1.2, 0.8, 0.05]) for f in FRAMES}
CUP = {f: ([0.1, 0.0, 0.83], [0.08, 0.08, 0.1]) for f in FRAMES}
PERSON = {f: ([0.0, 3.0, 0.85], [0.5, 0.3, 1.7]) for f in FRAMES}


def test_corners_are_in_vla3ds_order_top_face_first() -> None:
    c = box_corners(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert c.shape == (8, 3)
    assert (c[:4, 2] == 3).all() and (c[4:, 2] == -3).all()
    assert c[0].tolist() == [-1, 2, 3] and c[2].tolist() == [1, -2, 3]


def test_scale_comes_from_people_when_present() -> None:
    tracks = entity_tracks([_node("1", "person", PERSON), _node("2", "cup", CUP)])
    units_per_m, meta = metric_scale(tracks)
    assert meta["source"] == "person_height"
    assert abs(units_per_m - UNITS_PER_M) < 1e-9


def test_intervals_break_where_an_endpoint_is_missing_and_drop_flicker() -> None:
    both = [0, 1, 2, 5, 6, 7, 9]                       # absent at 3-4 and 8
    held = [0, 1, 2, 5, 6, 7, 9]
    assert _intervals(held, both, min_frames=2) == [[0, 2], [5, 7]]    # 9 alone is flicker


LAMP = {f: ([0.0, 0.0, 1.6], [0.3, 0.3, 0.3]) for f in FRAMES}


@requires_vla3d
def test_touching_is_on_and_clearance_is_above() -> None:
    """VLA-3D's semantics, adopted as they are: a cup resting on a table is ``on`` it and not
    ``above`` it (above needs more than 1 cm of clearance); a lamp hanging clear of it is above."""
    edges, meta = compute_relations(_graph([_node("1", "person", PERSON),
                                            _node("2", "dining table", TABLE),
                                            _node("3", "cup", CUP), _node("4", "lamp", LAMP)]))
    rel = {(str(e.edge_type), e.src.rsplit(":", 1)[-1], e.dst.rsplit(":", 1)[-1]) for e in edges}
    assert ("on", "3", "2") in rel and ("above", "3", "2") not in rel
    assert ("above", "4", "2") in rel and ("below", "3", "4") in rel
    # their rule: supporting furniture (a table) is never reported as below something
    assert ("below", "2", "4") not in rel
    assert meta["metric_scale"]["source"] == "person_height"
    on = next(e for e in edges if str(e.edge_type) == "on")
    assert on.attrs["intervals"] == [[0, 5]] and on.confidence == 1.0


@requires_vla3d
def test_a_relation_ends_at_an_occlusion_gap() -> None:
    cup = {f: v for f, v in CUP.items() if f not in (2, 3)}          # cup unseen at 2-3
    edges, _ = compute_relations(_graph([_node("1", "person", PERSON),
                                         _node("2", "dining table", TABLE),
                                         _node("3", "cup", cup)]))
    on = next(e for e in edges if str(e.edge_type) == "on")
    assert on.attrs["intervals"] == [[0, 1], [4, 5]]


@requires_vla3d
def test_no_vertical_relations_without_gravity() -> None:
    edges, meta = compute_relations(_graph([_node("1", "person", PERSON),
                                            _node("2", "dining table", TABLE),
                                            _node("3", "cup", CUP)], gravity=False))
    assert not {str(e.edge_type) for e in edges} & {"above", "below", "on", "in"}
    assert "on" not in meta["predicates"]


@requires_vla3d
def test_near_is_one_edge_per_pair_and_add_relations_replaces_not_duplicates() -> None:
    g = _graph([_node("1", "person", PERSON), _node("2", "dining table", TABLE),
                _node("3", "cup", CUP)])
    add_relations(g)
    first = len(g.edges)
    add_relations(g)
    assert len(g.edges) == first
    near = [(e.src, e.dst) for e in g.edges if str(e.edge_type) == "near"]
    assert len(near) == len({tuple(sorted(p)) for p in near})
    assert g.metadata["relations"]["source"].startswith("VLA-3D")

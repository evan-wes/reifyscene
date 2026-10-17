"""Presence and the structured query (reifyscene/graph/presence.py, query/scene_query.py).

What these pin: presence gaps are reported and labelled but never end an entity; ``select``
matches an entity's category, not its runner-up guesses; ``relate`` says relations are missing
instead of inventing them; bad queries fail loudly.
"""

from __future__ import annotations

import pytest

from reifyscene.common.types import GraphEdge, GraphNode
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.presence import presence_from_frames
from reifyscene.query.scene_query import QueryError, SceneQuery


def test_presence_labels_short_gaps_dropouts_and_long_ones_occlusion_candidates() -> None:
    p = presence_from_frames([0, 1, 2, 4, 5, 9, 10, 10])
    assert (p.first, p.last) == (0, 10)
    assert [(g.start, g.end, g.kind) for g in p.gaps] == [
        (3, 3, "dropout"), (6, 8, "occlusion_candidate")]
    assert p.in_span(7) and not p.present_at(7) and p.present_at(9)


def test_presence_of_nothing_is_empty() -> None:
    p = presence_from_frames([])
    assert p.span is None and p.gaps == []


def _node(nid: str, ntype: str, category: str, frames: list[int], topk=None) -> GraphNode:
    return GraphNode(node_id=nid, node_type=ntype, label=category, confidence=0.9, attrs={
        "category": category, "frame_indices": frames,
        "category_topk": topk or [[category, 0.5]],
        "positions": [{"frame_idx": f, "xyz": [0.0, 0.0, float(f)]} for f in frames]})


def _graph() -> GraphArtifact:
    return GraphArtifact(run_id="t", stage_name="graph", save_dir="unsaved", nodes=[
        GraphNode(node_id="cam_00", node_type="camera", label="camera", confidence=1.0),
        _node("human:cam_00:sam2:cam_00:2", "human", "person", list(range(10)),
              topk=[["person", 0.3], ["toothbrush", 0.1]]),
        _node("human:cam_00:sam2:cam_00:3", "human", "person", [0, 1, 2, 7, 8, 9]),
        _node("object:cam_00:sam2:cam_00:5", "object", "cup", [4, 5]),
    ], edges=[GraphEdge(src="object:cam_00:sam2:cam_00:5", dst="cam_00", edge_type="observed_by",
                        confidence=1.0)])


def test_select_and_count_by_category_and_frame() -> None:
    q = SceneQuery(_graph())
    assert [e.node_id.rsplit(":", 1)[-1] for e in q.select("person")] == ["2", "3"]
    assert q.count("person", frame=5) == 1              # entity 3 is in a gap at 5
    assert q.count("cup") == 1 and q.count("Person") == 2
    assert q.categories() == {"person": 2, "cup": 1}


def test_select_does_not_match_runner_up_categories() -> None:
    assert SceneQuery(_graph()).select("toothbrush") == []


def test_when_reports_the_gap_and_entities_resolve_by_tracklet_number() -> None:
    q = SceneQuery(_graph())
    ans = q.run("when", {"entity": "3"})
    assert ans["entity"] == "human:cam_00:sam2:cam_00:3"
    assert ans["presence"]["gaps"][0] | {} == {"start": 3, "end": 6, "kind": "occlusion_candidate",
                                               "length": 4}


def test_relate_says_relations_are_missing_rather_than_answering() -> None:
    ans = SceneQuery(_graph()).run("relate", {"entity": "2", "predicate": "near"})
    assert ans["available"] is False and "relations" in ans["reason"]


def test_relate_reads_edges_both_ways_and_filters_by_frame() -> None:
    g = _graph()
    p2, p3, cup = ("human:cam_00:sam2:cam_00:2", "human:cam_00:sam2:cam_00:3",
                   "object:cam_00:sam2:cam_00:5")
    g.edges += [
        GraphEdge(src=p2, dst=p3, edge_type="near", confidence=0.5,
                  attrs={"intervals": [[0, 2], [7, 9]]}),
        GraphEdge(src=cup, dst=p2, edge_type="on", confidence=1.0, attrs={"intervals": [[4, 5]]}),
    ]
    g.metadata = {"relations": {"counts": {"near": 1, "on": 1}}}
    q = SceneQuery(g)
    ans = q.run("relate", {"entity": "3"})
    assert [r["sentence"] for r in ans["relations"]] == ["person 3 is near person 2"]
    assert q.run("relate", {"entity": "2", "predicate": "on"})["relations"][0]["sentence"] \
        == "cup 5 is on person 2"
    assert q.run("relate", {"entity": "2", "frame": 4})["count"] == 1      # only the cup then
    assert q.run("relate", {"entity": "2", "frame": 5, "predicate": "near"})["count"] == 0


def test_below_is_answered_from_the_stored_above_both_ways() -> None:
    g = _graph()
    p2, cup = "human:cam_00:sam2:cam_00:2", "object:cam_00:sam2:cam_00:5"
    g.edges.append(GraphEdge(src=cup, dst=p2, edge_type="above", confidence=1.0,
                             attrs={"intervals": [[4, 5]]}))
    g.metadata = {"relations": {"counts": {"above": 1}}}
    q = SceneQuery(g)
    below = q.run("relate", {"entity": "2", "predicate": "below"})["relations"]
    assert [r["sentence"] for r in below] == ["person 2 is below cup 5"]
    assert below[0]["stored_as"] == ["above", cup, p2]
    assert q.run("relate", {"entity": "5", "predicate": "above"})["relations"][0]["sentence"] \
        == "cup 5 is above person 2"
    assert q.run("relate", {"entity": "2", "predicate": "above"})["count"] == 0


def test_bad_queries_fail_loudly() -> None:
    q = SceneQuery(_graph())
    with pytest.raises(QueryError, match="unknown template"):
        q.run("describe", {})
    with pytest.raises(QueryError, match="no entity"):
        q.run("when", {"entity": "99"})
    with pytest.raises(QueryError):                      # camera is not an entity
        q.entity("cam_00")

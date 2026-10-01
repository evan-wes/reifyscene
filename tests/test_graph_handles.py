"""The graph's ``gaussians`` layer (reifyscene/graph/handles.py) and the layer rule.

What these pin: the layer adds its own attribute, file and metadata and changes nothing else in
the graph; its grade says how many entities got Gaussians; indices are handed out only for the
checkpoint they were computed against; the stage reports ``absent`` with a reason when there is
nothing to bind rather than failing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from reifyscene.api.run_outputs import Checkpoint, RunOutputs
from reifyscene.common.types import GraphEdge, GraphNode
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.handles import (
    LAYER,
    StaleHandles,
    layer_info,
    load_handles,
    run_handles,
    write_layer,
)
from tests.conftest import write_checkpoint, write_manifest

P, CUP = "human:cam_00:sam2:cam_00:2", "object:cam_00:sam2:cam_00:5"


def _graph(root: Path) -> GraphArtifact:
    write_manifest(root)
    g = GraphArtifact(run_id="t", stage_name="graph", save_dir=root / "graph", nodes=[
        GraphNode(node_id="cam_00", node_type="camera", label="camera", confidence=1.0),
        GraphNode(node_id=P, node_type="human", label="person", confidence=0.9,
                  attrs={"category": "person", "frame_indices": [0, 1]}),
        GraphNode(node_id=CUP, node_type="object", label="cup", confidence=0.8,
                  attrs={"category": "cup", "frame_indices": [1]}),
    ], edges=[GraphEdge(src=CUP, dst=P, edge_type="near", confidence=1.0,
                        attrs={"intervals": [[0, 1]]})],
        metadata={"capability": {"provides": {"renderable": "pending: gaussians layer"}}})
    g.save()
    return g


def _checkpoint(root: Path, payload: bytes = b"scene one", name: str = "render") -> Checkpoint:
    return write_checkpoint(root, payload, name=name)


def _handles(person_fg=(3, 4, 9), cup=()):
    return {P: (np.array(person_fg), np.array([1]), {"filter_dropped": {"outlier": 1}}),
            CUP: (np.array(cup, dtype=np.int64), np.array([], dtype=np.int64), {})}


def test_the_layer_adds_its_own_key_and_changes_nothing_else(tmp_path) -> None:
    g = _graph(tmp_path)
    before = copy.deepcopy((g.nodes, g.edges, g.metadata))
    grade = write_layer(g, _handles(), checkpoint=_checkpoint(tmp_path), num_gaussians=100,
                        slackness=0.0, filtered=True)
    assert grade == "degraded"                               # the cup got nothing
    again = GraphArtifact.load(tmp_path / "graph")
    person = next(n for n in again.nodes if n.node_id == P)
    assert person.attrs[LAYER] == {"status": "grounded", "fg": 3, "bg": 1,
                                   "assigned_fraction": 0.04, "filter_dropped": {"outlier": 1}}
    assert next(n for n in again.nodes if n.node_id == CUP).attrs[LAYER]["status"] \
        == "unassignable"
    assert LAYER not in next(n for n in again.nodes if n.node_id == "cam_00").attrs
    # every fact the graph stage wrote is exactly as it was
    nodes0, edges0, meta0 = before
    for n0, n1 in zip(nodes0, again.nodes, strict=True):
        assert {k: v for k, v in n1.attrs.items() if k != LAYER} == n0.attrs
    assert [e.attrs for e in again.edges] == [e.attrs for e in edges0]
    assert again.metadata["capability"] == meta0["capability"]
    info = layer_info(again)
    assert info["provides"] == {"renderable": "degraded"} and info["stage"] == "handles"
    assert (tmp_path / "graph" / info["file"]).is_file()


def test_every_entity_bound_is_grounded_and_a_rewrite_replaces_the_layer(tmp_path) -> None:
    g = _graph(tmp_path)
    ckpt = _checkpoint(tmp_path)
    write_layer(g, _handles(), checkpoint=ckpt, num_gaussians=100, slackness=0, filtered=True)
    assert write_layer(g, _handles(person_fg=(7,), cup=(2,)), checkpoint=ckpt,
                       num_gaussians=100, slackness=0, filtered=True) == "grounded"
    fg, bg = load_handles(RunOutputs.open(tmp_path), P, checkpoint=ckpt)
    assert fg.tolist() == [7] and bg.tolist() == [1]


def test_indices_are_refused_for_another_checkpoint(tmp_path) -> None:
    g = _graph(tmp_path)
    ckpt = _checkpoint(tmp_path)
    write_layer(g, _handles(), checkpoint=ckpt, num_gaussians=100, slackness=0, filtered=True)
    retrained = _checkpoint(tmp_path, b"scene two", name="render_retrained")
    run = RunOutputs.open(tmp_path)
    with pytest.raises(StaleHandles):
        load_handles(run, P, checkpoint=retrained)
    with pytest.raises(KeyError):
        load_handles(run, "object:cam_00:sam2:cam_00:99", checkpoint=ckpt)


def test_no_layer_is_a_key_error_not_empty_indices(tmp_path) -> None:
    _graph(tmp_path)
    with pytest.raises(KeyError):
        load_handles(RunOutputs.open(tmp_path), P, checkpoint=_checkpoint(tmp_path))


class _Render:
    def __init__(self, save_dir, kind="dynamic", checkpoint="trained_dynamic.npz"):
        self.save_dir, self.scene_state = save_dir, {"kind": kind, "checkpoint": checkpoint}


def test_the_stage_is_absent_with_a_reason_when_there_is_nothing_to_bind(tmp_path) -> None:
    g = _graph(tmp_path)
    run = RunOutputs.open(tmp_path)
    static = run_handles(g, _Render(tmp_path / "render", kind="static"), run=run, device="cpu")
    assert static["renderable"] == "absent" and "dynamic checkpoint" in static["reason"]
    empty = GraphArtifact(run_id="t", stage_name="graph", save_dir=tmp_path / "g2", nodes=[])
    out = run_handles(empty, _Render(tmp_path / "render"), run=run, device="cpu")
    assert out == {"renderable": "absent", "reason": "the graph has no entities"}
    assert layer_info(g) is None                               # nothing was written

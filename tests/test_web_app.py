"""The local web app's backend (reifyscene/api/web/app.py), against a fixture run, no GPU.

What these pin: runs are discovered only with a graph and a checkpoint; the graph endpoint
carries presence and says when there are no relations; queries go through SceneQuery and bad
ones return 400, not 500; an edit queues, runs on the worker and exposes its images; outlines
come from the id maps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reifyscene.api.web.app import create_app, discover_runs  # noqa: E402
from reifyscene.common.types import GraphNode  # noqa: E402
from reifyscene.graph.graph_artifact import GraphArtifact  # noqa: E402

H, W = 6, 8
PERSON = "human:cam_00:sam2:cam_00:2"
CUP = "object:cam_00:sam2:cam_00:5"


def _fixture_run(root: Path, name: str = "loop_demo") -> Path:
    run = root / name
    nodes = [
        GraphNode(node_id=PERSON, node_type="human", label="person", confidence=0.9,
                  attrs={"category": "person", "frame_indices": [0, 1, 2, 5]}),
        GraphNode(node_id=CUP, node_type="object", label="cup", confidence=0.8,
                  attrs={"category": "cup", "frame_indices": [1, 2]}),
    ]
    GraphArtifact(run_id=name, stage_name="graph", save_dir=run / "graph", nodes=nodes).save()
    views = []
    (run / "scene" / "train_views").mkdir(parents=True)
    for f in range(6):
        fn = f"train_views/{f:04d}.png"
        Image.fromarray(np.full((H, W, 3), 40 * f, np.uint8)).save(run / "scene" / fn)
        views.append({"frame_key": f"cam_00:{f}", "image_file": fn})
    (run / "scene" / "artifact.json").write_text(json.dumps({"train_views": views}))
    (run / "render").mkdir()
    (run / "render" / "trained_dynamic.npz").write_bytes(b"not read by the fake editor")
    ident = run / "identity"
    ident.mkdir()
    frames = []
    for f in range(6):
        m = np.zeros((H, W), np.uint16)
        m[1:4, 1:4] = 1
        Image.fromarray(m).save(ident / f"frame_{f:05d}.png")
        frames.append({"frame_idx": f, "file": f"frame_{f:05d}.png"})
    (ident / "manifest.json").write_text(json.dumps({
        "format_version": 1, "raster": {"width": W, "height": H}, "frames": frames,
        "classes": [{"index": 0, "node_id": None, "label": "background"},
                    {"index": 1, "node_id": PERSON, "label": "person"},
                    {"index": 2, "node_id": CUP, "label": "cup"}]}))
    return run


class FakeRenderer:
    time_of = {f: f for f in range(6)}


class FakeEditor:
    def __init__(self, run) -> None:
        self.run = run
        self.renderer = FakeRenderer()

    def render_before(self, frame: int) -> np.ndarray:
        return np.full((H, W, 3), 99, np.uint8)

    def render_view(self, frame: int, *, edit_dir=None, show="scene", **camera) -> np.ndarray:
        self.last_view = (frame, edit_dir, show, camera)
        return np.full((H, W, 3), 7, np.uint8)

    def remove(self, node_id, *, slackness, filtered, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("before", "after", "took"):
            Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(out_dir / f"{kind}_00001.jpg")
        return {"node_id": node_id, "removed_fg": 3, "removed_bg": 1, "assigned_fraction": 0.1,
                "renderable": "grounded", "frames": [1], "slackness": slackness,
                "filtered": filtered, "seconds": 0.0}


@pytest.fixture()
def client(tmp_path):
    _fixture_run(tmp_path)
    (tmp_path / "loop_nograph").mkdir()                       # no graph: not a run
    return TestClient(create_app(tmp_path, editor_factory=FakeEditor))


def test_only_runs_with_a_graph_and_a_checkpoint_are_listed(client, tmp_path) -> None:
    runs = client.get("/api/runs").json()
    assert [r["name"] for r in runs] == ["loop_demo"]
    assert runs[0]["entities"] == 2 and runs[0]["frames"] == 6
    assert list(discover_runs(tmp_path, ["render_sharp", "render"])) == ["loop_demo"]


def test_graph_carries_presence_and_admits_no_relations(client) -> None:
    g = client.get("/api/runs/loop_demo/graph").json()
    person = next(e for e in g["entities"] if e["node_id"] == PERSON)
    assert person["gaps"] == [{"start": 3, "end": 4, "kind": "dropout"}]
    assert g["relations_available"] is False and g["edges"] == []
    assert g["gaussians_layer"] is None and person["gaussians"] is None
    assert g["frames"] == [0, 1, 2, 3, 4, 5]


def test_the_gaussians_layer_is_shown_only_for_the_checkpoint_it_describes(tmp_path) -> None:
    from reifyscene.graph.handles import write_layer

    run = _fixture_run(tmp_path)
    graph = GraphArtifact.load(run / "graph")
    handles = {PERSON: (np.array([1, 2]), np.array([0]), {}),
               CUP: (np.array([], np.int64), np.array([], np.int64), {})}
    write_layer(graph, handles, checkpoint_path=run / "render" / "trained_dynamic.npz",
                num_gaussians=10, slackness=0.0, filtered=True)
    g = TestClient(create_app(tmp_path, editor_factory=FakeEditor)).get(
        "/api/runs/loop_demo/graph").json()
    assert g["gaussians_layer"]["current"] is True and g["gaussians_layer"]["renderable"] == \
        "degraded"
    by = {e["node_id"]: e["gaussians"] for e in g["entities"]}
    assert by[PERSON]["status"] == "grounded" and by[CUP]["status"] == "unassignable"
    # retrained: the same layer now describes another scene, and the app will not use it
    (run / "render" / "trained_dynamic.npz").write_bytes(b"a retrained scene")
    g = TestClient(create_app(tmp_path, editor_factory=FakeEditor)).get(
        "/api/runs/loop_demo/graph").json()
    assert g["gaussians_layer"]["current"] is False
    assert all(e["gaussians"] is None for e in g["entities"])


def test_queries_go_through_scene_query_and_bad_ones_are_400(client) -> None:
    r = client.post("/api/runs/loop_demo/query", json={"template": "select",
                                                       "args": {"category": "person"}})
    assert r.status_code == 200 and r.json()["count"] == 1
    r = client.post("/api/runs/loop_demo/query", json={"template": "count",
                                                       "args": {"category": "cup", "frame": 5}})
    assert r.json()["count"] == 0
    assert client.post("/api/runs/loop_demo/query",
                       json={"template": "nope", "args": {}}).status_code == 400
    assert client.post("/api/runs/loop_demo/query",
                       json={"template": "when", "args": {"entity": "77"}}).status_code == 400


def test_frames_renders_and_outlines_are_served(client) -> None:
    assert client.get("/api/runs/loop_demo/frames/2.jpg").status_code == 200
    assert client.get("/api/runs/loop_demo/frames/9.jpg").status_code == 404
    assert client.get("/api/runs/loop_demo/render/2.jpg").status_code == 200
    r = client.get(f"/api/runs/loop_demo/outline/2.png?node={PERSON}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/runs/loop_demo/outline/2.png?node=nobody").status_code == 404
    two = client.get(f"/api/runs/loop_demo/outline/2.png?node={PERSON}&node={CUP}")
    assert two.status_code == 200      # a relation: both entities, one image


def test_an_edit_queues_runs_and_exposes_its_images(client) -> None:
    r = client.post("/api/runs/loop_demo/edits", json={"entity": "2", "slackness": 0.4})
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(50):
        st = client.get(f"/api/edits/{jid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert st["status"] == "done", st
    assert st["result"]["node_id"] == PERSON and st["params"]["slackness"] == 0.4
    assert client.get(f"/api/edits/{jid}/after/1.jpg").status_code == 200
    assert client.get(f"/api/edits/{jid}/after/9.jpg").status_code == 404


def test_the_3d_view_passes_the_camera_and_the_edit_through(client) -> None:
    r = client.get("/api/runs/loop_demo/view/2.jpg?yaw=30&pitch=-10&zoom=0.5&px=0.1&width=320")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/runs/loop_demo/view/2.jpg?pitch=120").status_code == 422
    assert client.get("/api/runs/loop_demo/view/2.jpg?show=all").status_code == 422
    assert client.get("/api/runs/loop_demo/view/2.jpg?show=took").status_code == 400  # no edit
    assert client.get("/api/runs/loop_demo/view/2.jpg?show=edited&edit=nope").status_code == 404
    jid = client.post("/api/runs/loop_demo/edits", json={"entity": "2"}).json()["id"]
    for _ in range(50):
        if client.get(f"/api/edits/{jid}").json()["status"] == "done":
            break
        time.sleep(0.05)
    for show in ("scene", "edited", "took"):
        r = client.get(f"/api/runs/loop_demo/view/2.jpg?edit={jid}&show={show}&yaw=90")
        assert r.status_code == 200, show


def test_edits_refuse_unknown_ops_and_entities(client) -> None:
    assert client.post("/api/runs/loop_demo/edits",
                       json={"op": "move", "entity": "2"}).status_code == 400
    assert client.post("/api/runs/loop_demo/edits", json={"entity": "77"}).status_code == 400


def test_the_page_is_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "reifyscene" in r.text

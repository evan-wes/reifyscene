"""RunOutputs (reifyscene/api/run_outputs.py): what a run folder holds, found by what is in it.

What these pin: a folder is a run because it has a manifest, never because of its name; a
stage's artifact and record come from the stage registry's folder; a checkpoint is whatever
folder holds a render record naming its weights, and those weights and their kind are what the
record says -- a stage folder is never mistaken for one.
"""

from __future__ import annotations

import pytest

from reifyscene.api.run_outputs import NotARun, RunOutputs
from reifyscene.common.types import GraphNode
from reifyscene.graph.graph_artifact import GraphArtifact
from tests.conftest import write_checkpoint, write_ingest, write_manifest


def _run(root, name="anything"):
    run = root / name
    write_ingest(run, num_frames=3, width=8, height=6)          # writes the manifest first
    GraphArtifact(run_id=name, stage_name="graph", save_dir=run / "graph",
                  nodes=[GraphNode(node_id="n", node_type="object", label="cup",
                                   confidence=1.0)]).save()
    return run


def test_a_run_is_recognised_by_its_manifest_not_its_name(tmp_path) -> None:
    _run(tmp_path, "whatever_it_is_called")
    (tmp_path / "loop_lookalike" / "graph").mkdir(parents=True)   # a run's shape, no manifest
    write_manifest(tmp_path / "empty_run")
    assert [r.name for r in RunOutputs.discover(tmp_path)] == ["empty_run",
                                                               "whatever_it_is_called"]
    with pytest.raises(NotARun, match="no run_manifest.json"):
        RunOutputs.open(tmp_path / "loop_lookalike")


def test_stages_load_through_the_registry_and_records_need_no_arrays(tmp_path) -> None:
    run = RunOutputs.open(_run(tmp_path))
    assert run.has("graph") and not run.has("render")
    assert run.artifact("graph").nodes[0].label == "cup"
    assert run.record("graph")["stage_name"] == "graph"
    assert run.stage_dir("graph") == run.root / "graph"
    with pytest.raises(NotARun, match="no render artifact"):
        run.artifact("render")
    with pytest.raises(KeyError, match="no stage"):
        run.stage_dir("not_a_stage")


def test_a_checkpoint_is_what_its_render_record_says(tmp_path) -> None:
    run = RunOutputs.open(_run(tmp_path))
    assert not run.checkpoint().exists() and run.checkpoints() == []
    write_checkpoint(run.root, kind="static", weights="trained_gaussians.npz")
    write_checkpoint(run.root, b"retrained", name="render_retrain")
    ckpt = run.checkpoint()
    assert ckpt.exists() and ckpt.kind == "static" and ckpt.weights.name == "trained_gaussians.npz"
    # the render stage's own first; the graph folder (a record, no scene_state) is not one
    assert [c.name for c in run.checkpoints()] == ["render", "render_retrain"]
    assert run.checkpoint("render_retrain").digest() != ckpt.digest()


def test_a_checkpoint_says_how_it_was_trained_or_refuses(tmp_path) -> None:
    run = RunOutputs.open(_run(tmp_path))
    recipe = {"trainer": {"views_per_step": 8}, "densify": {}, "identity": None, "scene": "ab12"}
    write_checkpoint(run.root, recipe=recipe)
    write_checkpoint(run.root, name="render_unrecorded")
    assert run.checkpoint().recipe == recipe
    with pytest.raises(NotARun, match="records no recipe"):
        _ = run.checkpoint("render_unrecorded").recipe


def test_what_accumulates_beside_the_stages_has_its_place(tmp_path) -> None:
    run = RunOutputs.open(_run(tmp_path))
    assert run.output_dir("heldout_damage", "masks") == run.root / "edits/heldout_damage/masks"
    assert run.identity_dir.parent == run.root
    assert run.streams.only().id == "cam_00"
    assert run.config == {}

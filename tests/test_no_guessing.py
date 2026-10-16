"""Guards: what the pipeline used to guess stays owned by the one module that knows it.

Each rule names its owner. Code elsewhere asks the owner -- ``RunOutputs`` for a run's layout,
``FrameStream`` / ``VideoStreamRef`` for a stream's frames, ``frame_keys`` / ``node_ids`` for ids,
the dataset classes for a dataset's folders -- instead of writing the name, path or number
itself. These scan the code (its syntax tree, so a docstring or comment that mentions a name is
not a use) of the ``reifyscene`` package and ``scripts/``.

A file allowed to break a rule is listed with the reason. Adding one needs a reason as good.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VENDORED = ("reifyscene/tracking/tapnet/",)
STAGES = {"ingest", "sync", "dyn_mask", "geometry", "pose_refine", "segment", "tracking", "reid",
          "semantics", "objects", "humans", "graph", "optimization", "scene", "render",
          "handles", "identity", "edits", "calibration"}
RUN_FILES = {"artifact.json", "run_manifest.json", "trained_dynamic.npz", "trained_gaussians.npz"}


def _sources() -> list[tuple[str, ast.AST]]:
    out = []
    for path in sorted([*REPO.glob("reifyscene/**/*.py"), *REPO.glob("scripts/*.py")]):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(VENDORED):
            continue
        out.append((rel, ast.parse(path.read_text(), filename=rel)))
    return out


SOURCES = _sources()


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _strings(tree: ast.AST):
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node


def _violations(rule) -> list[str]:
    return [f"{rel}:{line}: {what}" for rel, tree in SOURCES for line, what in rule(tree)]


def _check(rule, allowed: dict[str, str]) -> None:
    found = [v for v in _violations(rule) if v.split(":", 1)[0] not in allowed]
    assert not found, "\n".join(found)


# -- a run's layout: RunOutputs ---------------------------------------------------------------

def _stage_joins(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant) and node.right.value in STAGES):
            yield node.lineno, f"joins a path with {node.right.value!r}"


def _run_file_names(tree):
    for node in _strings(tree):
        if node.value in RUN_FILES:
            yield node.lineno, f"names {node.value!r}"


def test_only_run_outputs_joins_a_run_folder_with_a_stage_name() -> None:
    _check(_stage_joins, {
        "reifyscene/api/run_outputs.py": "the owner",
        "scripts/f0_validate_windmill.py": "builds its own scratch folder, not a run",
        "scripts/run_calibration_reifyscene.py": "the repo's configs/calibration/, not a run",
    })


def test_only_their_owners_name_a_runs_files() -> None:
    _check(_run_file_names, {
        "reifyscene/api/run_outputs.py": "the owner of the manifest",
        "reifyscene/common/types.py": "BaseArtifact.FILE",
        "reifyscene/render/gaussian4d/render_backend.py": "writes the weights and records "
                                                          "their name in the render record",
    })


# -- streams and ids --------------------------------------------------------------------------

def _named(value: str):
    def rule(tree):
        for node in _strings(tree):
            if node.value == value:
                yield node.lineno, f"names {value!r}"
    return rule


def test_no_stream_is_assumed_to_be_cam_00() -> None:
    synthetic = "builds a synthetic stream it names itself"
    _check(_named("cam_00"), {
        "scripts/f0_validate_windmill.py": synthetic,
        "scripts/fp3_validate_reid_merge.py": synthetic,
        "scripts/fp4_validate_semantics.py": synthetic,
        "scripts/fp5_validate_humans.py": synthetic,
    })


def test_nothing_guesses_a_render_raster() -> None:
    def rule(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "replay_size" or \
                    isinstance(node, ast.Attribute) and node.attr == "replay_size" or \
                    isinstance(node, ast.FunctionDef) and node.name == "replay_size":
                yield node.lineno, "replay_size (the raster is recorded: train_raster_wh)"
    _check(rule, {})


def test_frame_numbers_come_from_the_stream_not_a_count() -> None:
    def rule(tree):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "range"
                    and any(isinstance(n, ast.Attribute) and n.attr == "num_frames"
                            for a in node.args for n in ast.walk(a))):
                yield node.lineno, "range(... .num_frames): use the stream's frame_numbers"
    _check(rule, {
        "reifyscene/common/types.py": "the owner (VideoStreamRef.frame_numbers of a video)",
        "reifyscene/tracking/all_pairs.py": "positions in a stacked clip, not frame numbers",
    })


def test_ids_are_split_only_by_their_owners() -> None:
    def rule(tree):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("split", "rsplit") and node.args
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == ":"):
                yield node.lineno, f".{node.func.attr}(':'): use frame_keys / node_ids"
    proc_text = "parses 'key: value' lines of /proc and nvidia-smi output, not ids"
    _check(rule, {
        "reifyscene/common/frame_keys.py": "the owner of frame keys",
        "reifyscene/graph/node_ids.py": "the owner of node ids",
        "reifyscene/common/memory.py": proc_text,
        "reifyscene/common/system_report.py": proc_text,
    })


# -- datasets: their own classes --------------------------------------------------------------

@pytest.mark.parametrize("name", ["JPEGImages", "Annotations", "480p", "extra.json"])
def test_only_the_dataset_classes_know_a_datasets_folders(name: str) -> None:
    _check(_named(name), {
        "reifyscene/ingest/datasets.py": "the owner",
        "scripts/run_reference_benchmark.py": "checks a folder holds what the Shape-of-Motion "
                                              "reference's own loader requires",
    })


# -- the guards themselves --------------------------------------------------------------------

def test_the_guards_see_a_violation() -> None:
    tree = ast.parse('p = run / "graph"\nq = "artifact.json"\nk.rsplit(":", 1)\n'
                     'range(stream.num_frames)\ns = "cam_00"\n')
    assert [w for _, w in _stage_joins(tree)] == ["joins a path with 'graph'"]
    assert [w for _, w in _run_file_names(tree)] == ["names 'artifact.json'"]
    assert list(_named("cam_00")(tree))
    doc = ast.parse('"""run / "graph" and artifact.json in a docstring are not uses."""\n')
    assert not list(_stage_joins(doc)) and not list(_run_file_names(doc))

"""Every setting a stage reads is in its contract's fingerprint (reifyscene/api/contracts.py).

A resume accepts a loaded stage when its fingerprint matches, so a setting a stage
reads but does not fingerprint is one a resume would silently ignore. This runs the
whole pipeline (CPU backends) on a config that records every key read from it, credits
each read to the stage that stamps its fingerprint next, and checks it is declared.
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path

from reifyscene.api.contracts import contract_for
from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.api.stages import STAGE_ORDER
from tests.conftest import write_test_video

# Not settings: where the run lives and what it reads (the input is fingerprinted).
NOT_SETTINGS = {"run_dir", "run_id", "_resume_video_paths", "_profile_picked"}


class RecordingDict(dict):
    """A config mapping that logs the dotted path of every leaf read from it."""

    def __init__(self, data, log, prefix=""):
        super().__init__()
        self._log, self._prefix = log, prefix
        for key, value in data.items():
            if isinstance(value, dict):
                value = RecordingDict(value, log, self._path(key))
            dict.__setitem__(self, key, value)

    def _path(self, key):
        return f"{self._prefix}.{key}" if self._prefix else str(key)

    def _read(self, key, value):
        if not isinstance(value, dict):  # a section object is not a setting; its leaves are
            self._log.append(self._path(key))

    def _read_all(self):
        self._log.append(self._path("*"))

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        self._read(key, value)
        return value

    def get(self, key, default=None):
        value = dict.get(self, key, default)
        self._read(key, value)
        return value

    def __contains__(self, key):
        self._read(key, dict.get(self, key))
        return dict.__contains__(self, key)

    def __iter__(self):
        self._read_all()
        return dict.__iter__(self)

    def keys(self):
        self._read_all()
        return dict.keys(self)

    def items(self):
        self._read_all()
        return dict.items(self)

    def values(self):
        self._read_all()
        return dict.values(self)

    def setdefault(self, key, default=None):
        value = dict.setdefault(self, key, default)
        self._read(key, value)
        return value

    def pop(self, key, *default):
        value = dict.pop(self, key, *default)
        self._read(key, value)
        return value

    def __deepcopy__(self, memo):
        self._read_all()
        return copy.deepcopy(dict(dict.items(self)), memo)


def _declared(stage: str, path: str, cfg: dict) -> bool:
    contract = contract_for(stage)
    if path.endswith(".*"):
        section = path[:-2]
        # Reading a whole section is declared when the section (or a parent) is a subtree entry.
        return any(section == e or section.startswith(e + ".") for e in contract.settings)
    for entry in contract.settings:
        if path == entry or path.startswith(entry + "."):
            return True
    return any(fnmatch.fnmatchcase(path, p) for p in contract.excluded(cfg))


def test_every_setting_a_stage_reads_is_in_its_fingerprint(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    write_test_video(video, num_frames=6)
    cfg = {
        "run_dir": str(tmp_path / "run"),
        "geometry": {"backend_mode": "heuristic"},
        "tracking": {"max_frames": 4},
        "lifting": {"point_conf_threshold": 0.0},
        "render": {"device": "cpu"},
    }
    log: list[str] = []
    pipeline = ReifyScenePipeline(RecordingDict(copy.deepcopy(cfg), log))
    construction = list(log)
    log.clear()

    by_stage: dict[str, list[str]] = {}
    stamp = pipeline._stamp_stage_metadata

    def attributing_stamp(artifact, stage_name, *, video_paths):
        by_stage.setdefault(stage_name, []).extend(log)
        log.clear()
        out = stamp(artifact, stage_name, video_paths=video_paths)
        log.clear()  # the fingerprint's own reads of the config are not the stage's
        return out

    pipeline._stamp_stage_metadata = attributing_stamp
    registry = pipeline.stage_registry()
    outputs: dict = {}
    for stage in STAGE_ORDER:
        outputs[stage] = registry[stage].run_fn(pipeline, outputs, [str(video)])
        if stage == "tracking":
            outputs["tracks"] = outputs[stage]

    undeclared = {
        stage: sorted({p for p in paths if p.split(".")[0] not in NOT_SETTINGS and not _declared(stage, p, cfg)})
        for stage, paths in by_stage.items()
    }
    assert {s: p for s, p in undeclared.items() if p} == {}
    # Settings read while the pipeline is built belong to whichever stage uses them.
    orphans = sorted(
        {
            p for p in construction
            if p.split(".")[0] not in NOT_SETTINGS and not any(_declared(s, p, cfg) for s in STAGE_ORDER)
        }
    )
    assert orphans == []

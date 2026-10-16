"""Resume: load a prior run's earlier stages only when they are what this run would make.

``--start-stage X --prior-run-dir P`` loads every stage before X from P instead of
re-running it. :func:`check_resume` decides for each of those stages, read-only and
before anything is copied or run, whether it can be loaded: its fingerprint
(:mod:`reifyscene.api.contracts` -- the settings it uses, the input frames, its
contract version) must match this invocation, and it must have been built from the
upstream artifacts it is being loaded with. Anything else rejects the resume, naming
the stage to restart from; one report lists every problem and the latest stage the
prior run can be resumed from.

Settings the run script picks from the machine are pinned to the prior run's values
instead of re-picked, so a busy GPU does not reject a resume.

There is no partial credit: an artifact is loaded exactly as it is, or the stage is
stale. A stale stage that is cheap and deterministic (``rebuildable``) is re-run;
anything else rejects the resume. Nothing is upgraded, relinked, re-rooted or derived.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.api.contracts import (
    MISSING,
    contract_for,
    effective_settings,
    fingerprint,
    get_path,
    input_identity,
    set_path,
)
from reifyscene.api.requirements import REQUIREMENTS, ResumeContext
from reifyscene.api.run_outputs import RunOutputs
from reifyscene.api.stages import STAGE_ORDER, StageSpec
from reifyscene.common.io import read_json, write_json
from reifyscene.common.types import BaseArtifact


class ResumeRejected(ValueError):
    """The prior run cannot be resumed as asked; the message is the full report."""


# ------------------------------------------------------------------ stage window

def ensure_start_stage(stage_name: str) -> None:
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{stage_name}'. Expected one of: {', '.join(STAGE_ORDER)}")


def ensure_stop_stage(stage_name: str) -> None:
    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{stage_name}'. Expected one of: {', '.join(STAGE_ORDER)}")


def resolve_stage_window(start_stage: str, stop_stage: str) -> tuple[int, int]:
    """Resolve and validate an inclusive ``[start, stop]`` stage window.

    Returns the ``(start_idx, stop_idx)`` pair into ``STAGE_ORDER``. Stages
    before ``start`` are loaded from a prior run; stages in ``[start, stop]``
    run; stages after ``stop`` are skipped. Raises ``ValueError`` if either
    name is unknown or if ``stop_stage`` precedes ``start_stage``.
    """
    ensure_start_stage(start_stage)
    ensure_stop_stage(stop_stage)
    start_idx = STAGE_ORDER.index(start_stage)
    stop_idx = STAGE_ORDER.index(stop_stage)
    if stop_idx < start_idx:
        raise ValueError(
            f"--stop-stage '{stop_stage}' precedes --start-stage '{start_stage}'; "
            f"the stage window must run forward through: {', '.join(STAGE_ORDER)}."
        )
    return start_idx, stop_idx


# ------------------------------------------------------------------ the plan

@dataclass
class StageVerdict:
    """What a resume does with one prior stage, and why."""

    stage: str
    action: str = "load"  # "load" | "rebuild" (re-run, it is cheap) | "reject"
    notes: list[str] = field(default_factory=list)  # why a stage is rebuilt
    problems: list[str] = field(default_factory=list)
    restart_from: str | None = None

    def reject(self, problem: str, restart_from: str | None = None) -> None:
        self.action = "reject"
        self.problems.append(problem)
        self.restart_from = restart_from or self.restart_from or self.stage


@dataclass
class ResumePlan:
    """The outcome of :func:`check_resume`: a verdict per checked prior stage."""

    prior_run_dir: Path
    run_dir: Path
    start_stage: str
    verdicts: dict[str, StageVerdict]
    records: dict[str, dict[str, Any]]
    artifacts: dict[str, Any]
    pins: dict[str, Any]

    @property
    def loads(self) -> list[str]:
        """The stages this resume loads: every stage before the start stage."""
        return list(STAGE_ORDER[: STAGE_ORDER.index(self.start_stage)])

    @property
    def ok(self) -> bool:
        return all(
            s in self.verdicts and self.verdicts[s].action != "reject" for s in self.loads
        )

    def latest_start(self) -> str:
        """The latest stage this prior run can be resumed from, given what was checked."""
        restarts = [
            STAGE_ORDER.index(v.restart_from or v.stage)
            for v in self.verdicts.values()
            if v.action == "reject"
        ]
        if restarts:
            return STAGE_ORDER[min(restarts)]
        checked = [STAGE_ORDER.index(s) for s in self.verdicts]
        nxt = max(checked) + 1 if checked else 0
        return STAGE_ORDER[min(nxt, len(STAGE_ORDER) - 1)]

    def report(self) -> str:
        lines = [
            f"resume check: {self.prior_run_dir} -> {self.run_dir}, start at {self.start_stage}"
        ]
        for stage in STAGE_ORDER:
            verdict = self.verdicts.get(stage)
            if verdict is None:
                continue
            mark = {"load": "ok", "rebuild": "rebuilt", "reject": "REJECTED"}[verdict.action]
            lines.append(f"  {stage:13s} {mark}")
            lines.extend(f"      problem: {p}" for p in verdict.problems)
            lines.extend(f"      {n}" for n in verdict.notes)
        for path, value in sorted(self.pins.items()):
            lines.append(f"  pinned from the prior run: {path} = {json.dumps(value)}")
        latest = self.latest_start()
        if self.ok:
            beyond = set(self.verdicts) - set(self.loads)
            later = f" (this prior run supports starting as late as {latest})" if beyond else ""
            lines.append(f"=> resumable at {self.start_stage}{later}")
        else:
            lines.append(
                f"=> NOT resumable at {self.start_stage}; the latest stage it can be resumed "
                f"from is {latest}"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------ the check

def _settings_diff(recorded: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    def flat(value: Any, prefix: str, out: dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict) and value:
            for k, v in value.items():
                flat(v, f"{prefix}.{k}", out)
        else:
            out[prefix] = value
        return out

    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    for path, value in recorded.items():
        flat(value, path, a)
    for path, value in expected.items():
        flat(value, path, b)
    diffs = []
    for key in sorted(set(a) | set(b)):
        if a.get(key, MISSING) != b.get(key, MISSING):
            was = "absent" if key not in a else json.dumps(a[key])
            now = "absent" if key not in b else json.dumps(b[key])
            diffs.append(f"{key}: prior {was}, now {now}")
    return diffs


_ABSENT = "<absent>"


def _prior_state(stage: str, record: dict[str, Any], path: str) -> tuple[str, Any] | None:
    """What the prior run's config held at ``path``, as ``stage`` recorded it.

    ``("value", v)``, or ``("absent", default)`` when the key was not set (a subtree
    entry lacks it; a leaf entry holds the code default ``default``), or ``None``
    when this stage does not record the path at all.
    """
    contract = contract_for(stage)
    settings = record.get("settings") or {}
    if path in settings:
        default = contract.settings.get(path, MISSING)
        if default is MISSING:
            return ("value", settings[path])
        return ("leaf", (settings[path], default))
    for entry, value in settings.items():
        if path.startswith(entry + ".") and isinstance(value, dict):
            found = get_path(value, path[len(entry) + 1:], MISSING)
            return ("absent", None) if found is MISSING else ("value", found)
    return None


def _pin(records: dict[str, dict[str, Any]], verdicts: dict[str, StageVerdict],
         cfg: dict[str, Any]) -> dict[str, Any]:
    """Take machine-picked settings from the prior run; returns what was pinned.

    A machine-picked setting (``cfg["_profile_picked"]``) or one this invocation does
    not set at all is set to what the prior run's config held -- including "not set",
    so a stage that fingerprints the whole section and one that fingerprints the key
    with its default both reproduce. A setting the user gave explicitly is left alone
    (a difference then rejects: that is intent, not a machine pick).
    """
    picked = set(cfg.get("_profile_picked") or [])
    states: dict[str, dict[str, tuple[str, Any]]] = {}
    for stage, record in records.items():
        for path in contract_for(stage).pinned:
            state = _prior_state(stage, record, path)
            if state is not None:
                states.setdefault(path, {})[stage] = state
    pins: dict[str, Any] = {}
    for path, by_stage in states.items():
        unset = any(kind == "absent" for kind, _ in by_stage.values())
        held: set[str] = set()  # the values the prior run's config held at ``path``
        for kind, payload in by_stage.values():
            if kind == "value":
                held.add(json.dumps(payload, sort_keys=True))
            elif kind == "leaf":
                value, default = payload
                if not (unset and value == default):  # a leaf at its default agrees with "not set"
                    held.add(json.dumps(value, sort_keys=True))
        if len(held) > 1 or (unset and held):
            for stage in by_stage:
                verdicts[stage].reject(
                    f"the prior stages disagree on {path}: "
                    + ", ".join(f"{s}={json.dumps(v)}" for s, v in by_stage.items())
                )
            continue
        current = get_path(cfg, path, MISSING)
        if current is not MISSING and path not in picked:
            continue  # the user's own value: intent, not a machine pick
        if not held:  # the prior run did not set it
            if current is not MISSING:
                _del_path(cfg, path)
                pins[path] = None
            continue
        prior_value = json.loads(held.pop())
        if current is MISSING:
            leaves = [payload for kind, payload in by_stage.values() if kind == "leaf"]
            if len(leaves) == len(by_stage) and all(d == prior_value for _, d in leaves):
                continue  # unset now, the code default then: the same effective value
        elif current == prior_value:
            continue
        set_path(cfg, path, prior_value)
        pins[path] = prior_value
    return pins


def _del_path(cfg: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def check_resume(
    prior_run_dir: str | Path,
    run_dir: str | Path,
    start_stage: str,
    cfg: dict[str, Any],
    video_paths: list[str],
    registry: dict[str, StageSpec],
    *,
    stages: list[str] | None = None,
) -> ResumePlan:
    """Decide, read-only, what a resume does with each prior stage.

    Checks every stage before ``start_stage`` (or ``stages``, e.g. every stage the
    prior run has, for a report), in order. A stage is loaded when its fingerprint
    matches, its upstream is the one it was built from and it meets its
    requirements (:mod:`reifyscene.api.requirements`) -- after deriving whatever can
    be derived from what is loaded. A stale rebuildable stage is re-run instead; a
    rebuilt or upgraded stage makes the stages built from it stale in turn. Anything
    else rejects. Mutates ``cfg`` to pin machine-picked settings to the prior run's
    values. Raises :class:`ResumeRejected` only for a request that cannot be checked
    at all (the prior run and the new run are the same folder).
    """
    prior = Path(prior_run_dir).expanduser().resolve()
    prior_outputs = RunOutputs.open(prior)
    run = Path(run_dir).expanduser().resolve()
    if prior == run:
        raise ResumeRejected(
            f"--prior-run-dir and --run-dir are the same folder ({prior}): a resume copies the "
            "prior stages into the run folder and would overwrite the stages it is loading. "
            "Resume into a new --run-dir."
        )
    ensure_start_stage(start_stage)
    if stages is None:
        stages = list(STAGE_ORDER[: STAGE_ORDER.index(start_stage)])
    stages = [s for s in STAGE_ORDER if s in set(stages)]
    video_paths = [str(p) for p in video_paths]
    current_input = input_identity(video_paths, frames_camera=cfg.get("frames_camera"))

    verdicts: dict[str, StageVerdict] = {}
    records: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for stage in stages:
        verdict = verdicts[stage] = StageVerdict(stage)
        stage_dir = prior_outputs.stage_dir(stage)
        rebuildable = contract_for(stage).is_rebuildable(cfg)
        if not stage_dir.exists():
            if rebuildable:
                verdict.action = "rebuild"
                verdict.notes.append("rebuilt: the prior run does not have it")
            else:
                verdict.reject(f"the prior run has no {stage} stage ({stage_dir})")
            continue
        try:
            artifact = registry[stage].load_fn(stage_dir)
        except Exception as exc:  # an incomplete or unreadable artifact cannot be loaded
            why = f"{type(exc).__name__}: {exc}"
            if rebuildable:
                verdict.action = "rebuild"
                verdict.notes.append(f"rebuilt: it cannot be loaded ({why})")
            else:
                verdict.reject(f"cannot be loaded: {why}")
            continue
        record = (getattr(artifact, "metadata", {}) or {}).get("contract")
        if record is None:
            verdict.reject("it carries no fingerprint")
            continue
        artifacts[stage] = artifact
        records[stage] = record

    pins = _pin(records, verdicts, cfg)

    # What each stage's digest will be once the plan is carried out (lineage for the next).
    digests: dict[str, str | None] = {}
    for stage in stages:
        verdict = verdicts[stage]
        if verdict.action == "load":
            _check_loaded(stage, verdict, records, artifacts, digests, cfg, current_input)
        if verdict.action == "rebuild":
            upstream = {u: digests.get(u) for u in contract_for(stage).upstream}
            records[stage] = fingerprint(stage, effective_settings(stage, cfg), current_input,
                                         upstream)
        digests[stage] = records[stage]["digest"] if verdict.action != "reject" else None

    return ResumePlan(prior, run, start_stage, verdicts, records, artifacts, pins)


def _check_loaded(stage, verdict, records, artifacts, digests, cfg, current_input) -> None:
    """Load one stage as it is -- or mark it for rebuild, or reject it.

    It loads only when every part of its fingerprint matches this invocation: input,
    contract version, settings, and the digest of every upstream it is loaded with (an
    upstream it does not record counts as changed). Then its requirements must hold.
    """
    contract = contract_for(stage)
    record = records[stage]
    if record.get("input") != current_input:
        verdict.reject("it was made from a different input (the frames it read have changed)")
        return
    stale: list[str] = []
    if record.get("version") != contract.version:
        stale.append(f"made by contract version {record.get('version')} of {stage}; this code "
                     f"is version {contract.version}")
    diffs = _settings_diff(record.get("settings") or {}, effective_settings(stage, cfg))
    if diffs:
        stale.append("settings differ: " + "; ".join(diffs))
    recorded_upstream = record.get("upstream") or {}
    stale.extend(
        f"it was built from a different {up} artifact than the one being loaded"
        for up in contract.upstream
        if digests.get(up) is None or recorded_upstream.get(up) != digests[up]
    )
    if stale:
        if contract.is_rebuildable(cfg):
            verdict.action = "rebuild"
            verdict.notes.append("rebuilt: " + "; ".join(stale))
        else:
            for problem in stale:
                verdict.reject(problem)
        return
    ctx = ResumeContext(stage, artifacts[stage], artifacts, cfg)
    for requirement in REQUIREMENTS.get(stage, ()):
        problem = requirement.check(ctx)
        if problem is not None:
            verdict.reject(problem)


# ------------------------------------------------------------------ loading

def _patch_artifact_json(stage_dir: Path, metadata: dict[str, Any]) -> None:
    """Record a copied artifact's provenance, in place (it names no folder of its own)."""
    path = stage_dir / BaseArtifact.FILE
    raw = read_json(path)
    raw["metadata"] = {**raw["metadata"], **metadata}
    write_json(path, raw)


def materialize_stage(plan: ResumePlan, spec: StageSpec) -> Any:
    """Copy one checked prior stage into the run folder and return its artifact.

    The copy is exact; only ``artifact.json`` is patched -- the fingerprint record and
    ``metadata["resume"]`` (where it was loaded from).
    """
    verdict = plan.verdicts[spec.name]
    if verdict.action != "load":
        raise ResumeRejected(
            f"{spec.name} is not loaded by this resume ({verdict.action})\n" + plan.report()
        )
    src = RunOutputs.open(plan.prior_run_dir).stage_dir(spec.name)
    dst = RunOutputs.create(plan.run_dir).stage_dir(spec.name)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    patch = {"contract": plan.records[spec.name], "resume": {"loaded_from": str(src)}}
    artifact = plan.artifacts[spec.name]
    artifact.save_dir = dst
    artifact.metadata = {**artifact.metadata, **patch}
    _patch_artifact_json(dst, patch)
    return artifact

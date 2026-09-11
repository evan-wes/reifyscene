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

Artifacts made before fingerprint records existed ("legacy": a ``compatibility``
block only) get one derived from what the prior run recorded -- the settings from the
run manifest of the invocation that produced the stage, the input by matching the
recorded name, size and modification time (a renamed project folder still resumes),
the lineage from the manifest chain. Every derivation is listed in the report and on
the loaded artifact (``metadata["resume"]``).
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
    stable,
)
from reifyscene.api.stages import STAGE_ORDER, StageSpec
from reifyscene.common.io import read_json, write_json


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
    action: str = "load"  # "load" | "reject"
    derived: list[str] = field(default_factory=list)
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
        # A derivation every checked stage shares (a moved input, legacy records) is said once.
        per_stage = [set(v.derived) for v in self.verdicts.values()]
        shared = set.intersection(*per_stage) if len(per_stage) > 1 else set()
        ordered_shared = [d for d in next(iter(self.verdicts.values())).derived if d in shared] \
            if shared else []
        lines.extend(f"  every stage: {d}" for d in ordered_shared)
        for stage in STAGE_ORDER:
            verdict = self.verdicts.get(stage)
            if verdict is None:
                continue
            mark = {"load": "ok", "reject": "REJECTED"}[verdict.action]
            if verdict.action == "load" and verdict.derived:
                mark = "ok (derived)"
            lines.append(f"  {stage:13s} {mark}")
            lines.extend(f"      problem: {p}" for p in verdict.problems)
            lines.extend(f"      derived: {d}" for d in verdict.derived if d not in shared)
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


# ------------------------------------------------------------------ legacy records

# The pre-fingerprint signature, kept only to recognize the run that produced a legacy
# artifact (its ``compatibility.stage_config``); never used to accept anything.
_LEGACY_VOLATILE_STAGE_KEYS: dict[str, set[str]] = {
    "geometry": {
        "device", "system_report", "frame_cache_dir", "output_cache_dir", "cpu_threads",
        "debug_shapes", "debug_max_points",
    },
    "pose_refine": {"device"},
    "tracking": {"device", "cpu_threads", "allow_github", "debug", "debug_max_points"},
    "humans": {"device"},
    "objects": {"device"},
    "render": set(),
}


def _legacy_stage_config(cfg: dict[str, Any], stage: str) -> Any:
    if stage == "ingest":
        sig = {
            "max_keyframes_per_stream": int(cfg.get("max_keyframes_per_stream", 8)),
            "keyframe_stride_sec": float(cfg.get("keyframe_stride_sec", 1.0)),
        }
        if bool(cfg.get("keyframes_all", False)):
            sig["keyframes_all"] = True
        return stable(sig)
    section = dict(cfg.get(stage, {}) or {})
    for key in _LEGACY_VOLATILE_STAGE_KEYS.get(stage, set()):
        section.pop(key, None)
    return stable(section)


def _producing_invocation(run_dir: Path, stage: str, seen: frozenset = frozenset()):
    """``(run_dir, manifest)`` of the invocation that produced ``stage``, or ``(None, why)``.

    A run's manifest records only its last invocation: the stages it ran are its own,
    the stages before its start were loaded from its prior run (recurse), and any
    stage after its stop is a leftover from an earlier invocation in that folder.
    """
    if run_dir in seen:
        return None, f"the run manifests loop back to {run_dir}"
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None, f"{run_dir} has no run_manifest.json to say which invocation produced it"
    manifest = read_json(manifest_path)
    start = STAGE_ORDER.index(manifest.get("start_stage") or STAGE_ORDER[0])
    stop = STAGE_ORDER.index(manifest.get("stop_stage") or STAGE_ORDER[-1])
    idx = STAGE_ORDER.index(stage)
    if start <= idx <= stop:
        return (run_dir, manifest), None
    if idx < start:
        prior = manifest.get("prior_run_dir")
        if not prior:
            return None, f"{run_dir} loaded it from a prior run its manifest does not name"
        prior_dir = Path(prior)
        if not prior_dir.exists():
            return None, f"{run_dir} loaded it from {prior_dir}, which no longer exists"
        return _producing_invocation(prior_dir, stage, seen | {run_dir})
    return None, (
        f"it is left over from an earlier invocation: {run_dir}'s last run stopped at "
        f"{STAGE_ORDER[stop]}"
    )


def _legacy_dir_signature(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"name": path.name, "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _legacy_input_matches(
    signatures: list[dict[str, Any]], video_paths: list[str]
) -> tuple[bool, list[str]]:
    """Whether a legacy video signature describes this invocation's input (maybe moved)."""
    current = [Path(p).expanduser().resolve() for p in video_paths]
    if len(signatures) != len(current):
        return False, [f"it recorded {len(signatures)} input(s), this run has {len(current)}"]
    notes: list[str] = []
    for sig, path in zip(signatures, current, strict=True):
        if not path.exists():
            return False, [f"input {path} does not exist"]
        now = _legacy_dir_signature(path)
        recorded = {k: sig.get(k) for k in ("name", "size_bytes", "mtime_ns")}
        if recorded != now:
            return False, [
                f"input differs: recorded {json.dumps(sig)}, found {json.dumps(now)} at {path}"
            ]
        if str(sig.get("path")) != str(path):
            notes.append(
                f"input re-rooted: recorded at {sig.get('path')}, found at {path} with the same "
                "name, size and modification time"
            )
    return True, notes


def _legacy_record(stage: str, artifact: Any, prior: Path, video_paths: list[str],
                   current_input: dict[str, Any], verdict: StageVerdict) -> dict[str, Any] | None:
    """A fingerprint for a pre-fingerprint artifact, derived from what the prior run recorded.

    Returns the record without upstream digests or its own digest (filled in once the
    upstream records are known), or ``None`` after rejecting the verdict.
    """
    compat = (getattr(artifact, "metadata", {}) or {}).get("compatibility")
    if not compat:
        verdict.reject("it carries no fingerprint and no legacy compatibility record")
        return None
    found, why = _producing_invocation(prior, stage)
    if found is None:
        verdict.reject(f"cannot tell which run produced it: {why}")
        return None
    run_dir, manifest = found
    cfg = manifest.get("config") or {}
    if stable(compat.get("stage_config")) != _legacy_stage_config(cfg, stage):
        verdict.reject(
            f"its recorded settings do not match {run_dir / 'run_manifest.json'}, the run that "
            "should have produced it"
        )
        return None
    ok, notes = _legacy_input_matches(list(compat.get("video_signatures") or []), video_paths)
    if not ok:
        verdict.reject("; ".join(notes))
        return None
    verdict.derived.extend(notes)
    verdict.derived.append(
        f"fingerprint derived from {run_dir / 'run_manifest.json'} "
        "(made before fingerprint records)"
    )
    return {
        "stage": stage,
        "version": 1,  # the pre-fingerprint baseline of every stage
        "settings": effective_settings(stage, cfg),
        "input": current_input,
        "_legacy": True,
    }


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
    """Decide, read-only, whether each prior stage can be loaded for this invocation.

    Checks every stage before ``start_stage`` (or ``stages``, e.g. every stage the
    prior run has, for a report). Mutates ``cfg`` to pin machine-picked settings to
    the prior run's values. Raises :class:`ResumeRejected` only for a request that
    cannot be checked at all (the prior run and the new run are the same folder).
    """
    prior = Path(prior_run_dir).expanduser().resolve()
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
    current_input = input_identity(video_paths)

    verdicts: dict[str, StageVerdict] = {}
    records: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for stage in stages:
        verdict = verdicts[stage] = StageVerdict(stage)
        stage_dir = prior / registry[stage].dir_name
        if not stage_dir.exists():
            verdict.reject(f"the prior run has no {stage} stage ({stage_dir})")
            continue
        try:
            artifact = registry[stage].load_fn(stage_dir)
        except Exception as exc:  # an incomplete or unreadable artifact cannot be resumed
            verdict.reject(f"cannot be loaded: {type(exc).__name__}: {exc}")
            continue
        artifacts[stage] = artifact
        record = (getattr(artifact, "metadata", {}) or {}).get("contract")
        if record is None:
            record = _legacy_record(stage, artifact, prior, video_paths, current_input, verdict)
            if record is None:
                continue
            # Lineage cannot be checked for a legacy artifact; the manifest chain above is
            # what vouches that it was built from the prior run's other stages.
            upstream = {
                u: (records.get(u) or {}).get("digest") for u in contract_for(stage).upstream
            }
            if any(d is None for d in upstream.values()):
                missing = sorted(u for u, d in upstream.items() if d is None)
                verdict.reject(f"its upstream stage(s) {', '.join(missing)} could not be loaded")
                continue
            record = fingerprint(
                stage, record["settings"], record["input"], upstream, record["version"]
            )
            verdict.derived.append("lineage derived from the run manifest chain")
        records[stage] = record

    pins = _pin(records, verdicts, cfg)

    for stage in stages:
        verdict = verdicts[stage]
        record = records.get(stage)
        if record is None or verdict.action == "reject":
            continue
        contract = contract_for(stage)
        if int(record.get("version", 0)) != contract.version:
            verdict.reject(
                f"made by contract version {record.get('version')} of {stage}; this code is "
                f"version {contract.version}"
            )
        diffs = _settings_diff(record.get("settings") or {}, effective_settings(stage, cfg))
        if diffs:
            verdict.reject("settings differ: " + "; ".join(diffs))
        if record.get("input") != current_input:
            verdict.reject("it was made from a different input (the frames it read have changed)")
        for up in contract.upstream:
            have = (records.get(up) or {}).get("digest")
            if up in verdicts and have is not None and record.get("upstream", {}).get(up) != have:
                verdict.reject(
                    f"it was built from a different {up} artifact than the one being loaded"
                )

    return ResumePlan(prior, run, start_stage, verdicts, records, artifacts, pins)


# ------------------------------------------------------------------ loading

def _patch_artifact_json(stage_dir: Path, metadata: dict[str, Any]) -> None:
    """Point a copied artifact at its new folder and record its provenance, in place."""
    path = stage_dir / "artifact.json"
    raw = read_json(path)
    saved = raw.get("save_dir")
    if isinstance(saved, dict) and saved.get("__type__") == "path":
        raw["save_dir"] = {**saved, "value": str(stage_dir)}
    else:
        raw["save_dir"] = str(stage_dir)
    raw["metadata"] = {**(raw.get("metadata") or {}), **metadata}
    write_json(path, raw)


def materialize_stage(plan: ResumePlan, spec: StageSpec) -> Any:
    """Copy one checked prior stage into the run folder and return its artifact.

    The copy is exact -- the artifact is not re-saved, so nothing a save/load round
    trip could lose is lost; only ``artifact.json`` is patched with the new folder,
    the fingerprint record and ``metadata["resume"]`` (where it was loaded from and
    what was derived).
    """
    verdict = plan.verdicts[spec.name]
    if verdict.action == "reject":
        raise ResumeRejected(plan.report())
    src = plan.prior_run_dir / spec.dir_name
    dst = plan.run_dir / spec.dir_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    provenance = {"loaded_from": str(src), "derived": list(verdict.derived)}
    patch = {"contract": plan.records[spec.name], "resume": provenance}
    _patch_artifact_json(dst, patch)
    artifact = plan.artifacts[spec.name]
    artifact.save_dir = dst
    artifact.metadata = {**(getattr(artifact, "metadata", {}) or {}), **patch}
    return artifact

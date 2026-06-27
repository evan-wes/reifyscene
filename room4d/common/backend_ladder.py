"""Backend fallback ladders for degradable perception stages.

Each perception stage (``segment`` / ``reid`` / ``objects`` / ``humans`` /
``tracking``) is defined by an artifact *contract* and filled by an ordered
ladder of backend *rungs*, best → graceful-empty. The ladder probes rungs
top-down, runs the first available one, and records a :class:`CapabilityManifest`
so degradation is **observable** (which rung ran, at what fidelity, why the
higher rungs were skipped) rather than silent.

This generalizes the ad-hoc per-backend ``backend_used`` idiom already used by
the object/human detectors and embedders (``real`` → ``heuristic`` /
``histogram``): a 2-rung ladder with a try/except. The contract guarantee
(design §4A) is that the floor rung is today's working backend or a
graceful-empty, so a missing/ungated model degrades *fidelity*, never
*completeness*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A probe returns ``None`` when the rung is available, else a short string
# explaining why it is not (e.g. ``'not-installed'``, ``'no-access'``).
Probe = Callable[[], Optional[str]]
Runner = Callable[[Any], Any]


@dataclass
class RungStatus:
    """Per-rung outcome recorded in the manifest."""

    name: str
    fidelity: str
    status: str  # 'ran' | 'skipped' | 'failed'
    reason: str = ''  # why skipped/failed; '' when it ran

    def to_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'fidelity': self.fidelity, 'status': self.status, 'reason': self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'RungStatus':
        return cls(name=str(d['name']), fidelity=str(d.get('fidelity', '')), status=str(d['status']), reason=str(d.get('reason', '')))


@dataclass
class CapabilityManifest:
    """Observable record of how a stage's ladder resolved.

    Stamped into ``artifact.metadata['capability']`` so downstream stages
    (graph / scene / eval) can read the *fidelity* the artifact carries.
    """

    stage: str
    backend_used: Optional[str] = None  # None => nothing ran (no graceful-empty rung)
    fidelity: Optional[str] = None
    rungs: list[RungStatus] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'stage': self.stage,
            'backend_used': self.backend_used,
            'fidelity': self.fidelity,
            'rungs': [r.to_dict() for r in self.rungs],
            'notes': list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'CapabilityManifest':
        return cls(
            stage=str(d.get('stage', '')),
            backend_used=d.get('backend_used'),
            fidelity=d.get('fidelity'),
            rungs=[RungStatus.from_dict(r) for r in d.get('rungs', [])],
            notes=list(d.get('notes', [])),
        )


class BackendRung:
    """One candidate backend in a ladder.

    name:     stable identifier stamped as ``backend_used``.
    fidelity: the fidelity tag the artifact carries when this rung runs
              (e.g. ``'open_vocab_propagation'`` vs ``'per_frame_assoc'``).
    runner:   ``(inputs) -> result`` — produces the stage's artifact/result.
    probe:    optional ``() -> str | None``; ``None`` return means available,
              a string is the skip reason. Omit for an always-available rung
              (e.g. the graceful-empty floor).
    """

    def __init__(self, name: str, fidelity: str, runner: Runner, probe: Optional[Probe] = None) -> None:
        self.name = name
        self.fidelity = fidelity
        self._runner = runner
        self._probe = probe

    def availability(self) -> Optional[str]:
        """Return ``None`` if available, else a short skip reason."""
        if self._probe is None:
            return None
        try:
            return self._probe()
        except Exception as exc:  # a probe must never crash the ladder
            return f'probe-error: {exc}'

    def run(self, inputs: Any) -> Any:
        return self._runner(inputs)


class BackendLadder:
    """An ordered ladder of backend rungs for one stage (best → floor)."""

    def __init__(self, stage: str, rungs: list[BackendRung]) -> None:
        if not rungs:
            raise ValueError(f"ladder for stage '{stage}' has no rungs")
        self.stage = stage
        self.rungs = rungs

    def resolve(self) -> tuple[Optional[BackendRung], CapabilityManifest]:
        """Probe top-down; return the first available rung + a manifest of skips.

        Does not run anything. The returned manifest records every rung above
        the chosen one as ``skipped`` with its reason; the chosen rung is left
        out (it is marked ``ran`` by :meth:`run`).
        """
        manifest = CapabilityManifest(stage=self.stage)
        for rung in self.rungs:
            reason = rung.availability()
            if reason is not None:
                manifest.rungs.append(RungStatus(rung.name, rung.fidelity, 'skipped', reason))
                continue
            return rung, manifest
        manifest.notes.append('no rung is available')
        return None, manifest

    def run(self, inputs: Any) -> tuple[Any, CapabilityManifest]:
        """Resolve and execute the first available rung.

        If a resolved rung's ``run`` raises, it is recorded as ``failed`` and
        the ladder continues to the next rung — so an installed-but-broken
        backend still degrades gracefully. Returns ``(result, manifest)``;
        ``result`` is ``None`` and ``manifest.backend_used`` is ``None`` only if
        no rung produced a result (i.e. no graceful-empty floor was provided).
        """
        manifest = CapabilityManifest(stage=self.stage)
        for rung in self.rungs:
            reason = rung.availability()
            if reason is not None:
                manifest.rungs.append(RungStatus(rung.name, rung.fidelity, 'skipped', reason))
                continue
            try:
                result = rung.run(inputs)
            except Exception as exc:
                manifest.rungs.append(RungStatus(rung.name, rung.fidelity, 'failed', f'run-error: {exc}'))
                continue
            manifest.rungs.append(RungStatus(rung.name, rung.fidelity, 'ran'))
            manifest.backend_used = rung.name
            manifest.fidelity = rung.fidelity
            return result, manifest
        manifest.notes.append('no rung produced a result')
        return None, manifest


def importable(*modules: str) -> Probe:
    """A probe that reports ``'not-installed: <mod>'`` if any module is missing.

    Convenience for model rungs whose package must import before use (e.g.
    ``importable('sam3')`` for the SAM 3 rung). Import side effects are avoided
    via :func:`importlib.util.find_spec`.
    """

    def _probe() -> Optional[str]:
        import importlib.util

        for mod in modules:
            try:
                if importlib.util.find_spec(mod) is None:
                    return f'not-installed: {mod}'
            except (ImportError, ValueError):
                return f'not-installed: {mod}'
        return None

    return _probe

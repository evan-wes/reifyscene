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
from typing import Any, Callable

class BackendRequired(RuntimeError):
    """A rung whose backend was asked for as ``real`` could not run: no silent fallback."""


# A probe returns ``None`` when the rung is available, else a short string
# explaining why it is not (e.g. ``'not-installed'``, ``'no-access'``).
Probe = Callable[[], str | None]
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
    def from_dict(cls, d: dict[str, Any]) -> RungStatus:
        return cls(name=str(d['name']), fidelity=str(d.get('fidelity', '')), status=str(d['status']), reason=str(d.get('reason', '')))


@dataclass
class CapabilityManifest:
    """Observable record of how a stage's ladder resolved.

    Stamped into ``artifact.metadata['capability']`` so downstream stages
    (graph / scene / eval) can read the *fidelity* the artifact carries.

    ``backend_used`` and ``fidelity`` answer *which backend ran*. That is not the same
    question as *what can be asserted from the result*, and the difference is where a
    green manifest over a worthless artifact hides: the semantics stage reported
    ``open_vocab`` while naming a person ``cup``, because it was asked a closed-vocabulary
    question, and every rung below it would have reported a degradation that never
    happened. ``provides`` answers the second question -- per graph-relevant capability,
    one of ``grounded`` (usable evidence), ``degraded`` (produced, but weakly) or
    ``absent`` (not produced at all) -- so "all green, graph is garbage" stops being
    representable.
    """

    stage: str
    backend_used: str | None = None  # None => nothing ran (no graceful-empty rung)
    fidelity: str | None = None
    rungs: list[RungStatus] = field(default_factory=list)
    provides: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'stage': self.stage,
            'backend_used': self.backend_used,
            'fidelity': self.fidelity,
            'rungs': [r.to_dict() for r in self.rungs],
            'provides': dict(self.provides),
            'notes': list(self.notes),
        }

    @classmethod
    def direct(
        cls, stage: str, *, backend_used: str, fidelity: str,
        provides: dict[str, str] | None = None, notes: list[str] | None = None,
    ) -> CapabilityManifest:
        """A manifest for a stage with no ladder to probe.

        Not every stage chooses between backends -- `graph` and `objects` assemble what
        upstream produced, and `tracking` runs one tracker. They still have to say what
        they produced, because the guarantee that matters downstream is about the
        *artifact's* fidelity, not about how many alternatives were considered.
        """
        return cls(
            stage=stage, backend_used=backend_used, fidelity=fidelity,
            rungs=[RungStatus(name=backend_used, fidelity=fidelity, status='ran')],
            provides=dict(provides or {}), notes=list(notes or []),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilityManifest:
        return cls(
            stage=str(d.get('stage', '')),
            backend_used=d.get('backend_used'),
            fidelity=d.get('fidelity'),
            rungs=[RungStatus.from_dict(r) for r in d.get('rungs', [])],
            provides=dict(d.get('provides') or {}),
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
    strict:   ``''`` (default) falls through to the next rung on a skip or a
              failure; ``'fail'`` raises :class:`BackendRequired` when the rung
              fails (a backend it relies on was asked for as ``real``);
              ``'always'`` also raises when it is skipped (the rung *is* the
              backend asked for).
    """

    def __init__(self, name: str, fidelity: str, runner: Runner, probe: Probe | None = None,
                 strict: str = '') -> None:
        if strict not in ('', 'fail', 'always'):
            raise ValueError(f"strict must be '', 'fail' or 'always', got {strict!r}")
        self.name = name
        self.fidelity = fidelity
        self._runner = runner
        self._probe = probe
        self.strict = strict

    def availability(self) -> str | None:
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

    def resolve(self) -> tuple[BackendRung | None, CapabilityManifest]:
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
                if rung.strict == 'always':
                    raise BackendRequired(
                        f"{self.stage}: the {rung.name} backend was asked for as real but is "
                        f"unavailable ({reason}); not falling back"
                    )
                manifest.rungs.append(RungStatus(rung.name, rung.fidelity, 'skipped', reason))
                continue
            try:
                result = rung.run(inputs)
            except Exception as exc:
                if rung.strict:
                    raise BackendRequired(
                        f"{self.stage}: the {rung.name} backend, asked for as real, failed "
                        f"({type(exc).__name__}: {exc}); not falling back"
                    ) from exc
                manifest.rungs.append(
                    RungStatus(rung.name, rung.fidelity, 'failed', f'run-error: {exc}')
                )
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

    def _probe() -> str | None:
        import importlib.util

        for mod in modules:
            try:
                if importlib.util.find_spec(mod) is None:
                    return f'not-installed: {mod}'
            except (ImportError, ValueError):
                return f'not-installed: {mod}'
        return None

    return _probe


def fallbacks_of(stage: str, artifact: Any) -> list[dict[str, Any]]:
    """What ``artifact``'s stage fell back from: the rungs above the one that ran.

    Read from the capability manifest every laddered stage records; the tracker has no
    ladder and records its own ``backend_used`` per stream. Empty when the best
    backend ran.
    """
    out: list[dict[str, Any]] = []
    cap = (getattr(artifact, 'metadata', {}) or {}).get('capability') or {}
    rungs = list(cap.get('rungs') or [])
    missed = [r for r in rungs if r.get('status') in ('skipped', 'failed')]
    if missed:
        out.append({
            'stage': stage,
            'wanted': missed[0].get('name'),
            'used': cap.get('backend_used') or 'nothing',
            'why': '; '.join(
                f"{r.get('name')} {r.get('status')}: {r.get('reason')}" for r in missed
            ),
        })
    for stream_id, payload in (getattr(artifact, 'track_payloads_by_stream', {}) or {}).items():
        used = str((payload or {}).get('backend_used') or '')
        if used and not used.startswith('cotracker'):
            out.append({'stage': stage, 'wanted': 'cotracker', 'used': used,
                        'why': f'stream {stream_id} was tracked by the {used} fallback'})
    return out

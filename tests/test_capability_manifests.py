"""Every stage must say how it resolved, and what can be asserted from the result.

The stage audit found seven of fifteen stages stamping no manifest at all -- including
`tracking`, `objects` and `graph`, the three the scene graph leans on hardest. A stage
that records nothing is indistinguishable from one that ran perfectly.
"""

from __future__ import annotations

from reifyscene.common.backend_ladder import CapabilityManifest, RungStatus


def test_a_manifest_carries_both_axes() -> None:
    m = CapabilityManifest.direct(
        'graph', backend_used='entity_builder', fidelity='entities_with_evidence',
        provides={'position': 'grounded', 'renderable': 'absent'},
    )
    d = m.to_dict()
    assert d['backend_used'] == 'entity_builder'      # which backend ran
    assert d['provides']['position'] == 'grounded'    # and what may be claimed from it
    assert d['provides']['renderable'] == 'absent'
    assert CapabilityManifest.from_dict(d).provides == m.provides


def test_provides_round_trips_through_an_older_manifest() -> None:
    """A manifest written before the second axis existed loads with an empty one."""
    legacy = {'stage': 'reid', 'backend_used': 'dinov2', 'fidelity': 'dino_features',
              'rungs': [], 'notes': []}
    assert CapabilityManifest.from_dict(legacy).provides == {}


def test_a_green_backend_can_still_report_a_useless_capability() -> None:
    """The failure the second axis exists to make representable.

    `semantics` reported backend `siglip2` at fidelity `open_vocab` while naming a person
    `cup`, because it was asked a closed-vocabulary question. Nothing degraded; the
    manifest was accurate and the output was worthless. Only `provides` can say so.
    """
    m = CapabilityManifest(
        stage='semantics', backend_used='siglip2', fidelity='open_vocab',
        rungs=[RungStatus(name='siglip2', fidelity='open_vocab', status='ran')],
        provides={'category': 'degraded'},
    )
    assert m.fidelity == 'open_vocab'
    assert m.provides['category'] == 'degraded'


def test_every_stage_that_runs_records_a_manifest() -> None:
    """A guard against the seven-of-fifteen state the audit found.

    Checked by reading the pipeline source rather than by running it: each stage's
    method must attach a manifest, whether or not it probes a ladder.
    """
    import inspect

    from reifyscene.api import pipeline as mod

    source = inspect.getsource(mod)
    for method in ('def track(', 'def perceive_objects(', 'def build_graph(',
                   'def compute_reid(', 'def compute_semantics(', 'def compute_humans('):
        start = source.index(method)
        nxt = source.find('\n    def ', start + 1)
        body = source[start:nxt if nxt != -1 else len(source)]
        assert 'attach_manifest' in body, f"{method.strip('def (')} stamps no manifest"


# --- a naming stage is graded on its ability to refuse ------------------------

def _sem(category_grade=None):
    """A semantics artifact carrying (or lacking) its own capability verdict."""
    from types import SimpleNamespace

    cap = {'provides': {'category': category_grade}} if category_grade else {}
    return SimpleNamespace(metadata={'capability': cap} if cap else {})


def test_downstream_inherits_the_naming_stage_s_verdict() -> None:
    from reifyscene.api.pipeline import _grade_category

    assert _grade_category(['e'], _sem('grounded')) == 'grounded'
    assert _grade_category(['e'], _sem('degraded')) == 'degraded'
    assert _grade_category(['e'], _sem('absent')) == 'absent'


def test_no_verdict_to_inherit_is_not_permission_to_assume_the_best() -> None:
    """The failure this axis exists to prevent, pinned.

    Read from the outside, a run whose separation gate was never armed looks exactly like
    one where every label was checked and stood: in both cases nothing abstained. Deriving
    a grade from the records is how a fence panel ends up labelled `cup` and graded
    `grounded`, so a consumer with nothing to inherit must assume the weaker answer.
    """
    from reifyscene.api.pipeline import _grade_category

    assert _grade_category(['e1', 'e2'], _sem(None)) == 'degraded'
    assert _grade_category(['e1'], None) == 'degraded'
    assert _grade_category([], None) == 'absent'

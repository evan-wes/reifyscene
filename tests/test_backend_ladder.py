"""Tests for the degradable backend ladder (design §4A, FP0)."""

from __future__ import annotations

import pytest

from room4d.common.backend_ladder import (
    BackendLadder,
    BackendRung,
    CapabilityManifest,
    RungStatus,
    importable,
)


def _rung(name, fidelity, value, reason=None):
    """A rung that returns ``value`` and is unavailable iff ``reason`` is set."""
    return BackendRung(name, fidelity, runner=lambda _inp: value, probe=(lambda: reason))


def test_empty_ladder_raises():
    with pytest.raises(ValueError):
        BackendLadder('segment', [])


def test_resolve_picks_first_available():
    ladder = BackendLadder('segment', [
        _rung('sam3', 'open_vocab', 'A', reason='no-access'),
        _rung('gsam2', 'open_vocab_apache', 'B'),
        _rung('maskrcnn', 'per_frame_assoc', 'C'),
    ])
    rung, manifest = ladder.resolve()
    assert rung is not None and rung.name == 'gsam2'
    # the skipped top rung is recorded with its reason; the chosen one is not yet 'ran'
    assert manifest.rungs == [RungStatus('sam3', 'open_vocab', 'skipped', 'no-access')]


def test_run_executes_first_available_and_stamps_manifest():
    ladder = BackendLadder('segment', [
        _rung('sam3', 'open_vocab', 'A', reason='not-installed: sam3'),
        _rung('gsam2', 'open_vocab_apache', 'B'),
    ])
    result, manifest = ladder.run(inputs=None)
    assert result == 'B'
    assert manifest.backend_used == 'gsam2'
    assert manifest.fidelity == 'open_vocab_apache'
    assert [r.status for r in manifest.rungs] == ['skipped', 'ran']
    assert manifest.rungs[0].reason == 'not-installed: sam3'


def test_run_error_falls_through_to_next_rung():
    def _boom(_inp):
        raise RuntimeError('cuda oom')

    ladder = BackendLadder('reid', [
        BackendRung('dinov3', 'strong', runner=_boom),  # available but crashes
        _rung('histogram', 'weak', 'fallback'),
    ])
    result, manifest = ladder.run(inputs=None)
    assert result == 'fallback'
    assert manifest.backend_used == 'histogram'
    assert manifest.rungs[0].status == 'failed'
    assert 'run-error' in manifest.rungs[0].reason


def test_graceful_empty_floor_always_runs():
    ladder = BackendLadder('segment', [
        _rung('sam3', 'open_vocab', 'A', reason='no-access'),
        _rung('maskrcnn', 'per_frame_assoc', 'B', reason='not-installed: torchvision'),
        BackendRung('empty', 'empty', runner=lambda _inp: []),  # no probe => always available
    ])
    result, manifest = ladder.run(inputs=None)
    assert result == []
    assert manifest.backend_used == 'empty'
    assert manifest.fidelity == 'empty'


def test_no_rung_available_returns_none():
    ladder = BackendLadder('segment', [
        _rung('sam3', 'open_vocab', 'A', reason='no-access'),
        _rung('gsam2', 'open_vocab', 'B', reason='not-installed'),
    ])
    result, manifest = ladder.run(inputs=None)
    assert result is None
    assert manifest.backend_used is None
    assert manifest.notes  # explains nothing produced a result


def test_probe_that_raises_is_treated_as_unavailable():
    def _bad_probe():
        raise OSError('weights dir unreadable')

    ladder = BackendLadder('segment', [
        BackendRung('flaky', 'x', runner=lambda _inp: 'A', probe=_bad_probe),
        _rung('floor', 'empty', 'B'),
    ])
    result, manifest = ladder.run(inputs=None)
    assert result == 'B'
    assert manifest.rungs[0].status == 'skipped'
    assert 'probe-error' in manifest.rungs[0].reason


def test_importable_probe():
    assert importable('os', 'sys')() is None
    reason = importable('definitely_not_a_real_module_xyz')()
    assert reason is not None and reason.startswith('not-installed')


def test_manifest_roundtrip():
    manifest = CapabilityManifest(
        stage='segment',
        backend_used='gsam2',
        fidelity='open_vocab_apache',
        rungs=[RungStatus('sam3', 'open_vocab', 'skipped', 'no-access'), RungStatus('gsam2', 'open_vocab_apache', 'ran')],
        notes=['n'],
    )
    restored = CapabilityManifest.from_dict(manifest.to_dict())
    assert restored == manifest

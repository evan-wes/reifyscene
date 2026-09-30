"""The run script's resume path end to end, on CPU backends: load, rebuild, reject, report."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from reifyscene.api.resume import ResumeRejected
from tests.conftest import write_test_video


def _main(monkeypatch, argv: list[str]) -> None:
    import scripts.run_single_video_reifyscene as runner

    monkeypatch.setattr(sys, 'argv', ['run_single_video_reifyscene.py', *argv])
    runner.main()


def _meta(run: Path, stage: str) -> dict:
    return json.loads((run / stage / 'artifact.json').read_text())['metadata']


def test_the_run_script_resumes_rebuilds_rejects_and_reports(tmp_path, monkeypatch, capsys) -> None:
    video = tmp_path / 'clip.mp4'
    write_test_video(video, num_frames=6)
    config = tmp_path / 'cfg.yaml'
    config.write_text('render:\n  device: cpu\n')
    common = ['--video', str(video), '--config', str(config),
              '--geometry-backend-mode', 'heuristic', '--resource-profile', 'none',
              '--tracking-max-frames', '4']
    a, b, c = tmp_path / 'a', tmp_path / 'b', tmp_path / 'c'
    _main(monkeypatch, [*common, '--run-dir', str(a)])

    shutil.rmtree(a / 'graph')  # a cheap stage gone: rebuilt, not rejected
    _main(monkeypatch, [*common, '--run-dir', str(b), '--prior-run-dir', str(a),
                        '--start-stage', 'scene'])
    manifest_b = json.loads((b / 'run_manifest.json').read_text())
    assert isinstance(manifest_b['fallbacks'], list)  # every fallback, recorded (maybe none)
    stages = manifest_b['resume']['stages']
    assert stages['graph']['action'] == 'rebuild'
    assert stages['tracking']['action'] == 'load'
    assert _meta(b, 'tracking')['resume']['loaded_from'] == str(a / 'tracking')
    assert 'resume' not in _meta(b, 'graph')  # it ran in this invocation
    tracking_digest = _meta(b, 'tracking')['contract']['digest']
    assert _meta(b, 'graph')['contract']['upstream']['tracking'] == tracking_digest
    assert (b / 'scene' / 'artifact.json').exists() and (b / 'render' / 'artifact.json').exists()

    # A changed setting a loaded stage used rejects before anything is written.
    with pytest.raises(ResumeRejected, match='lifting.point_conf_threshold'):
        _main(monkeypatch, [*common, '--run-dir', str(c), '--prior-run-dir', str(b),
                            '--start-stage', 'scene', '--point-conf-threshold', '0.5'])
    assert not c.exists()

    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, [*common, '--run-dir', str(c), '--prior-run-dir', str(b),
                            '--start-stage', 'tracking', '--check-resume'])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert 'resumable at tracking' in out and 'as late as handles' in out
    assert not c.exists()

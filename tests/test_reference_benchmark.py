"""Unit tests for the pure logic in scripts/run_reference_benchmark.py.

Subprocess-orchestration paths (Shape of Motion training, real conda eval,
env-version capture against the reference env) are integration-scope and run
manually on GPU hardware. These tests cover the stdlib-only logic: metric
parsing, compare-mode direction handling, sequence-layout validation, conda
discovery fallbacks, and the CLI guard rails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_reference_benchmark as rb

VERIFY_SCRIPT = rb.REPO_ROOT / "scripts" / "verify_reference_data.sh"
SETUP_SCRIPT = rb.REPO_ROOT / "scripts" / "setup_reference_tool.sh"

# Realistic stdout from Shape of Motion's evaluate_iphone.py at the pinned
# commit — the exact "mean <name>: <float>" format METRIC_PATTERNS targets,
# embedded in the usual checkpoint/progress noise.
SAMPLE_EVAL_OUTPUT = """\
Loading checkpoint from runs/paper-windmill-full/checkpoints/last.ckpt
paper-windmill: evaluating 2D tracking ...
100%|##########| 50/50 [00:12<00:00,  4.05it/s]
mean 2D tracking AJ: 0.4523
mean 2D tracking avg PCK: 0.6217
mean 2D tracking occlusion accuracy: 0.8841
evaluating 3D tracking ...
mean 3D tracking EPE: 0.0821
mean 3D tracking PCK (10cm): 0.996
mean 3D tracking PCK (5cm): 0.8204
evaluating novel-view synthesis ...
mean NV mPSNR: 19.53
mean NV mSSIM: 0.559
mean NV mLPIPS: 0.198
"""

EXPECTED_METRICS = {
    "masked_mpsnr": 19.53,
    "masked_mssim": 0.559,
    "masked_mlpips": 0.198,
    "track3d_epe": 0.0821,
    "track3d_pck_10cm": 0.996,
    "track3d_pck_5cm": 0.8204,
    "track2d_aj": 0.4523,
    "track2d_avg_pck": 0.6217,
    "track2d_occ_acc": 0.8841,
}


def _make_iphone_sequence(root: Path) -> Path:
    """Create the minimal layout both validators accept.

    Built generically from the module's own lists so the fixture cannot drift
    from the Python validator; the bash-validator parity test then holds the
    two validators together via this same fixture.
    """
    for d in rb.IPHONE_REQUIRED_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in rb.IPHONE_REQUIRED_FILES:
        target = root / f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    return root


def _make_completed_run(root: Path) -> Path:
    """Run dir that training_completed() accepts (sentinel + checkpoint)."""
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "last.ckpt").write_text("ckpt")
    (root / rb.TRAINING_COMPLETE_SENTINEL).touch()
    return root


def _write_benchmark_json(
    path: Path, method: str, sequence: str, metrics: dict[str, float], schema: str = rb.SCHEMA
) -> Path:
    path.write_text(json.dumps({"schema": schema, "method": method, "sequence": sequence, "metrics": metrics}))
    return path


def _patch_eval_subprocess(monkeypatch: pytest.MonkeyPatch, output: str, returncode: int = 0) -> list[list[str]]:
    """Replace the module's subprocess with a fake whose run() returns canned output."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - mirrors subprocess.run loosely
        calls.append(list(cmd))
        return SimpleNamespace(stdout=output, returncode=returncode)

    monkeypatch.setattr(
        rb,
        "subprocess",
        SimpleNamespace(
            run=fake_run,
            PIPE=subprocess.PIPE,
            STDOUT=subprocess.STDOUT,
            DEVNULL=subprocess.DEVNULL,
        ),
    )
    return calls


def test_metric_patterns_parse_realistic_eval_output() -> None:
    parsed = {
        key: float(match.group(1))
        for key, pattern in rb.METRIC_PATTERNS.items()
        if (match := re.search(pattern, SAMPLE_EVAL_OUTPUT))
    }
    assert parsed == EXPECTED_METRICS
    # every parsed metric must have a comparison direction (compare-mode contract)
    assert set(rb.HIGHER_IS_BETTER) == set(rb.METRIC_PATTERNS)


def test_run_eval_parses_metrics_writes_log_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    partial = "mean NV mPSNR: 19.53\nmean NV mSSIM: 0.559\n"
    calls = _patch_eval_subprocess(monkeypatch, partial)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics, output = rb.run_eval("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, "seq")
    assert metrics == {"masked_mpsnr": 19.53, "masked_mssim": 0.559}
    assert output == partial
    assert (run_dir / "benchmark_eval.log").read_text() == partial
    assert calls[0][:5] == ["conda", "run", "--no-capture-output", "-n", "ref_env"]
    out = capsys.readouterr().out
    assert "[warn] metric not found in eval output: masked_mlpips" in out
    assert "[warn] metric not found in eval output: track3d_epe" in out


def test_run_eval_error_paths_exit_with_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # non-zero eval exit -> SystemExit carrying the output tail; log still written
    _patch_eval_subprocess(monkeypatch, "Traceback: CUDA out of memory\n", returncode=1)
    with pytest.raises(SystemExit) as exc:
        rb.run_eval("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, "seq")
    assert "exit 1" in str(exc.value)
    assert "CUDA out of memory" in str(exc.value)
    assert (run_dir / "benchmark_eval.log").read_text().startswith("Traceback")

    # zero exit but no masked_* metrics parsed -> print-format drift guard
    _patch_eval_subprocess(monkeypatch, "mean 3D tracking EPE: 0.08\n", returncode=0)
    with pytest.raises(SystemExit) as exc:
        rb.run_eval("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, "seq")
    assert "No novel-view metrics parsed" in str(exc.value)


def test_compare_via_main_direction_tie_and_missing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_benchmark_json(
        tmp_path / "room4d.json",
        "room4d",
        "paper-windmill",
        {"masked_mpsnr": 20.0, "masked_mssim": 0.559, "masked_mlpips": 0.25, "track3d_epe": 0.08},
    )
    b = _write_benchmark_json(
        tmp_path / "ref.json",
        "shape-of-motion",
        "paper-windmill",
        {"masked_mpsnr": 19.53, "masked_mssim": 0.559, "masked_mlpips": 0.198},
    )
    monkeypatch.setattr(rb.sys, "argv", ["run_reference_benchmark.py", "--compare", str(a), str(b)])
    rb.main()
    out = capsys.readouterr().out
    rows = {
        line.split()[0]: line.split()
        for line in out.splitlines()
        if line.strip().startswith(("masked_", "track"))
    }
    assert rows["masked_mpsnr"][-1] == "room4d"  # higher is better
    assert rows["masked_mlpips"][-1] == "shape-of-motion"  # lower is better
    assert rows["masked_mssim"] == ["masked_mssim", "0.5590", "0.5590"]  # tie -> no winner
    assert rows["track3d_epe"][1:] == ["0.0800", "—"]  # missing metric -> em dash, no winner


def test_compare_rejects_wrong_schema(tmp_path: Path) -> None:
    a = _write_benchmark_json(tmp_path / "a.json", "room4d", "apple", {"masked_mpsnr": 20.0})
    b = _write_benchmark_json(
        tmp_path / "b.json", "som", "apple", {"masked_mpsnr": 19.0}, schema="other_schema_v0"
    )
    with pytest.raises(SystemExit) as exc:
        rb.compare(str(a), str(b))
    assert "other_schema_v0" in str(exc.value)
    assert rb.SCHEMA in str(exc.value)


def test_compare_warns_on_sequence_mismatch_without_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_benchmark_json(tmp_path / "a.json", "room4d", "apple", {})
    b = _write_benchmark_json(tmp_path / "b.json", "som", "spin", {})
    rb.compare(str(a), str(b))  # no shared metrics: header-only table, no crash
    out = capsys.readouterr().out
    assert "[warn] comparing different sequences: apple vs spin" in out
    assert "Sequence: apple" in out
    assert "metric" in out


def test_validate_iphone_sequence_accepts_complete_layout(tmp_path: Path) -> None:
    seq = _make_iphone_sequence(tmp_path / "paper-windmill")
    assert rb.validate_iphone_sequence(seq) is None


def test_validate_iphone_sequence_exits_listing_missing_entries(tmp_path: Path) -> None:
    seq = tmp_path / "broken"
    (seq / "rgb").mkdir(parents=True)
    (seq / "dataset.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        rb.validate_iphone_sequence(seq)
    msg = str(exc.value)
    assert "missing dir:   depth/" in msg
    assert "missing dir:   splits/" in msg
    assert "missing dir:   keypoint/" in msg
    assert "missing entry: extra.json" in msg
    assert "missing entry: splits/train.json" in msg
    assert "missing entry: splits/val.json" in msg
    assert "missing entry: flow3d_preprocessed/colmap/scale.npy" in msg
    assert "missing dir:   rgb/" not in msg  # present entries are not flagged
    assert "missing entry: dataset.json" not in msg


def test_find_conda_resolution_order_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conda_exe = tmp_path / "exe" / "conda"
    conda_exe.parent.mkdir()
    conda_exe.write_text("")
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    path_conda = path_dir / "conda"
    path_conda.write_text("#!/bin/sh\n")
    path_conda.chmod(0o755)
    home = tmp_path / "home"
    home_conda = home / "miniconda3" / "condabin" / "conda"
    home_conda.parent.mkdir(parents=True)
    home_conda.write_text("")
    empty = tmp_path / "empty"
    empty.mkdir()

    # 1) CONDA_EXE wins when it points at a real file
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(path_dir))
    monkeypatch.setenv("CONDA_EXE", str(conda_exe))
    assert rb.find_conda() == str(conda_exe)

    # 2) stale CONDA_EXE falls through to PATH lookup
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "missing-conda"))
    assert rb.find_conda() == str(path_conda)

    # 3) no CONDA_EXE, nothing on PATH -> $HOME/miniconda3 fallback
    monkeypatch.delenv("CONDA_EXE")
    monkeypatch.setenv("PATH", str(empty))
    assert rb.find_conda() == str(home_conda)

    # 4) nothing anywhere -> SystemExit with the locations checked
    monkeypatch.setenv("HOME", str(empty))
    with pytest.raises(SystemExit) as exc:
        rb.find_conda()
    assert "conda executable not found" in str(exc.value)


def test_main_guards_missing_args_and_missing_som_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # no --compare and no --data-dir/--run-dir -> argparse error (exit code 2)
    monkeypatch.setattr(rb.sys, "argv", ["run_reference_benchmark.py"])
    with pytest.raises(SystemExit) as exc:
        rb.main()
    assert exc.value.code == 2
    assert "--data-dir and --run-dir are required" in capsys.readouterr().err

    # benchmark dispatch with a missing Shape of Motion checkout -> setup hint
    seq = _make_iphone_sequence(tmp_path / "seq")
    monkeypatch.setattr(
        rb.sys,
        "argv",
        [
            "run_reference_benchmark.py",
            "--data-dir", str(seq),
            "--run-dir", str(tmp_path / "run"),
            "--som-dir", str(tmp_path / "no-such-checkout"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        rb.main()
    assert "Shape of Motion checkout not found" in str(exc.value)
    assert "setup_reference_tool.sh" in str(exc.value)


def _benchmark_args(seq: Path, run_dir: Path, som_dir: Path, **overrides) -> Namespace:
    base = dict(
        data_dir=str(seq),
        run_dir=str(run_dir),
        som_dir=str(som_dir),
        seq_name=None,
        conda_env="ref_env",
        num_epochs=None,
        retrain=False,
        output=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_benchmark_reuses_completed_run_and_requires_results_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    run_dir = _make_completed_run(tmp_path / "run")
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))

    # completed run + no --retrain skips training entirely, then the missing
    # results/ dir aborts before eval.
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
    assert "Re-run with --retrain" in str(exc.value)
    assert "reusing completed run" in capsys.readouterr().out


def test_training_completed_requires_proof_not_just_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_text("ckpt")
    # A bare checkpoint is NOT completion — SoM writes one at epoch 0.
    assert rb.training_completed(run_dir) is False
    # Sentinel proves completion.
    (run_dir / rb.TRAINING_COMPLETE_SENTINEL).touch()
    assert rb.training_completed(run_dir) is True
    # Pre-sentinel runs: cfg.yaml num_epochs + final-epoch videos dir.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "cfg.yaml").write_text("work_dir: x\nnum_epochs: 500\nbatch_size: 8\n")
    assert rb.training_completed(legacy) is False  # no videos yet
    (legacy / "videos" / "epoch_0499").mkdir(parents=True)
    assert rb.training_completed(legacy) is True


def test_benchmark_resumes_incomplete_run_and_retrain_clears_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)

    # Interrupted run (checkpoint, no completion proof) -> training RESUMES
    # rather than being skipped; fake training produces no results/ -> exit.
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_text("ckpt")
    with pytest.raises(SystemExit):
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
    assert "incomplete run detected — resuming training" in capsys.readouterr().out
    assert any("run_training.py" in part for call in calls for part in call)
    assert (run_dir / rb.TRAINING_COMPLETE_SENTINEL).exists()  # written post-train

    # --retrain clears prior outputs before training (SoM would otherwise
    # auto-resume and train ~0 epochs against a finished run).
    run_dir2 = _make_completed_run(tmp_path / "run2")
    (run_dir2 / "results").mkdir()
    (run_dir2 / "videos" / "epoch_0499").mkdir(parents=True)
    stale = run_dir2 / "results" / "stale.txt"
    stale.write_text("old")
    calls.clear()
    with pytest.raises(SystemExit):
        rb.benchmark(_benchmark_args(seq, run_dir2, som_dir, retrain=True))
    assert not stale.exists()
    assert not (run_dir2 / "videos").exists()
    assert any("run_training.py" in part for call in calls for part in call)


def test_benchmark_happy_path_writes_contract_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seq = _make_iphone_sequence(tmp_path / "paper-windmill")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    run_dir = _make_completed_run(tmp_path / "run")
    (run_dir / "results").mkdir()
    video = run_dir / "videos" / "epoch_0499"
    video.mkdir(parents=True)
    (video / "rgbs.mp4").write_bytes(b"")
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)
    monkeypatch.setattr(rb, "som_pinned_commit", lambda p: "deadbeef+dirty")
    monkeypatch.setattr(rb, "capture_env_versions", lambda c, e: {"torch": "2.11.0+cu128"})

    out_path = tmp_path / "out.json"
    rb.benchmark(_benchmark_args(seq, run_dir, som_dir, output=str(out_path)))

    result = json.loads(out_path.read_text())
    assert result["schema"] == rb.SCHEMA
    assert result["method"] == "shape-of-motion"
    assert result["method_commit"] == "deadbeef+dirty"
    assert result["sequence"] == "paper-windmill"  # defaults to data-dir basename
    assert result["metrics"] == EXPECTED_METRICS
    assert result["videos"]["rgb"].endswith("epoch_0499/rgbs.mp4")
    assert result["env"] == {"torch": "2.11.0+cu128"}
    assert result["train_minutes"] is None  # completed run reused, no training


def test_run_training_command_construction_and_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _patch_eval_subprocess(monkeypatch, "")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, None)
    assert "--num-epochs" not in calls[0]
    assert (run_dir / rb.TRAINING_COMPLETE_SENTINEL).exists()

    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, 5)
    cmd = calls[1]
    idx = cmd.index("--num-epochs")
    assert cmd[idx + 1] == "5"
    # tyro subcommand args must come AFTER the optional flag
    assert cmd.index("data:iphone") > idx


def test_best_effort_helpers_degrade_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # latest_rgb_video: no videos -> None; otherwise lexically-latest epoch
    assert rb.latest_rgb_video(tmp_path) is None
    for epoch in ("epoch_0010", "epoch_0500"):
        d = tmp_path / "videos" / epoch
        d.mkdir(parents=True)
        (d / "rgbs.mp4").write_bytes(b"")
    assert rb.latest_rgb_video(tmp_path) == str(tmp_path / "videos" / "epoch_0500" / "rgbs.mp4")

    # som_pinned_commit: not a git checkout -> "unknown", never raises.
    # GIT_CEILING_DIRECTORIES keeps git from walking up out of tmp_path and
    # "succeeding" against an enclosing repo (environment-dependent flake).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert rb.som_pinned_commit(tmp_path) == "unknown"

    # capture_env_versions: unrunnable conda -> {}, never raises
    assert rb.capture_env_versions(str(tmp_path / "missing-conda"), "ref_env") == {}


def test_compare_nan_metric_renders_dash_and_no_winner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # json.loads accepts the bare NaN literal; without a guard, NaN != x is
    # always True and (NaN > x) is always False, silently awarding 'better'
    # to the other method regardless of direction.
    a = tmp_path / "a.json"
    a.write_text(
        '{"schema": "%s", "method": "room4d", "sequence": "s",'
        ' "metrics": {"masked_mpsnr": NaN}}' % rb.SCHEMA
    )
    b = _write_benchmark_json(tmp_path / "b.json", "som", "s", {"masked_mpsnr": 19.5})
    rb.compare(str(a), str(b))
    row = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("masked_mpsnr")
    )
    assert "—" in row  # NaN renders as missing
    assert not row.rstrip().endswith("som")  # and no winner is declared


def test_compare_shows_unknown_metrics_with_undefined_direction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Metrics outside METRIC_PATTERNS (e.g. another emitter's extra keys) must
    # not be silently dropped; with both sides present but no known direction,
    # the better column shows '?'.
    a = _write_benchmark_json(
        tmp_path / "a.json", "room4d", "s",
        {"masked_mpsnr": 20.0, "heldout_psnr_mean": 21.0},
    )
    b = _write_benchmark_json(
        tmp_path / "b.json", "som", "s",
        {"masked_mpsnr": 19.5, "heldout_psnr_mean": 19.0},
    )
    rb.compare(str(a), str(b))
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "heldout_psnr_mean" in line)
    assert row.rstrip().endswith("?")
    known_row = next(line for line in out.splitlines() if line.strip().startswith("masked_mpsnr"))
    assert known_row.rstrip().endswith("room4d")


def test_bash_validator_parity_and_exit_codes(tmp_path: Path) -> None:
    """The bash validator and the Python validator must accept the SAME layout.

    The fixture is built from the Python module's required lists, so this test
    fails if the two validators drift apart.
    """
    # exit 2: missing args / unknown dataset
    assert subprocess.run(["bash", str(VERIFY_SCRIPT)], capture_output=True).returncode == 2
    assert subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "bogus", "--seq-dir", str(tmp_path)],
        capture_output=True,
    ).returncode == 2
    # exit 1: nonexistent dir
    assert subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "iphone", "--seq-dir", str(tmp_path / "nope")],
        capture_output=True,
    ).returncode == 1
    # exit 1 + [MISS] markers: incomplete layout
    broken = tmp_path / "broken"
    (broken / "rgb").mkdir(parents=True)
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "iphone", "--seq-dir", str(broken)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "[MISS]" in proc.stdout
    # exit 0: the SAME fixture the Python validator accepts (parity)
    seq = _make_iphone_sequence(tmp_path / "seq")
    rb.validate_iphone_sequence(seq)  # python side accepts
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "iphone", "--seq-dir", str(seq)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: OK" in proc.stdout


@pytest.mark.parametrize(
    ("args", "code", "needle"),
    [
        (["--conda-env", "room4d"], 1, "Refusing to run against protected"),
        (["--conda-env", "base"], 1, "Refusing to run against protected"),
        (["--with-mosca"], 2, "not implemented"),
        (["--bogus-flag"], 2, "Unknown option"),
        (["--help"], 0, "Usage:"),
    ],
)
def test_setup_script_guards_fire_before_any_install(args: list, code: int, needle: str) -> None:
    """The installer's safety guards (protected-env refusal, MoSca stub,
    arg validation) are conda-free paths — the highest-cost regressions the
    script could have, since the env guard is what protects the main room4d
    env's torch from being downgraded."""
    proc = subprocess.run(
        ["bash", str(SETUP_SCRIPT), *args], capture_output=True, text=True
    )
    assert proc.returncode == code, f"{args}: {proc.stdout}{proc.stderr}"
    assert needle in (proc.stdout + proc.stderr)


def test_som_pinned_commit_marks_patched_tree_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "som"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, env={**os.environ, **env})

    clean = rb.som_pinned_commit(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", clean)

    # sed-style in-place patch (what setup_reference_tool.sh does) -> +dirty
    (repo / "a.py").write_text("x = 2\n")
    assert rb.som_pinned_commit(repo) == f"{clean}+dirty"

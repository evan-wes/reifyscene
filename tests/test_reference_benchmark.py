"""Unit tests for the pure logic in scripts/run_reference_benchmark.py.

Subprocess-orchestration paths (Shape of Motion training, real conda eval,
env-version capture against the reference env) are integration-scope and run
manually on GPU hardware. These tests cover the stdlib-only logic: metric
parsing, the nvidia TensorBoard-harvest mapping, compare-mode direction
handling, sequence-layout validation, conda discovery fallbacks, and the CLI
guard rails.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
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

# What the _TB_HARVEST_SNIPPET prints: every scalar tag mapped to its
# highest-step event. train/* tags ride along for provenance and must be
# ignored by the val/* mapping.
SAMPLE_TB_TAGS = {
    "train/loss": {"value": 0.012, "step": 11900},
    "val/psnr": {"value": 23.41, "step": 11900},
    "val/ssim": {"value": 0.71, "step": 11900},
    "val/lpips": {"value": 0.21, "step": 11900},
    "val/fg_psnr": {"value": 21.05, "step": 11900},
    "val/fg_ssim": {"value": 0.65, "step": 11900},
    "val/fg_lpips": {"value": 0.30, "step": 11900},
    "val/bg_psnr": {"value": 24.2, "step": 11900},
    "val/bg_ssim": {"value": 0.74, "step": 11900},
    "val/bg_lpips": {"value": 0.19, "step": 11900},
}
SAMPLE_TB_OUTPUT = json.dumps(SAMPLE_TB_TAGS) + "\n"

EXPECTED_NVIDIA_METRICS = {
    "masked_mpsnr": 23.41,
    "masked_mssim": 0.71,
    "masked_mlpips": 0.21,
    "fg_psnr": 21.05,
    "fg_ssim": 0.65,
    "fg_lpips": 0.30,
    "bg_psnr": 24.2,
    "bg_ssim": 0.74,
    "bg_lpips": 0.19,
}


def _make_sequence(root: Path, dataset: str) -> Path:
    """Create the minimal layout both validators accept for a dataset.

    Built generically from the module's own lists so the fixture cannot drift
    from the Python validator; the bash-validator parity tests then hold the
    two validators together via this same fixture.
    """
    _, required_dirs, required_files = rb.DATASET_LAYOUTS[dataset]
    for d in required_dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in required_files:
        target = root / f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    if dataset == "nvidia":
        # the nvidia validator additionally requires a non-empty val split
        (root / "splits" / "val.json").write_text('{"frame_names": ["1_00000"]}')
    return root


def _make_iphone_sequence(root: Path) -> Path:
    return _make_sequence(root, "iphone")


def _make_completed_run(root: Path, epochs: str = "default") -> Path:
    """Run dir that completed_epochs() accepts (sentinel + checkpoint)."""
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "last.ckpt").write_text("ckpt")
    (root / rb.TRAINING_COMPLETE_SENTINEL).write_text(epochs + "\n")
    return root


def _write_benchmark_json(
    path: Path, method: str, sequence: str, metrics: dict[str, float], schema: str = rb.SCHEMA
) -> Path:
    path.write_text(json.dumps({"schema": schema, "method": method, "sequence": sequence, "metrics": metrics}))
    return path


def _patch_eval_subprocess(
    monkeypatch: pytest.MonkeyPatch, output: str, returncode: int = 0, stderr: str = ""
) -> list[list[str]]:
    """Replace the module's subprocess with a fake whose run() returns canned output."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - mirrors subprocess.run loosely
        calls.append(list(cmd))
        return SimpleNamespace(stdout=output, returncode=returncode, stderr=stderr)

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
    # every emittable metric must have a comparison direction (compare-mode
    # contract) — both the iphone-parsed keys and the nvidia-harvested ones
    assert set(rb.HIGHER_IS_BETTER) == (
        set(rb.METRIC_PATTERNS) | set(rb.NVIDIA_TB_METRICS.values())
    )


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


def test_harvest_nvidia_metrics_maps_tags_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_TB_OUTPUT, stderr="TF not found\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert metrics == EXPECTED_NVIDIA_METRICS
    # the snippet runs in the reference env against the run dir
    cmd = calls[0]
    assert cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "ref_env"]
    assert cmd[-1] == str(run_dir)
    assert "-c" in cmd
    # raw tags (train/* included) and stderr land in the log for provenance
    log = (run_dir / "benchmark_eval.log").read_text()
    assert "train/loss" in log
    assert "TF not found" in log


def test_harvest_nvidia_metrics_warns_on_missing_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # fg/bg tags absent (e.g. a bg-free model) -> warn and continue; the
    # masked_* trio present keeps the result usable.
    partial = {t: v for t, v in SAMPLE_TB_TAGS.items() if not t.startswith("val/fg")}
    _patch_eval_subprocess(monkeypatch, json.dumps(partial) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert metrics["masked_mpsnr"] == 23.41
    assert "fg_psnr" not in metrics
    assert metrics["bg_psnr"] == 24.2
    out = capsys.readouterr().out
    assert "[warn] scalar tag not found in event files: val/fg_psnr" in out


def test_harvest_nvidia_metrics_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # snippet crash -> SystemExit with the stderr tail; log still written
    _patch_eval_subprocess(monkeypatch, "", returncode=1, stderr="ImportError: tensorboard\n")
    with pytest.raises(SystemExit) as exc:
        rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert "exit 1" in str(exc.value)
    assert "ImportError: tensorboard" in str(exc.value)
    assert "ImportError" in (run_dir / "benchmark_eval.log").read_text()

    # zero exit but non-JSON stdout -> parse guard
    _patch_eval_subprocess(monkeypatch, "no json here\n")
    with pytest.raises(SystemExit) as exc:
        rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert "printed no JSON" in str(exc.value)

    # valid JSON but no val/* tags (validation never ran) -> hard abort,
    # an empty-metrics benchmark JSON must never be written
    _patch_eval_subprocess(monkeypatch, json.dumps({"train/loss": {"value": 0.1, "step": 5}}) + "\n")
    with pytest.raises(SystemExit) as exc:
        rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert "No val/* scalars" in str(exc.value)


def test_tb_harvest_snippet_is_valid_python() -> None:
    # The snippet only ever executes inside the reference conda env via
    # `python -c`; compile() catches syntax errors at unit-test time so a
    # future edit can't ship a snippet that only fails mid-benchmark on GPU.
    compile(rb._TB_HARVEST_SNIPPET, "<tb-harvest>", "exec")


def test_harvest_nvidia_metrics_tolerates_noise_before_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # conda run can prepend banner lines to stdout; only the LAST line is the
    # snippet's JSON object.
    _patch_eval_subprocess(monkeypatch, "conda banner\nanother line\n" + SAMPLE_TB_OUTPUT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert rb.harvest_nvidia_metrics("conda", "ref_env", run_dir) == EXPECTED_NVIDIA_METRICS


def test_harvest_nvidia_metrics_skips_non_finite_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A NaN/Inf val/* scalar (degenerate validation) must not be recorded as
    # the baseline — json.dumps would emit a non-RFC literal into the
    # comparison artifact. Mirrors the compare-side NaN hardening.
    tags = dict(SAMPLE_TB_TAGS)
    tags["val/psnr"] = {"value": float("nan"), "step": 11900}
    tags["val/fg_psnr"] = {"value": float("inf"), "step": 11900}
    _patch_eval_subprocess(monkeypatch, json.dumps(tags) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert "masked_mpsnr" not in metrics
    assert "fg_psnr" not in metrics
    assert metrics["masked_mssim"] == 0.71  # finite values still harvested
    assert all(math.isfinite(v) for v in metrics.values())
    out = capsys.readouterr().out
    assert "non-finite value for val/psnr" in out


def test_validate_nvidia_sequence_rejects_empty_val_split(tmp_path: Path) -> None:
    # SoM's has_validation is content-gated: an empty val split would train
    # for hours and only then fail the harvest with no val/* scalars.
    seq = _make_sequence(tmp_path / "Balloon1", "nvidia")
    (seq / "splits" / "val.json").write_text('{"frame_names": []}')
    with pytest.raises(SystemExit) as exc:
        rb.validate_sequence(seq, "nvidia")
    assert "empty frame_names" in str(exc.value)


def test_validate_nvidia_sequence_detects_partial_pairwise_download(tmp_path: Path) -> None:
    # The dataloader np.loads pairwise tracks for RANDOM frame pairs per
    # batch — a partial download passes layout checks then crashes at an
    # arbitrary epoch. The validator must check the full N^2 set.
    seq = _make_sequence(tmp_path / "Balloon1", "nvidia")
    (seq / "splits" / "train.json").write_text('{"frame_names": ["0_00000", "0_00001"]}')
    tracks = seq / "flow3d_preprocessed" / "2d_tracks" / "2x"
    for pair in ("0_00000_0_00000", "0_00000_0_00001", "0_00001_0_00000"):
        (tracks / f"{pair}.npy").write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        rb.validate_sequence(seq, "nvidia")
    assert "missing 1 of 4 pairwise track files" in str(exc.value)
    assert "0_00001_0_00001.npy" in str(exc.value)
    # completing the set passes
    (tracks / "0_00001_0_00001.npy").write_bytes(b"")
    assert rb.validate_sequence(seq, "nvidia") is None


def test_validate_nvidia_sequence_rejects_malformed_split(tmp_path: Path) -> None:
    seq = _make_sequence(tmp_path / "Balloon1", "nvidia")
    (seq / "splits" / "val.json").write_text('["not", "an", "object"]')
    with pytest.raises(SystemExit) as exc:
        rb.validate_sequence(seq, "nvidia")
    assert "not an object" in str(exc.value)


def test_harvest_nvidia_metrics_aborts_on_mixed_run_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # All val/* scalars are logged in one log_dict call at a single step; a
    # disagreement means the event files mix two training runs (shared run
    # dir) and the harvested numbers would be plausible-looking corruption.
    tags = {tag: dict(event) for tag, event in SAMPLE_TB_TAGS.items()}
    tags["val/ssim"]["step"] = 400
    _patch_eval_subprocess(monkeypatch, json.dumps(tags) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        rb.harvest_nvidia_metrics("conda", "ref_env", run_dir)
    assert "mix more than one training run" in str(exc.value)


def test_compare_warns_on_eval_protocol_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nvidia_eval = {"source": "training-validator-tensorboard", "resolution_factor": 2}
    a = tmp_path / "a.json"
    a.write_text(json.dumps({
        "schema": rb.SCHEMA, "method": "shape-of-motion", "sequence": "Jumping",
        "metrics": {"masked_mpsnr": 18.7}, "eval": nvidia_eval,
    }))
    b = _write_benchmark_json(tmp_path / "b.json", "reifyscene", "Jumping", {"masked_mpsnr": 19.0})
    rb.compare(str(a), str(b))
    assert "comparing different eval protocols" in capsys.readouterr().out

    # identical protocols stay silent
    c = tmp_path / "c.json"
    c.write_text(json.dumps({
        "schema": rb.SCHEMA, "method": "reifyscene", "sequence": "Jumping",
        "metrics": {"masked_mpsnr": 19.0}, "eval": nvidia_eval,
    }))
    rb.compare(str(a), str(c))
    assert "eval protocols" not in capsys.readouterr().out


def test_trained_data_dir_parsing_and_drift_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert rb.trained_data_dir(run) is None  # no cfg.yaml -> guard disarmed
    # yaml.dump quotes paths with special characters
    (run / "cfg.yaml").write_text("work_dir: x\ndata:\n  data_dir: '/with spaces/seq'\n")
    assert rb.trained_data_dir(run) == "/with spaces/seq"
    # a cfg.yaml without the key must warn, never silently disarm
    (run / "cfg.yaml").write_text("work_dir: x\nnothing: here\n")
    assert rb.trained_data_dir(run) is None
    assert "no data_dir key" in capsys.readouterr().out


def test_bash_validator_nvidia_content_checks(tmp_path: Path) -> None:
    """Parity for the content-level checks: empty val split and partial
    pairwise downloads must fail the bash validator too, not just Python."""
    seq = _make_sequence(tmp_path / "empty-val", "nvidia")
    (seq / "splits" / "val.json").write_text('{"frame_names": []}')
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "nvidia", "--seq-dir", str(seq)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "frame_names is empty" in proc.stdout

    partial = _make_sequence(tmp_path / "partial", "nvidia")
    (partial / "splits" / "train.json").write_text('{"frame_names": ["0_00000", "0_00001"]}')
    (partial / "flow3d_preprocessed/2d_tracks/2x/0_00000_0_00000.npy").write_bytes(b"")
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "nvidia", "--seq-dir", str(partial)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "partial download" in proc.stdout

    for fi in ("0_00000", "0_00001"):
        for fj in ("0_00000", "0_00001"):
            (partial / f"flow3d_preprocessed/2d_tracks/2x/{fi}_{fj}.npy").write_bytes(b"")
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "nvidia", "--seq-dir", str(partial)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_main_rejects_non_positive_num_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SoM would accept 0/negative, pay the full dataset-load cost, train an
    # empty epoch range, and get stamped complete.
    seq = _make_iphone_sequence(tmp_path / "seq")
    for bad in ("0", "-3"):
        monkeypatch.setattr(
            rb.sys, "argv",
            ["run_reference_benchmark.py", "--data-dir", str(seq),
             "--run-dir", str(tmp_path / "run"), "--num-epochs", bad],
        )
        with pytest.raises(SystemExit) as exc:
            rb.main()
        assert exc.value.code == 2
    assert "--num-epochs must be >= 1" in capsys.readouterr().err


def test_benchmark_reuse_rejects_mismatched_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run dir only proves metrics for the sequence it was trained on; the
    # nvidia harvest reads only the run dir, so reusing it with the wrong
    # --data-dir would silently relabel old metrics as the new sequence.
    seq = _make_sequence(tmp_path / "Jumping", "nvidia")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    run_dir = _make_completed_run(tmp_path / "run")
    (run_dir / "results").mkdir()
    (run_dir / "cfg.yaml").write_text(
        "work_dir: /elsewhere\ndata:\n  data_dir: /trained/on/Balloon1\nnum_epochs: 500\n"
    )
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))

    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir, dataset="nvidia"))
    assert "was trained on /trained/on/Balloon1" in str(exc.value)
    # --retrain is the explicit override: state is cleared, training restarts
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_TB_OUTPUT)
    with pytest.raises(SystemExit):  # fake training produces no results/
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir, dataset="nvidia", retrain=True))
    assert any("run_training.py" in part for call in calls for part in call)


@pytest.mark.parametrize("dataset", ["iphone", "nvidia"])
def test_bash_validator_fails_when_any_required_entry_missing(
    tmp_path: Path, dataset: str
) -> None:
    """Mutation-style parity: the fixture is Python-list-derived, so the plain
    parity tests can't catch the bash list silently becoming a SUBSET of the
    Python list. Deleting each required entry one at a time and demanding a
    bash failure closes that drift direction."""
    _, required_dirs, required_files = rb.DATASET_LAYOUTS[dataset]
    for i, entry in enumerate([*required_dirs, *required_files]):
        seq = _make_sequence(tmp_path / f"{dataset}-{i}", dataset)
        target = seq / entry
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        proc = subprocess.run(
            ["bash", str(VERIFY_SCRIPT), "--dataset", dataset, "--seq-dir", str(seq)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1, f"bash validator passed without {entry!r}"
        assert "[MISS]" in proc.stdout


def test_compare_nvidia_results_uses_fg_bg_directions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The fg_/bg_ keys must get REAL winner verdicts (not '?'), with the
    # direction values exercised behaviorally: flipping fg_lpips to
    # higher-is-better would award the wrong method here.
    a = _write_benchmark_json(
        tmp_path / "reifyscene.json", "reifyscene", "Balloon1",
        {"masked_mpsnr": 20.0, "fg_lpips": 0.20, "bg_psnr": 25.0},
    )
    b = _write_benchmark_json(
        tmp_path / "ref.json", "shape-of-motion", "Balloon1",
        {"masked_mpsnr": 19.0, "fg_lpips": 0.30, "bg_psnr": 26.0},
    )
    rb.compare(str(a), str(b))
    rows = {
        line.split()[0]: line.rstrip()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(("masked_", "fg_", "bg_"))
    }
    assert rows["fg_lpips"].endswith("reifyscene")  # lower lpips wins
    assert rows["bg_psnr"].endswith("shape-of-motion")  # higher psnr wins
    assert not rows["fg_lpips"].endswith("?")  # known direction, never '?'


def test_main_accepts_dataset_nvidia_and_rejects_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Through REAL argv (not a Namespace): --dataset nvidia must parse and
    # dispatch benchmark() with the nvidia validator — proven by reaching the
    # SoM-checkout guard with a layout only the nvidia validator accepts.
    seq = _make_sequence(tmp_path / "Balloon1", "nvidia")
    monkeypatch.setattr(
        rb.sys, "argv",
        [
            "run_reference_benchmark.py",
            "--dataset", "nvidia",
            "--data-dir", str(seq),
            "--run-dir", str(tmp_path / "run"),
            "--som-dir", str(tmp_path / "no-such-checkout"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        rb.main()
    assert "Shape of Motion checkout not found" in str(exc.value)

    # an unknown dataset is an argparse choices error (exit code 2)
    monkeypatch.setattr(
        rb.sys, "argv",
        ["run_reference_benchmark.py", "--dataset", "bogus",
         "--data-dir", str(seq), "--run-dir", str(tmp_path / "run")],
    )
    with pytest.raises(SystemExit) as exc:
        rb.main()
    assert exc.value.code == 2
    assert "invalid choice: 'bogus'" in capsys.readouterr().err


def test_benchmark_nvidia_happy_path_writes_contract_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seq = _make_sequence(tmp_path / "Balloon1", "nvidia")
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
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_TB_OUTPUT)
    monkeypatch.setattr(rb, "som_pinned_commit", lambda p: "deadbeef")
    monkeypatch.setattr(rb, "capture_env_versions", lambda c, e: {})

    out_path = tmp_path / "out.json"
    rb.benchmark(_benchmark_args(seq, run_dir, som_dir, dataset="nvidia", output=str(out_path)))

    result = json.loads(out_path.read_text())
    assert result["schema"] == rb.SCHEMA
    assert result["dataset_format"] == "nvidia"
    assert result["sequence"] == "Balloon1"
    assert result["metrics"] == EXPECTED_NVIDIA_METRICS
    # the 2x protocol is disclosed so a future emitter can detect mismatch
    assert result["eval"] == {
        "source": "training-validator-tensorboard",
        "resolution_factor": 2,
    }
    # completed run reused -> the only subprocess call is the TB harvest,
    # never evaluate_iphone.py
    assert len(calls) == 1
    assert not any("evaluate_iphone" in part for part in calls[0])
    # the run lock is released on the happy path too (file persists)
    os.close(rb.acquire_run_lock(run_dir))
    # not a resumed run -> provenance fields say so
    assert result["resumed"] is False
    assert result["resumed_from_epoch"] is None
    # the summary table must print the nvidia extras too — the old
    # `for key in METRIC_PATTERNS` loop silently dropped fg_/bg_ rows
    out = capsys.readouterr().out
    assert re.search(r"masked_mpsnr\s+23\.4100\s+↑", out)
    assert re.search(r"fg_lpips\s+0\.3000\s+↓", out)
    assert re.search(r"bg_psnr\s+24\.2000\s+↑", out)


def test_compare_via_main_direction_tie_and_missing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_benchmark_json(
        tmp_path / "reifyscene.json",
        "reifyscene",
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
    assert rows["masked_mpsnr"][-1] == "reifyscene"  # higher is better
    assert rows["masked_mlpips"][-1] == "shape-of-motion"  # lower is better
    assert rows["masked_mssim"] == ["masked_mssim", "0.5590", "0.5590"]  # tie -> no winner
    assert rows["track3d_epe"][1:] == ["0.0800", "—"]  # missing metric -> em dash, no winner


def test_compare_rejects_wrong_schema(tmp_path: Path) -> None:
    a = _write_benchmark_json(tmp_path / "a.json", "reifyscene", "apple", {"masked_mpsnr": 20.0})
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
    a = _write_benchmark_json(tmp_path / "a.json", "reifyscene", "apple", {})
    b = _write_benchmark_json(tmp_path / "b.json", "som", "spin", {})
    rb.compare(str(a), str(b))  # no shared metrics: header-only table, no crash
    out = capsys.readouterr().out
    assert "[warn] comparing different sequences: apple vs spin" in out
    assert "Sequence: apple" in out
    assert "metric" in out


def test_validate_sequence_accepts_complete_layouts(tmp_path: Path) -> None:
    assert rb.validate_sequence(_make_sequence(tmp_path / "pw", "iphone"), "iphone") is None
    assert rb.validate_sequence(_make_sequence(tmp_path / "b1", "nvidia"), "nvidia") is None


def test_validate_sequence_exits_listing_missing_entries(tmp_path: Path) -> None:
    seq = tmp_path / "broken"
    (seq / "rgb").mkdir(parents=True)
    (seq / "dataset.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        rb.validate_sequence(seq, "iphone")
    msg = str(exc.value)
    assert "iPhone/DyCheck" in msg
    assert "missing dir:   depth/" in msg
    assert "missing dir:   splits/" in msg
    assert "missing dir:   keypoint/" in msg
    assert "missing entry: extra.json" in msg
    assert "missing entry: splits/train.json" in msg
    assert "missing entry: splits/val.json" in msg
    assert "missing entry: flow3d_preprocessed/colmap/scale.npy" in msg
    assert "missing dir:   rgb/" not in msg  # present entries are not flagged
    assert "missing entry: dataset.json" not in msg


def test_validate_nvidia_sequence_requires_2x_and_val_assets(tmp_path: Path) -> None:
    # A bare rgb/ (the old placeholder check) must no longer pass: the
    # dataloader needs the factor-2 pyramid, per-frame cameras, depth,
    # masks, tracks, and the val covisible masks.
    seq = tmp_path / "shallow"
    (seq / "rgb").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        rb.validate_sequence(seq, "nvidia")
    msg = str(exc.value)
    assert "NVIDIA Dynamic Scenes" in msg
    assert "missing dir:   rgb/1x/" in msg
    assert "missing dir:   rgb/2x/" in msg
    assert "missing dir:   depth/2x/" in msg
    assert "missing dir:   flow3d_preprocessed/2d_tracks/2x/" in msg
    assert "missing dir:   flow3d_preprocessed/track_anything/2x/" in msg
    assert "missing dir:   covisible/2x/val/" in msg
    assert "missing entry: splits/train.json" in msg
    assert "missing entry: splits/val.json" in msg


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
        dataset="iphone",
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


def test_trained_num_epochs_parses_cfg(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert rb.trained_num_epochs(run) is None  # no cfg.yaml
    (run / "cfg.yaml").write_text("work_dir: x\nbatch_size: 8\n")
    assert rb.trained_num_epochs(run) is None  # no num_epochs key
    (run / "cfg.yaml").write_text("work_dir: x\nnum_epochs: 1000\nbatch_size: 8\n")
    assert rb.trained_num_epochs(run) == 1000
    if os.geteuid() != 0:  # root ignores file modes
        (run / "cfg.yaml").chmod(0)
        assert rb.trained_num_epochs(run) is None  # unreadable -> None, loudly
        assert "[warn] could not read" in capsys.readouterr().out
        (run / "cfg.yaml").chmod(0o644)


def test_checkpoint_epoch_parses_and_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = tmp_path / "last.ckpt"
    calls = _patch_eval_subprocess(monkeypatch, "700\n")
    assert rb.checkpoint_epoch("conda", "ref_env", ckpt) == 700
    assert calls[0][:5] == ["conda", "run", "--no-capture-output", "-n", "ref_env"]
    assert calls[0][-1] == str(ckpt)
    _patch_eval_subprocess(monkeypatch, "", returncode=1)
    assert rb.checkpoint_epoch("conda", "ref_env", ckpt) is None
    _patch_eval_subprocess(monkeypatch, "not a number\n")
    assert rb.checkpoint_epoch("conda", "ref_env", ckpt) is None
    # rc 0 with empty stdout (snippet printed nothing) -> the IndexError leg
    _patch_eval_subprocess(monkeypatch, "")
    assert rb.checkpoint_epoch("conda", "ref_env", ckpt) is None
    # the snippet itself must stay valid python (only runs in the ref env)
    compile(rb._CKPT_EPOCH_SNIPPET, "<ckpt-epoch>", "exec")


def _make_interrupted_run(root: Path, cfg_epochs: int | None) -> Path:
    """Run dir with a checkpoint but no completion sentinel."""
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints" / "last.ckpt").write_text("ckpt")
    if cfg_epochs is not None:
        (root / "cfg.yaml").write_text(f"work_dir: x\nnum_epochs: {cfg_epochs}\n")
    return root


def test_resume_recalls_recorded_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SoM never re-reads cfg.yaml, so the wrapper must pass the recorded
    # schedule back explicitly — a plain resume of an interrupted 1000-epoch
    # run previously retargeted to the default 500.
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)
    monkeypatch.setattr(rb, "checkpoint_epoch", lambda *a: 300)

    run_dir = _make_interrupted_run(tmp_path / "run", cfg_epochs=1000)
    with pytest.raises(SystemExit):  # fake training produces no results/
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
    train_cmd = next(c for c in calls if "run_training.py" in " ".join(c))
    idx = train_cmd.index("--num-epochs")
    assert train_cmd[idx + 1] == "1000"
    # sentinel records the recalled schedule, not 'default'
    assert (run_dir / rb.TRAINING_COMPLETE_SENTINEL).read_text().strip() == "1000"
    assert "resuming from epoch 300 toward its recorded 1000-epoch schedule" in capsys.readouterr().out

    # a recorded default schedule resumes WITHOUT --num-epochs so the
    # sentinel label stays 'default' and later default-reuse matches
    calls.clear()
    run_dir2 = _make_interrupted_run(tmp_path / "run2", cfg_epochs=rb.SOM_DEFAULT_EPOCHS)
    with pytest.raises(SystemExit):
        rb.benchmark(_benchmark_args(seq, run_dir2, som_dir))
    train_cmd = next(c for c in calls if "run_training.py" in " ".join(c))
    assert "--num-epochs" not in train_cmd
    assert (run_dir2 / rb.TRAINING_COMPLETE_SENTINEL).read_text().strip() == "default"


def test_resume_refuses_zero_epoch_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Checkpoint epoch >= recorded schedule -> range(ckpt, schedule) is empty:
    # SoM would exit 0 having trained nothing and the run would be stamped
    # complete with mid-training metrics. Must abort BEFORE training.
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)
    monkeypatch.setattr(rb, "checkpoint_epoch", lambda *a: 700)

    run_dir = _make_interrupted_run(tmp_path / "run", cfg_epochs=500)
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
    assert "already >= its recorded 500-epoch schedule" in str(exc.value)
    assert "train zero epochs" in str(exc.value)
    assert not any("run_training.py" in " ".join(c) for c in calls)
    assert not (run_dir / rb.TRAINING_COMPLETE_SENTINEL).exists()
    os.close(rb.acquire_run_lock(run_dir))  # lock released on abort

    # equality boundary: the "finished but died before being stamped" case
    # the error message itself names — a >= -> > regression must not slip
    monkeypatch.setattr(rb, "checkpoint_epoch", lambda *a: 500)
    run_dir2 = _make_interrupted_run(tmp_path / "run2", cfg_epochs=500)
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, run_dir2, som_dir))
    assert "already >= its recorded 500-epoch schedule" in str(exc.value)

    # one epoch left -> resumes (training invoked)
    monkeypatch.setattr(rb, "checkpoint_epoch", lambda *a: 499)
    run_dir3 = _make_interrupted_run(tmp_path / "run3", cfg_epochs=500)
    with pytest.raises(SystemExit):  # fake training -> no results/
        rb.benchmark(_benchmark_args(seq, run_dir3, som_dir))
    assert any("run_training.py" in " ".join(c) for c in calls)


def test_run_lock_excludes_holders_and_releases_on_death(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / rb.RUN_LOCK_NAME

    # exclusive while held: a second acquire (separate fd, flock treats it
    # independently even in-process) is refused with the holder's pid
    fd = rb.acquire_run_lock(run_dir)
    assert lock_path.read_text().strip() == str(os.getpid())
    with pytest.raises(SystemExit) as exc:
        rb.acquire_run_lock(run_dir)
    assert "locked by another running benchmark" in str(exc.value)
    assert str(os.getpid()) in str(exc.value)

    # releasing the fd releases the flock; the FILE persists by design
    # (unlinking would detach waiters from the locked inode)
    os.close(fd)
    assert lock_path.exists()
    os.close(rb.acquire_run_lock(run_dir))

    # the kernel releases the flock on process death — no reclaim protocol,
    # so a kill -9 can never wedge the run dir
    subprocess.run(
        [sys.executable, "-c",
         "import fcntl, os, sys\n"
         "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n",
         str(lock_path)],
        check=True,
    )
    os.close(rb.acquire_run_lock(run_dir))  # holder died -> acquirable

    # stale file content is diagnostics only — never blocks acquisition
    lock_path.write_text("999999\n")
    os.close(rb.acquire_run_lock(run_dir))


def test_benchmark_aborts_on_held_lock_without_breaking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Contention abort must leave the HOLDER's flock fully intact.
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    holder_fd = os.open(run_dir / rb.RUN_LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(SystemExit) as exc:
            rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
        assert "locked by another running benchmark" in str(exc.value)
        # the holder's flock survived the abort: a fresh acquire still fails
        with pytest.raises(SystemExit):
            rb.acquire_run_lock(run_dir)
    finally:
        os.close(holder_fd)


def test_epochs_label_canonicalizes_explicit_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # '500' and 'default' name the SAME schedule; two labels for it would
    # make an explicit --num-epochs 500 rerun spuriously abort against a
    # resumed run's 'default' sentinel.
    assert rb._epochs_label(None) == "default"
    assert rb._epochs_label(rb.SOM_DEFAULT_EPOCHS) == "default"
    assert rb._epochs_label(5) == "5"

    # end to end: a 'default' sentinel satisfies an explicit-500 request
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)
    run_dir = _make_completed_run(tmp_path / "run")  # sentinel 'default'
    (run_dir / "results").mkdir()
    out_path = tmp_path / "out.json"
    rb.benchmark(_benchmark_args(
        seq, run_dir, som_dir, num_epochs=rb.SOM_DEFAULT_EPOCHS, output=str(out_path)
    ))
    assert "reusing completed run" in capsys.readouterr().out
    assert json.loads(out_path.read_text())["epochs"] == "default"


def test_resumed_run_records_provenance_in_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # train_minutes on a resumed run covers only the final segment; the
    # resume fields keep the artifact honest about that.
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)
    monkeypatch.setattr(rb, "checkpoint_epoch", lambda *a: 300)
    monkeypatch.setattr(rb, "som_pinned_commit", lambda p: "deadbeef")
    monkeypatch.setattr(rb, "capture_env_versions", lambda c, e: {})

    run_dir = _make_interrupted_run(tmp_path / "run", cfg_epochs=500)
    (run_dir / "results").mkdir()  # renders from the dead segment persist
    out_path = tmp_path / "out.json"
    rb.benchmark(_benchmark_args(seq, run_dir, som_dir, output=str(out_path)))
    result = json.loads(out_path.read_text())
    assert result["resumed"] is True
    assert result["resumed_from_epoch"] == 300
    assert result["epochs"] == "default"


def test_benchmark_fresh_run_dir_trains_with_requested_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No checkpoint at all -> the resume machinery stays out of the way and
    # the requested schedule passes straight through to training.
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)

    run_dir = tmp_path / "fresh"
    with pytest.raises(SystemExit):  # fake training -> no results/
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir, num_epochs=5))
    train_cmd = next(c for c in calls if "run_training.py" in " ".join(c))
    idx = train_cmd.index("--num-epochs")
    assert train_cmd[idx + 1] == "5"
    assert (run_dir / rb.TRAINING_COMPLETE_SENTINEL).read_text().strip() == "5"


def test_completed_epochs_requires_proof_and_records_schedule(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_text("ckpt")
    # A bare checkpoint is NOT completion — SoM writes one at epoch 0.
    assert rb.completed_epochs(run_dir) is None
    # Sentinel proves completion AND records the schedule (a smoke run can
    # never be mistaken for a full baseline).
    (run_dir / rb.TRAINING_COMPLETE_SENTINEL).write_text("5\n")
    assert rb.completed_epochs(run_dir) == "5"
    # Empty/legacy sentinel -> default schedule.
    (run_dir / rb.TRAINING_COMPLETE_SENTINEL).write_text("")
    assert rb.completed_epochs(run_dir) == "default"
    # Pre-sentinel runs: cfg.yaml num_epochs + final-epoch videos dir.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "cfg.yaml").write_text("work_dir: x\nnum_epochs: 500\nbatch_size: 8\n")
    assert rb.completed_epochs(legacy) is None  # no videos yet
    (legacy / "videos" / "epoch_0499").mkdir(parents=True)
    assert rb.completed_epochs(legacy) == "default"  # 500 = SoM default
    # Non-default legacy schedule reports its actual epoch count.
    short = tmp_path / "short"
    short.mkdir()
    (short / "cfg.yaml").write_text("num_epochs: 37\n")
    (short / "videos" / "epoch_0036").mkdir(parents=True)
    assert rb.completed_epochs(short) == "37"


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
    run_dir = _make_interrupted_run(tmp_path / "run", cfg_epochs=rb.SOM_DEFAULT_EPOCHS)
    with pytest.raises(SystemExit):
        rb.benchmark(_benchmark_args(seq, run_dir, som_dir))
    out = capsys.readouterr().out
    # a default-schedule resume passes no flag (label stays 'default')
    train_cmd = next(c for c in calls if "run_training.py" in " ".join(c))
    assert "--num-epochs" not in train_cmd
    # the fake subprocess output is unparseable as a checkpoint epoch, so
    # the pre-check degrades to a warning and the resume proceeds
    assert "could not read the checkpoint's epoch" in out
    assert "incomplete run detected — resuming from epoch ?" in out
    os.close(rb.acquire_run_lock(run_dir))  # lock released on exit
    assert any("run_training.py" in part for call in calls for part in call)
    assert (run_dir / rb.TRAINING_COMPLETE_SENTINEL).exists()  # written post-train

    # an interrupted run with NO recorded schedule fails CLOSED — guessing
    # could silently retarget the run toward the wrong schedule
    calls.clear()
    run_dir_nocfg = _make_interrupted_run(tmp_path / "run-nocfg", cfg_epochs=None)
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, run_dir_nocfg, som_dir))
    assert "missing or records no schedule" in str(exc.value)
    assert not any("run_training.py" in " ".join(c) for c in calls)

    # --retrain clears prior outputs before training (SoM would otherwise
    # auto-resume and train ~0 epochs against a finished run) — including the
    # previous run's result artifacts, which must not survive a retrain.
    run_dir2 = _make_completed_run(tmp_path / "run2")
    (run_dir2 / "results").mkdir()
    (run_dir2 / "videos" / "epoch_0499").mkdir(parents=True)
    stale = run_dir2 / "results" / "stale.txt"
    stale.write_text("old")
    stale_json = run_dir2 / "reference_benchmark.json"
    stale_json.write_text("{}")
    # Stale event files would outrank the retrained run's scalars in the
    # nvidia harvest (a retrain restarts global_step at 0); a stale cfg.yaml
    # would false-positive the provenance guard if the retrain dies during
    # dataset load; code/ is SoM's per-invocation source backup (disk leak).
    stale_events = run_dir2 / "events.out.tfevents.1700000000.host.1.0"
    stale_events.write_bytes(b"")
    (run_dir2 / "cfg.yaml").write_text("data:\n  data_dir: /old/sequence\n")
    (run_dir2 / "code" / "2026-01-01").mkdir(parents=True)
    calls.clear()
    with pytest.raises(SystemExit):
        rb.benchmark(_benchmark_args(seq, run_dir2, som_dir, retrain=True))
    assert not stale.exists()
    assert not stale_json.exists()
    assert not (run_dir2 / "videos").exists()
    assert not stale_events.exists()
    assert not (run_dir2 / "cfg.yaml").exists()
    assert not (run_dir2 / "code").exists()
    assert any("run_training.py" in part for call in calls for part in call)


def test_benchmark_epoch_schedule_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seq = _make_iphone_sequence(tmp_path / "seq")
    som_dir = tmp_path / "som"
    som_dir.mkdir()
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("")
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    calls = _patch_eval_subprocess(monkeypatch, SAMPLE_EVAL_OUTPUT)

    # Reuse with a DIFFERENT schedule than the run completed with -> abort
    # (a 5-epoch smoke result must never be published as a full baseline,
    # and vice versa).
    full_run = _make_completed_run(tmp_path / "full")
    (full_run / "results").mkdir()
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, full_run, som_dir, num_epochs=5))
    assert "completed with epochs=default" in str(exc.value)
    assert "epochs=5 was requested" in str(exc.value)
    assert not calls  # no training, no eval — hard abort

    # Resume of an interrupted run with an explicit --num-epochs -> abort:
    # resume recalls the recorded schedule from cfg.yaml, so an explicit
    # value is ambiguous (zero-epoch stamping is prevented separately by
    # the checkpoint-epoch pre-check).
    interrupted = _make_interrupted_run(tmp_path / "interrupted", cfg_epochs=None)
    with pytest.raises(SystemExit) as exc:
        rb.benchmark(_benchmark_args(seq, interrupted, som_dir, num_epochs=50))
    assert "--retrain" in str(exc.value)
    # the message must not repeat the old false claim about saved schedules
    assert "saved schedule" not in str(exc.value)
    assert not calls


def test_run_training_sentinel_not_written_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(
        rb, "subprocess",
        SimpleNamespace(run=failing_run, PIPE=subprocess.PIPE,
                        STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL,
                        CalledProcessError=subprocess.CalledProcessError),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, None, "iphone")
    # A failed training (OOM, Ctrl-C) must never stamp the run complete.
    assert not (run_dir / rb.TRAINING_COMPLETE_SENTINEL).exists()


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
    assert result["epochs"] == "default"  # schedule recorded in the artifact
    assert result["eval"] == {"source": "evaluate_iphone", "resolution_factor": 1}
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

    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, None, "iphone")
    assert "--num-epochs" not in calls[0]
    sentinel = run_dir / rb.TRAINING_COMPLETE_SENTINEL
    assert sentinel.read_text().strip() == "default"

    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, 5, "iphone")
    cmd = calls[1]
    idx = cmd.index("--num-epochs")
    assert cmd[idx + 1] == "5"
    # tyro subcommand args must come AFTER the optional flag
    assert cmd.index("data:iphone") > idx
    # ... and the sentinel records the schedule it completed with, written
    # atomically (no temp residue)
    assert sentinel.read_text().strip() == "5"
    assert not (run_dir / (rb.TRAINING_COMPLETE_SENTINEL + ".tmp")).exists()

    # an explicit 500 IS SoM's default schedule — one canonical label
    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, 500, "iphone")
    assert sentinel.read_text().strip() == "default"

    # the dataset flag selects the tyro data subcommand
    rb.run_training("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, None, "nvidia")
    assert "data:nvidia" in calls[3]
    assert "data:iphone" not in calls[3]


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
        '{"schema": "%s", "method": "reifyscene", "sequence": "s",'
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

    # Infinity has the same wrong-winner failure shape as NaN (json.loads
    # accepts the literal; inf passes a NaN-only guard and always "wins").
    c = tmp_path / "c.json"
    c.write_text(
        '{"schema": "%s", "method": "reifyscene", "sequence": "s",'
        ' "metrics": {"masked_mpsnr": Infinity}}' % rb.SCHEMA
    )
    rb.compare(str(c), str(b))
    row = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("masked_mpsnr")
    )
    assert "—" in row
    assert not row.rstrip().endswith("reifyscene")


def test_metric_patterns_handle_scientific_notation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # '([0-9.]+)' would truncate 8.21e-05 to 8.21 — a 100,000x silent error.
    output = "mean NV mPSNR: 19.53\nmean 3D tracking EPE: 8.21e-05\n"
    _patch_eval_subprocess(monkeypatch, output)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics, _ = rb.run_eval("conda", "ref_env", tmp_path / "som", tmp_path / "data", run_dir, "s")
    assert metrics["track3d_epe"] == pytest.approx(8.21e-05)


def test_compare_shows_unknown_metrics_with_undefined_direction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Metrics outside METRIC_PATTERNS (e.g. another emitter's extra keys) must
    # not be silently dropped; with both sides present but no known direction,
    # the better column shows '?'.
    a = _write_benchmark_json(
        tmp_path / "a.json", "reifyscene", "s",
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
    assert known_row.rstrip().endswith("reifyscene")


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
    rb.validate_sequence(seq, "iphone")  # python side accepts
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "iphone", "--seq-dir", str(seq)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: OK" in proc.stdout


def test_bash_validator_nvidia_parity(tmp_path: Path) -> None:
    """Same parity contract for the nvidia layout — and the old rgb-only
    placeholder check must be gone (a bare rgb/ dir now fails)."""
    shallow = tmp_path / "shallow"
    (shallow / "rgb").mkdir(parents=True)
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "nvidia", "--seq-dir", str(shallow)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "[MISS]" in proc.stdout

    seq = _make_sequence(tmp_path / "seq", "nvidia")
    rb.validate_sequence(seq, "nvidia")  # python side accepts
    proc = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--dataset", "nvidia", "--seq-dir", str(seq)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: OK" in proc.stdout
    assert "TODOS.md" not in proc.stdout  # placeholder caveat retired
    assert "data:nvidia" in proc.stdout  # usage hint names the right subcommand


@pytest.mark.parametrize(
    ("args", "code", "needle"),
    [
        (["--conda-env", "reifyscene"], 1, "Refusing to run against protected"),
        (["--conda-env", "base"], 1, "Refusing to run against protected"),
        (["--with-mosca"], 2, "not implemented"),
        (["--bogus-flag"], 2, "Unknown option"),
        (["--help"], 0, "Usage:"),
    ],
)
def test_setup_script_guards_fire_before_any_install(args: list, code: int, needle: str) -> None:
    """The installer's safety guards (protected-env refusal, MoSca stub,
    arg validation) are conda-free paths — the highest-cost regressions the
    script could have, since the env guard is what protects the main reifyscene
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

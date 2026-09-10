"""In-pipeline masked-NVS metric contract.

The metric is dataset-coupled (held-out val cameras + covisibility masks only
exist for benchmark datasets), so it returns ``{}`` gracefully when there is no
dataset source or no trained checkpoint -- arbitrary-video runs are unaffected.
A real scoring run is exercised by the paper-windmill validation, not a unit
test.
"""

from __future__ import annotations

from pathlib import Path

from reifyscene.render.eval.masked_nvs_benchmark import score_render_artifact_masked_nvs


class _Art:
    def __init__(self, save_dir, scene_state):
        self.save_dir = save_dir
        self.scene_state = scene_state


def test_masked_nvs_empty_without_dataset_root() -> None:
    art = _Art("/tmp", {"checkpoint": "trained_gaussians.npz"})
    assert score_render_artifact_masked_nvs(art, {}, device="cpu") == {}


def test_masked_nvs_empty_without_checkpoint(tmp_path: Path) -> None:
    art = _Art(str(tmp_path), {})  # render artifact carries no checkpoint
    assert score_render_artifact_masked_nvs(art, {"dataset_root": str(tmp_path)}, device="cpu") == {}


def test_masked_nvs_empty_when_checkpoint_file_missing(tmp_path: Path) -> None:
    art = _Art(str(tmp_path), {"checkpoint": "missing.npz"})
    assert score_render_artifact_masked_nvs(art, {"dataset_root": str(tmp_path)}, device="cpu") == {}

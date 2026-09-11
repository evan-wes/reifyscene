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


def test_static_scene_without_training_views_is_refused_not_misplaced(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np

    import reifyscene.render.eval.masked_nvs_benchmark as nvs

    ones = np.ones((1, 3), np.float32)
    np.savez(
        tmp_path / "trained_gaussians.npz",
        means=ones,
        quats=np.ones((1, 4), np.float32),
        scales=ones,
        opacities=np.ones(1, np.float32),
        colors=ones,
    )
    monkeypatch.setattr(nvs, "eval_dataset", lambda cfg: object())  # never reached for a camera
    art = _Art(str(tmp_path), {"checkpoint": "trained_gaussians.npz"})
    result = score_render_artifact_masked_nvs(art, {"dataset_root": str(tmp_path)}, device="cpu")
    assert "masked_mpsnr" not in result
    assert result["kind"] == "static" and result["alignment"]["status"] == "refused"


def test_render_stage_scores_a_static_scene_through_its_training_views(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import reifyscene.render.eval.masked_nvs_benchmark as nvs
    from reifyscene.api.pipeline import ReifyScenePipeline

    seen = {}

    def fake_score(artifact, eval_cfg, *, train_views=None, device="cuda"):
        seen.update(eval_cfg=eval_cfg, train_views=train_views)
        return {"masked_mpsnr": 1.0}

    monkeypatch.setattr(nvs, "score_render_artifact_masked_nvs", fake_score)
    render_artifact = SimpleNamespace(scene_state={"kind": "static"}, save=lambda: None)
    pipeline = SimpleNamespace(
        cfg={"evaluation": {"masked_nvs": {"dataset_root": "spin"}}, "render": {"device": "cpu"}},
        geometry_backend_name="dataset",  # even the oracle's static scene is canonicalized
        renderer=SimpleNamespace(fit=lambda scene, save_dir: render_artifact),
        _stage_dir=lambda name: tmp_path / name,
        _stamp_stage_metadata=lambda *args, **kwargs: None,
        _video_paths_for_stage=lambda *args: [],
    )
    scene = SimpleNamespace(train_views=[{"frame_key": "0:0"}])
    out = ReifyScenePipeline.fit_renderer(pipeline, scene)
    assert seen["eval_cfg"]["align"] == "sim3"
    assert seen["train_views"] is scene.train_views
    assert out.scene_state["evaluation"] == {"masked_mpsnr": 1.0}

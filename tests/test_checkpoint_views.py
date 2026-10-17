"""Held-out view placement (shared by the masked-NVS scorer and the view export)
and the checkpoint view export itself.

The renderer is gsplat/GPU-only, so these run on CPU against stand-ins for the
rasterizer, the scene and the dataset reader. A real export is exercised by
``scripts/export_checkpoint_views.py`` on a trained run, not a unit test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import reifyscene.render.eval.masked_nvs_benchmark as nvs
from reifyscene.api.run_outputs import RunOutputs
from reifyscene.geometry.similarity import apply_similarity_to_c2w
from reifyscene.render.eval.checkpoint_views import (
    export_heldout_views,
    export_replay,
    export_static_heldout_views,
    pick_evenly,
)
from reifyscene.render.eval.masked_nvs_benchmark import (
    AlignmentFailed,
    default_alignment,
    place_heldout_views,
    place_static_heldout_views,
    score_dynamic_bundle,
    score_render_artifact_masked_nvs_dynamic,
    score_static_cloud,
)
from reifyscene.render.gaussian4d.motion_init import BorrowedCamera
from scripts import export_checkpoint_views as cli
from tests.conftest import write_manifest

W, H = 80, 60


class _Dataset:
    """Stand-in for DatasetGeometryInitializer: 5 held-out views of camera '1'."""

    def __init__(self, n: int = 5) -> None:
        self.n = n

    def split_frames(self, split=None):
        return [("1", 10 * t) for t in range(self.n)]

    def frame_name(self, cam, t):
        return f"{cam}_{int(t):05d}"

    def load_camera(self, name):
        t_world_cam = np.eye(4, dtype=np.float32)
        t_world_cam[:3, 3] = [0.0, 0.0, -4.0]  # camera 4 units behind the origin
        return SimpleNamespace(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2), t_world_cam, (W, H)

    def load_rgb(self, name):
        return np.zeros((H, W, 3), dtype=np.uint8)

    def load_covisible(self, name, split="val"):
        return np.ones((H, W), dtype=bool)


class _Scene:
    def __init__(self) -> None:
        self.times: list[int] = []

    def cloud_at(self, t):
        self.times.append(int(t))
        z = torch.zeros(1, 3)
        return SimpleNamespace(means=z, quats=z, scales=z, opacities=z[:, 0], colors=z)


class _Rasterizer:
    """Renders flat grey (0.5) at the camera's size and records the cameras it saw."""

    def __init__(self) -> None:
        self.cameras = []

    def rasterize(self, means, quats, scales, opacities, colors, camera):
        self.cameras.append(camera)
        return SimpleNamespace(rgb=torch.full((camera.height, camera.width, 3), 0.5))


def _norm_bundle(scale: float = 2.0) -> dict:
    return {
        "scene_norm_transfm": np.eye(4, dtype=np.float32),
        "scene_norm_scale": np.float32(scale),
    }


def _val_cams(n: int) -> list[BorrowedCamera]:
    K = np.array([[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1]], np.float32)
    return [BorrowedCamera(f"1_{t:05d}", t, np.eye(4, dtype=np.float32), K) for t in range(n)]


# -- placement ----------------------------------------------------------------


def test_placements_from_bundle_val_scale_k_by_factor_ratio():
    views = place_heldout_views(
        {"factor": np.int64(1)},
        {"train": [], "val": _val_cams(4)},
        {"factor": 2, "max_eval_frames": 3},
        _Dataset(),
    )
    assert views.alignment == {"method": "bundle_val"}
    assert [p.frame_name for p in views.placements] == ["1_00000", "1_00001", "1_00002"]
    camera = views.placements[0].camera(W // 2, H // 2)
    assert (camera.width, camera.height) == (W // 2, H // 2)
    assert camera.K[0, 0] == pytest.approx(50.0)  # bundle factor 1 / eval factor 2


def test_placements_from_dataset_use_the_bundle_scene_norm():
    views = place_heldout_views(
        _norm_bundle(scale=2.0), {"train": [], "val": []}, {}, _Dataset(n=3)
    )
    assert views.alignment["method"] == "none"
    assert [(p.frame_name, p.time_id) for p in views.placements] == [
        ("1_00000", 0),
        ("1_00010", 10),
        ("1_00020", 20),
    ]
    camera = views.placements[0].camera(1, 1)  # dataset cameras keep their own size
    assert (camera.width, camera.height) == (W, H)
    # world->cam of a camera at z=-4 translates by +4; scene-norm scale 2 halves it
    assert camera.viewmat[2, 3] == pytest.approx(2.0)


def test_placements_none_without_val_cameras_or_scene_norm():
    assert place_heldout_views({}, {"train": [], "val": []}, {}, _Dataset()) is None


def test_default_alignment_by_geometry_backend():
    assert default_alignment("dataset") == "none"  # built on the benchmark's own geometry
    assert default_alignment("vggt-long") == "sim3"
    assert default_alignment(None) == "sim3"


def test_default_alignment_is_sim3_for_every_static_scene():
    # a static scene lives in the ground-plane-canonicalized frame, the oracle's included
    assert default_alignment("dataset", kind="static") == "sim3"
    assert default_alignment("vggt-long", kind="static") == "sim3"


# -- alignment for self-reconstructed scenes ------------------------------------


def _orbit_c2w(frame: int) -> np.ndarray:
    """A camera looking around a room: turning, tilting and moving along an arc."""
    ang, tilt = 0.05 * frame, 0.02 * frame
    rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    c2w = np.eye(4)
    c2w[:3, :3] = rz @ rx
    c2w[:3, 3] = [2 * np.cos(ang), 2 * np.sin(ang), 0.02 * frame]
    return c2w


class _PosedDataset(_Dataset):
    """Training camera '0' on an orbit; held-out camera '1' offset from it."""

    def __init__(self, val_frames: list[int]) -> None:
        self.val_frames = val_frames

    def split_frames(self, split=None):
        return [("1", t) for t in self.val_frames]

    def load_camera(self, name):
        cam, frame = name.split("_")
        c2w = _orbit_c2w(int(frame))
        if cam == "1":
            c2w[:3, 3] += [0.3, -0.2, 0.1]
        intr = SimpleNamespace(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2)
        return intr, c2w.astype(np.float32), (W, H)


# A self-reconstruction's frame: the dataset world rotated, shifted and scaled 3x.
S_TRUE, T_TRUE = 3.0, np.array([1.0, -2.0, 0.5])
R_TRUE = np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]])


def _scene_train_cams(frames: list[int]) -> list[BorrowedCamera]:
    K = np.eye(3, dtype=np.float32)
    return [
        BorrowedCamera(
            f"0:{f}",
            i,  # scene time counts keyframes, not frames
            np.linalg.inv(apply_similarity_to_c2w(_orbit_c2w(f), S_TRUE, R_TRUE, T_TRUE)),
            K,
        )
        for i, f in enumerate(frames)
    ]


def test_sim3_places_heldout_cameras_through_the_training_cameras():
    train = _scene_train_cams(list(range(0, 60, 5)))
    views = place_heldout_views(
        {}, {"train": train, "val": []}, {"align": "sim3"}, _PosedDataset([7, 33])
    )
    assert views.alignment["method"] == "sim3"
    assert views.alignment["scale"] == pytest.approx(S_TRUE)
    assert views.alignment["rotation_residual_deg_median"] < 0.05  # float32 poses in, ~0.003
    for p in views.placements:
        _, t_world_cam, _ = _PosedDataset([]).load_camera(p.frame_name)
        expected = np.linalg.inv(
            apply_similarity_to_c2w(t_world_cam.astype(np.float64), S_TRUE, R_TRUE, T_TRUE)
        )
        assert np.allclose(p.camera(W, H).viewmat, expected, atol=1e-4)


def test_sparse_keyframes_render_at_the_nearest_keyframe():
    train = _scene_train_cams([0, 15, 30])  # scene times 0, 1, 2
    views = place_heldout_views(
        {}, {"train": train, "val": []}, {"align": "sim3"}, _PosedDataset([0, 10, 20, 23, 29])
    )
    assert [p.time_id for p in views.placements] == [0, 10, 20, 23, 29]
    assert [p.scene_time for p in views.placements] == [0, 1, 1, 2, 2]


def _unrelated_c2ws(n: int) -> list[np.ndarray]:
    """Scene cameras with no relation to the dataset's."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        c2w = np.eye(4)
        c2w[:3, :3] = q * np.sign(np.linalg.det(q))
        c2w[:3, 3] = rng.normal(size=3) * 5
        out.append(c2w)
    return out


def test_untrustworthy_alignment_is_refused_and_scores_nothing():
    K = np.eye(3, dtype=np.float32)
    unrelated = [
        BorrowedCamera(f"0:{5 * i}", i, np.linalg.inv(c2w), K)
        for i, c2w in enumerate(_unrelated_c2ws(12))
    ]
    cams, cfg = {"train": unrelated, "val": []}, {"align": "sim3", "dataset_root": "spin"}
    with pytest.raises(AlignmentFailed):
        place_heldout_views({}, cams, cfg, _PosedDataset([7]))
    result = score_dynamic_bundle({}, None, cams, cfg, backend=_PosedDataset([7]), device="cpu")
    assert "masked_mpsnr" not in result
    assert result["alignment"]["status"] == "refused" and result["alignment"]["reason"]


# -- static scenes ----------------------------------------------------------------


def _static_train_views(frames) -> list[dict]:
    """A static scene artifact's training views in the self-reconstruction's frame,
    as persisted (float32 poses as lists)."""
    return [
        {
            "frame_key": f"0:{f}",
            "time_id": f,  # a static scene's time is the video frame
            "t_world_cam": apply_similarity_to_c2w(_orbit_c2w(f), S_TRUE, R_TRUE, T_TRUE)
            .astype(np.float32)
            .tolist(),
        }
        for f in frames
    ]


def _cloud():
    z = torch.zeros(1, 3)
    return SimpleNamespace(means=z, quats=z, scales=z, opacities=z[:, 0], colors=z)


def test_static_scene_places_heldout_cameras_through_its_training_views():
    views = place_static_heldout_views(
        _static_train_views(range(0, 60, 5)), {}, _PosedDataset([7, 33])
    )
    assert views.alignment["method"] == "sim3"  # the default for a static scene
    assert views.alignment["scale"] == pytest.approx(S_TRUE, rel=1e-5)
    assert views.alignment["rotation_residual_deg_median"] < 0.05
    for p in views.placements:
        _, t_world_cam, _ = _PosedDataset([]).load_camera(p.frame_name)
        expected = np.linalg.inv(
            apply_similarity_to_c2w(t_world_cam.astype(np.float64), S_TRUE, R_TRUE, T_TRUE)
        )
        assert np.allclose(p.camera(W, H).viewmat, expected, atol=1e-4)
        assert p.scene_time == p.time_id  # no time axis to snap to
    assert "scene_time" not in views.alignment


def test_static_align_none_places_dataset_cameras_untransformed():
    views = place_static_heldout_views([], {"align": "none"}, _Dataset(n=2))
    assert views.alignment == {"method": "none"}
    camera = views.placements[0].camera(1, 1)
    assert (camera.width, camera.height) == (W, H)
    assert camera.viewmat[2, 3] == pytest.approx(4.0)  # world->cam of a camera at z=-4


def test_static_unknown_alignment_is_an_error():
    with pytest.raises(ValueError, match="unknown alignment"):
        place_static_heldout_views([], {"align": "bogus"}, _Dataset())


@pytest.mark.parametrize("n_views", [12, 2])  # unrelated cameras; too few to fit
def test_static_untrustworthy_alignment_is_refused_and_scores_nothing(n_views):
    views = [
        {"frame_key": f"0:{5 * i}", "time_id": 5 * i, "t_world_cam": c2w.tolist()}
        for i, c2w in enumerate(_unrelated_c2ws(n_views))
    ]
    cfg = {"dataset_root": "spin"}
    with pytest.raises(AlignmentFailed):
        place_static_heldout_views(views, cfg, _PosedDataset([7]))
    result = score_static_cloud(_cloud(), views, cfg, backend=_PosedDataset([7]), device="cpu")
    assert "masked_mpsnr" not in result
    assert result["kind"] == "static"
    assert result["alignment"]["status"] == "refused" and result["alignment"]["reason"]


def test_static_score_renders_the_placed_views_and_records_the_alignment(monkeypatch):
    seen = []

    def fake_score(rasterizer, cloud, views, **_):
        seen.extend(views)
        return {"masked_mpsnr": 1.0, "num_views": len(seen)}

    monkeypatch.setattr(nvs, "score_masked_novel_views", fake_score)
    result = score_static_cloud(
        _cloud(),
        _static_train_views(range(0, 60, 5)),
        {"dataset_root": "dycheck_iphone/spin"},
        backend=_PosedDataset([7, 33]),
        device="cpu",
        rasterizer=_Rasterizer(),
    )
    assert result["num_views"] == 2 and len(seen) == 2
    assert (result["kind"], result["dataset"], result["eval_split"]) == ("static", "spin", "val")
    assert result["alignment"]["method"] == "sim3"


def test_export_static_heldout_views_writes_images_and_scores(tmp_path: Path):
    rasterizer = _Rasterizer()
    records = export_static_heldout_views(
        _cloud(),
        _static_train_views(range(0, 60, 5)),
        {},
        tmp_path,
        backend=_PosedDataset([0, 10, 20, 30]),
        rasterizer=rasterizer,
        count=2,
        device="cpu",
    )
    assert [r["time_id"] for r in records] == [0, 30]
    assert len(rasterizer.cameras) == 2
    assert all(r["masked_psnr"] == pytest.approx(6.0206, abs=1e-3) for r in records)
    for r in records:
        for name in r["files"].values():
            assert (tmp_path / name).exists()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["num_heldout_views"] == 4 and index["alignment"]["method"] == "sim3"


def test_dynamic_scorer_returns_empty_when_views_cannot_be_placed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    (tmp_path / "datasets" / "seq").mkdir(parents=True)   # a sequence with no cameras
    ones = np.ones((1, 3), np.float32)
    np.savez(
        tmp_path / "trained_dynamic.npz",
        fg_means=ones,
        fg_quats=np.ones((1, 4), np.float32),
        fg_scales=ones,
        fg_colors=ones,
        fg_opacities=np.ones(1, np.float32),
        fg_motion_coefs=np.ones((1, 2), np.float32),
        motion_rots=np.ones((2, 1, 6), np.float32),
        motion_transls=np.zeros((2, 1, 3), np.float32),
    )
    art = SimpleNamespace(save_dir=str(tmp_path), scene_state={"checkpoint": "trained_dynamic.npz"})
    assert (
        score_render_artifact_masked_nvs_dynamic(art, {"dataset_root": "seq"}, device="cpu")
        == {}
    )


# -- export -------------------------------------------------------------------


def test_pick_evenly_spreads_and_keeps_the_ends():
    assert pick_evenly(722, 4) == [0, 240, 481, 721]
    assert pick_evenly(3, 8) == [0, 1, 2]
    assert pick_evenly(9, 1) == [4]
    assert pick_evenly(0, 5) == [] and pick_evenly(5, 0) == []




def test_export_heldout_views_writes_images_and_scores(tmp_path: Path):
    scene, rasterizer = _Scene(), _Rasterizer()
    records = export_heldout_views(
        scene,
        _norm_bundle(),
        {"train": [], "val": []},
        {"eval_split": "val"},
        tmp_path,
        backend=_Dataset(n=5),
        rasterizer=rasterizer,
        count=3,
        device="cpu",
    )
    assert [r["time_id"] for r in records] == [0, 20, 40]
    assert scene.times == [0, 20, 40]  # each view rendered at its own time
    # flat 0.5 render against a black photo: MSE 0.25 -> 10*log10(4) dB
    assert all(r["masked_psnr"] == pytest.approx(6.0206, abs=1e-3) for r in records)
    for r in records:
        for name in r["files"].values():
            assert (tmp_path / name).exists()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["num_heldout_views"] == 5 and len(index["views"]) == 3


def test_export_heldout_views_refuses_unplaceable_bundles(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot place"):
        export_heldout_views(
            _Scene(),
            {},
            {"train": [], "val": []},
            {},
            tmp_path,
            backend=_Dataset(),
            rasterizer=_Rasterizer(),
            device="cpu",
        )


def test_export_replay_renders_each_training_camera_in_time_order(tmp_path: Path):
    K = np.array([[100.0, 0, 20.0], [0, 100.0, 15.0], [0, 0, 1]], np.float32)
    train = [BorrowedCamera(f"0:{t}", t, np.eye(4, dtype=np.float32), K) for t in (2, 0, 1)]
    scene, rasterizer = _Scene(), _Rasterizer()
    # the raster is the one given (the run's recorded one), not 2x the principal point (40x30)
    meta = export_replay(
        scene, {"train": train}, tmp_path / "replay.mp4", rasterizer=rasterizer,
        raster=(42, 30), fps=10,
    )
    assert meta["time_ids"] == [0, 1, 2] and scene.times == [0, 1, 2]
    assert (meta["width"], meta["height"]) == (42, 30)
    assert all((c.width, c.height) == (42, 30) for c in rasterizer.cameras)
    assert (tmp_path / "replay.mp4").stat().st_size > 0
    assert json.loads((tmp_path / "replay.json").read_text())["fps"] == 10


# -- export CLI ---------------------------------------------------------------


def _save_static_checkpoint(path: Path) -> None:
    ones = np.ones((1, 3), np.float32)
    np.savez(
        path,
        means=ones,
        quats=np.ones((1, 4), np.float32),
        scales=ones,
        opacities=np.ones(1, np.float32),
        colors=ones,
    )


def _render_record(ckpt_dir: Path, kind: str, weights: str) -> None:
    """The record the render stage saves beside its weights: which file, what kind."""
    (ckpt_dir / "artifact.json").write_text(
        json.dumps({"scene_state": {"kind": kind, "checkpoint": weights}}))


def _static_run(tmp_path: Path, config: dict | None = None) -> Path:
    write_manifest(tmp_path, config)
    (tmp_path / "render").mkdir(parents=True)
    (tmp_path / "scene").mkdir()
    _save_static_checkpoint(tmp_path / "render" / "trained_gaussians.npz")
    _render_record(tmp_path / "render", "static", "trained_gaussians.npz")
    scene = {"kind": "static", "train_views": _static_train_views(range(0, 60, 5))}
    for view in scene["train_views"]:
        view.update(width=W, height=H, intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0})
    (tmp_path / "scene" / "artifact.json").write_text(json.dumps(scene))
    return tmp_path


def test_cli_finds_a_static_run_and_aligns_it_by_its_kind(tmp_path: Path):
    run = RunOutputs.open(_static_run(tmp_path))
    ckpt = cli.find_checkpoint(run, "render")
    assert (ckpt.weights.name, ckpt.kind) == ("trained_gaussians.npz", "static")
    assert cli.resolve_alignment(run, "auto", ckpt.kind) == "sim3"  # whatever its geometry
    assert cli.resolve_alignment(run, "none", ckpt.kind) == "none"
    views = cli.read_train_views(run)
    assert len(views) == 12 and set(views[0]) == {"frame_key", "time_id", "t_world_cam"}


def test_cli_checkpoint_weights_and_kind_come_from_its_render_record(tmp_path: Path):
    run = RunOutputs.open(_static_run(tmp_path))
    retrain = tmp_path / "render_retrain"
    retrain.mkdir()
    np.savez(retrain / "bundle.npz", fg_means=np.ones((1, 3), np.float32))
    _render_record(retrain, "dynamic", "bundle.npz")
    ckpt = cli.find_checkpoint(run, "render_retrain")
    assert (ckpt.weights.name, ckpt.kind) == ("bundle.npz", "dynamic")
    with pytest.raises(SystemExit):
        cli.find_checkpoint(run, "render_missing")


def test_cli_dynamic_auto_alignment_reads_the_geometry_backend(tmp_path: Path):
    write_manifest(tmp_path, {"geometry": {"backend_name": "dataset"}})
    assert cli.resolve_alignment(RunOutputs.open(tmp_path), "auto", "dynamic") == "none"


def test_cli_exports_a_static_runs_heldout_views(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))     # the run is named in runs/
    run = _static_run(tmp_path / "runs" / "run")
    monkeypatch.setattr(cli, "build_rasterizer", lambda cfg: _Rasterizer())
    monkeypatch.setattr(cli, "eval_dataset", lambda cfg: _PosedDataset([0, 10, 20]))
    argv = ["export", "run", "--dataset-root", "spin", "--views", "2", "--no-score"]
    monkeypatch.setattr("sys.argv", [*argv, "--device", "cpu"])
    cli.main()
    index = json.loads((run / "render" / "exports" / "heldout" / "index.json").read_text())
    assert index["alignment"]["method"] == "sim3" and len(index["views"]) == 2
    assert not (run / "render" / "exports" / "replay.mp4").exists()
    assert "replay: skipped" in capsys.readouterr().out

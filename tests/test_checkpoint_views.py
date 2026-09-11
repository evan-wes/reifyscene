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

from reifyscene.render.eval.checkpoint_views import (
    export_heldout_views,
    export_replay,
    pick_evenly,
    replay_size,
)
from reifyscene.render.eval.masked_nvs_benchmark import (
    heldout_placements,
    score_render_artifact_masked_nvs_dynamic,
)
from reifyscene.render.gaussian4d.motion_init import BorrowedCamera

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
    placements = heldout_placements(
        {"factor": np.int64(1)},
        {"train": [], "val": _val_cams(4)},
        {"factor": 2, "max_eval_frames": 3},
        _Dataset(),
    )
    assert [p.frame_name for p in placements] == ["1_00000", "1_00001", "1_00002"]
    camera = placements[0].camera(W // 2, H // 2)
    assert (camera.width, camera.height) == (W // 2, H // 2)
    assert camera.K[0, 0] == pytest.approx(50.0)  # bundle factor 1 / eval factor 2


def test_placements_from_dataset_use_the_bundle_scene_norm():
    placements = heldout_placements(
        _norm_bundle(scale=2.0), {"train": [], "val": []}, {}, _Dataset(n=3)
    )
    assert [(p.frame_name, p.time_id) for p in placements] == [
        ("1_00000", 0),
        ("1_00010", 10),
        ("1_00020", 20),
    ]
    camera = placements[0].camera(1, 1)  # dataset cameras keep their own size
    assert (camera.width, camera.height) == (W, H)
    # world->cam of a camera at z=-4 translates by +4; scene-norm scale 2 halves it
    assert camera.viewmat[2, 3] == pytest.approx(2.0)


def test_placements_none_without_val_cameras_or_scene_norm():
    assert heldout_placements({}, {"train": [], "val": []}, {}, _Dataset()) is None


def test_dynamic_scorer_returns_empty_when_views_cannot_be_placed(tmp_path: Path):
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
        score_render_artifact_masked_nvs_dynamic(art, {"dataset_root": str(tmp_path)}, device="cpu")
        == {}
    )


# -- export -------------------------------------------------------------------


def test_pick_evenly_spreads_and_keeps_the_ends():
    assert pick_evenly(722, 4) == [0, 240, 481, 721]
    assert pick_evenly(3, 8) == [0, 1, 2]
    assert pick_evenly(9, 1) == [4]
    assert pick_evenly(0, 5) == [] and pick_evenly(5, 0) == []


def test_replay_size_is_twice_the_principal_point_and_even():
    assert replay_size(np.array([[714.7, 0, 259.0], [0, 710.4, 259.0], [0, 0, 1]])) == (518, 518)
    assert replay_size(np.array([[359.6, 0, 179.4], [0, 359.6, 241.6], [0, 0, 1]])) == (358, 484)


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
    meta = export_replay(
        scene, {"train": train}, tmp_path / "replay.mp4", rasterizer=rasterizer, fps=10
    )
    assert meta["time_ids"] == [0, 1, 2] and scene.times == [0, 1, 2]
    assert (meta["width"], meta["height"]) == (40, 30)
    assert all((c.width, c.height) == (40, 30) for c in rasterizer.cameras)
    assert (tmp_path / "replay.mp4").stat().st_size > 0
    assert json.loads((tmp_path / "replay.json").read_text())["fps"] == 10

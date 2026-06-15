from __future__ import annotations

import torch

from room4d.render.gaussian4d.densify import _quat_to_rotmat
from room4d.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
from room4d.render.gaussian4d.gaussian_field import GaussianField, logit
from room4d.render.gaussian4d.motion import MotionBases
from room4d.render.gaussian4d.transforms import (
    cont_6d_to_rmat,
    quat_multiply,
    rmat_to_cont_6d,
    rotmat_to_quat,
)


def _random_rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def _dynamic_field(n: int = 5, k: int = 2) -> DynamicGaussianField:
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    return DynamicGaussianField(
        means=torch.randn(n, 3),
        quats=quats,
        log_scales=torch.zeros(n, 3),
        logit_opacities=logit(torch.full((n,), 0.9)),
        colors_logit=logit(torch.rand(n, 3).clamp(0.1, 0.9)),
        motion_coefs=torch.zeros(n, k),
    )


def _identity_bases(k: int = 2, t: int = 4) -> MotionBases:
    rots = torch.zeros(k, t, 6)
    rots[..., 0] = 1.0  # 6D identity: x=(1,0,0), y=(0,1,0)
    rots[..., 4] = 1.0
    return MotionBases(rots, transls=torch.zeros(k, t, 3))


def _static_field(n: int = 3) -> GaussianField:
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    return GaussianField(
        means=torch.randn(n, 3),
        quats=quats,
        log_scales=torch.zeros(n, 3),
        logit_opacities=logit(torch.full((n,), 0.8)),
        colors_logit=logit(torch.rand(n, 3).clamp(0.1, 0.9)),
    )


# -- transforms -----------------------------------------------------------

def test_6d_rotation_roundtrip() -> None:
    rot = _random_rotation()
    assert torch.allclose(cont_6d_to_rmat(rmat_to_cont_6d(rot)), rot, atol=1e-5)


def test_rotmat_to_quat_roundtrip() -> None:
    rot = _random_rotation()
    quat = rotmat_to_quat(rot)
    assert torch.allclose(_quat_to_rotmat(quat[None])[0], rot, atol=1e-4)


def test_quat_multiply_identity() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    assert torch.allclose(quat_multiply(identity, q), q, atol=1e-6)


# -- warp / scene ---------------------------------------------------------

def test_warp_identity_bases_is_no_op() -> None:
    fg = _dynamic_field()
    bases = _identity_bases()
    means_t, quats_t = fg.warp(bases, torch.tensor([2]))
    assert torch.allclose(means_t[:, 0], fg.means, atol=1e-5)
    assert torch.allclose(quats_t[:, 0].abs(), fg.get_quats().abs(), atol=1e-4)


def test_warp_pure_translation() -> None:
    fg = _dynamic_field(n=4, k=1)
    rots = torch.zeros(1, 3, 6)
    rots[..., 0] = 1.0
    rots[..., 4] = 1.0
    transls = torch.zeros(1, 3, 3)
    transls[0, 2] = torch.tensor([1.0, 2.0, 3.0])  # frame 2 shifts by (1,2,3)
    bases = MotionBases(rots, transls)
    means_t, _ = fg.warp(bases, torch.tensor([2]))
    assert torch.allclose(means_t[:, 0], fg.means + torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


def test_dynamic_scene_cloud_and_entity_ids() -> None:
    scene = DynamicScene(_dynamic_field(n=5, k=2), _identity_bases(k=2), _static_field(n=3))
    cloud = scene.cloud_at(1)
    assert cloud.means.shape[0] == 8  # 5 fg + 3 bg
    assert scene.num_gaussians == 8
    assert scene.entity_ids().tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_load_borrowed_init_roundtrip(tmp_path) -> None:
    import numpy as np

    from room4d.render.gaussian4d.motion_init import load_borrowed_init

    n_fg, n_bg, k, t = 5, 3, 2, 4
    ident_q = np.tile([1.0, 0.0, 0.0, 0.0], (n_fg, 1)).astype("f4")
    np.savez(
        tmp_path / "bundle.npz",
        fg_means=np.zeros((n_fg, 3), "f4"), fg_quats=ident_q,
        fg_scales=np.zeros((n_fg, 3), "f4"), fg_opacities=np.zeros((n_fg,), "f4"),
        fg_colors=np.zeros((n_fg, 3), "f4"), fg_motion_coefs=np.zeros((n_fg, k), "f4"),
        motion_rots=np.zeros((k, t, 6), "f4"), motion_transls=np.zeros((k, t, 3), "f4"),
        bg_means=np.zeros((n_bg, 3), "f4"), bg_quats=np.tile([1.0, 0, 0, 0], (n_bg, 1)).astype("f4"),
        bg_scales=np.zeros((n_bg, 3), "f4"), bg_opacities=np.zeros((n_bg,), "f4"),
        bg_colors=np.zeros((n_bg, 3), "f4"),
        num_bases=np.asarray(k), factor=np.asarray(1),
        train_w2cs=np.tile(np.eye(4), (2, 1, 1)).astype("f4"), train_Ks=np.tile(np.eye(3), (2, 1, 1)).astype("f4"),
        train_time_ids=np.array([0, 1]), train_frame_names=np.array(["0_00000", "0_00001"]),
        val_w2cs=np.tile(np.eye(4), (2, 1, 1)).astype("f4"), val_Ks=(np.eye(3) * np.array([[2, 2, 1]]).T).astype("f4")[None].repeat(2, 0),
        val_time_ids=np.array([0, 1]), val_frame_names=np.array(["1_00000", "1_00001"]),
    )
    scene, cams = load_borrowed_init(str(tmp_path / "bundle.npz"), device="cpu")
    assert scene.fg.num_gaussians == 5 and scene.bg.num_gaussians == 3
    assert scene.num_frames == 4 and scene.fg.num_bases == 2
    assert len(cams["train"]) == 2 and len(cams["val"]) == 2
    assert cams["val"][0].frame_name == "1_00000" and cams["val"][0].time_id == 0
    assert scene.cloud_at(1).means.shape[0] == 8
    rc = cams["val"][0].render_camera(64, 48, k_scale=0.5)
    assert rc.width == 64 and abs(float(rc.K[0, 0]) - 1.0) < 1e-5  # fx 2.0 * 0.5

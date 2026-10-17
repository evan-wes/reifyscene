"""Unit tests for the Procrustes/clustering motion-init ports (dataset-free).

The end-to-end native init (and its mPSNR bar) is exercised by
``run_render_proof.py dynamic --native``; here we pin the building blocks on
synthetic data.
"""

from __future__ import annotations

import numpy as np
import torch

from reifyscene.lifting.native_track_lift import rotvec_to_rotmat
from reifyscene.lifting.tracks_3d import StaticPoints3D, Tracks3D
from reifyscene.render.gaussian4d.dynamic_field import DynamicGaussianField
from reifyscene.render.gaussian4d.gaussian_field import GaussianField
from reifyscene.render.gaussian4d.motion import MotionBases
from reifyscene.render.gaussian4d.procrustes_init import (
    _interp_masked_tracks,
    _rotvec_to_unitquat,
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    knn,
    sample_initial_bases_centers,
    solve_procrustes,
)
from reifyscene.render.gaussian4d.transforms import cont_6d_to_rmat


def test_solve_procrustes_recovers_known_se3() -> None:
    torch.manual_seed(0)
    src = torch.randn(64, 3)
    R_true = rotvec_to_rotmat(torch.tensor([0.3, -0.2, 0.5]))
    t_true = torch.tensor([1.0, 2.0, -3.0])
    dst = src @ R_true.T + t_true

    (rot_mat, t, s), (err, err_before) = solve_procrustes(src, dst, enforce_se3=True, rot_type="mat")
    assert torch.allclose(rot_mat, R_true, atol=1e-4)
    assert torch.allclose(t, t_true, atol=1e-4)
    assert err < 1e-4 < err_before

    # 6D path decodes back to the same rotation.
    (rot_6d, _t, _s), _ = solve_procrustes(src, dst, enforce_se3=True, rot_type="6d")
    assert rot_6d.shape == (6,)
    assert torch.allclose(cont_6d_to_rmat(rot_6d), R_true, atol=1e-4)


def test_knn_on_unit_line() -> None:
    pts = torch.tensor([[float(i), 0.0, 0.0] for i in range(5)])
    dists, idx = knn(pts, k=1)
    assert dists.shape == (5, 1) and idx.shape == (5, 1)
    # Every point's nearest neighbor is exactly one unit away.
    assert np.allclose(dists[:, 0], 1.0, atol=1e-5)


def test_rotvec_to_unitquat_identity_and_z180() -> None:
    assert torch.allclose(_rotvec_to_unitquat(torch.zeros(1, 3))[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    q = _rotvec_to_unitquat(torch.tensor([[0.0, 0.0, np.pi]]))[0]  # 180deg about z -> (0,0,0,1)
    assert torch.allclose(q.abs(), torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.isclose(q.norm(), torch.tensor(1.0), atol=1e-6)


def test_interp_masked_tracks_fills_gap() -> None:
    # One track, 5 frames, only endpoints visible -> linear interp between them.
    xyz = torch.zeros(1, 5, 3)
    xyz[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    xyz[0, 4] = torch.tensor([4.0, 8.0, -4.0])
    vis = torch.tensor([[True, False, False, False, True]])
    out = _interp_masked_tracks(xyz, vis)
    assert torch.allclose(out[0, 2], torch.tensor([2.0, 4.0, -2.0]), atol=1e-5)
    assert torch.allclose(out[0, 1], torch.tensor([1.0, 2.0, -1.0]), atol=1e-5)


def _two_cluster_tracks(n_per: int = 150, num_frames: int = 6) -> Tracks3D:
    """Two rigid groups with distinct constant velocities (+x vs +y)."""
    torch.manual_seed(1)
    a0 = torch.rand(n_per, 3)
    b0 = torch.rand(n_per, 3) + torch.tensor([5.0, 0.0, 0.0])
    ts = torch.arange(num_frames).float()
    a = a0[:, None, :] + ts[None, :, None] * torch.tensor([0.2, 0.0, 0.0])
    b = b0[:, None, :] + ts[None, :, None] * torch.tensor([0.0, 0.2, 0.0])
    xyz = torch.cat([a, b], dim=0)
    n = xyz.shape[0]
    return Tracks3D(
        xyz=xyz,
        visibles=torch.ones(n, num_frames, dtype=torch.bool),
        invisibles=torch.zeros(n, num_frames, dtype=torch.bool),
        confidences=torch.ones(n, num_frames),
        colors=torch.rand(n, 3),
    )


def test_sample_initial_bases_centers_two_clusters() -> None:
    tracks = _two_cluster_tracks()
    centers, k, labels = sample_initial_bases_centers(cano_t=0, tracks=tracks, num_bases=2, seed=0)
    assert k == 2
    assert centers.shape == (1, 2, 3)
    assert set(labels.tolist()) == {0, 1}


def test_init_motion_params_and_fg_bg_shapes() -> None:
    tracks = _two_cluster_tracks()
    bases, motion_coefs, filtered = init_motion_params_with_procrustes(
        tracks, num_bases=2, cano_t=0, seed=0
    )
    assert isinstance(bases, MotionBases)
    assert bases.num_bases == 2 and bases.num_frames == 6
    assert bases.rots.shape == (2, 6, 6) and bases.transls.shape == (2, 6, 3)
    assert motion_coefs.shape == (filtered.num_tracks, 2)
    assert torch.isfinite(bases.rots).all() and torch.isfinite(bases.transls).all()

    fg = init_fg_from_tracks_3d(cano_t=0, tracks=filtered, motion_coefs=motion_coefs)
    assert isinstance(fg, DynamicGaussianField)
    assert fg.num_gaussians == filtered.num_tracks and fg.num_bases == 2
    for p in (fg.means, fg.log_scales, fg.logit_opacities, fg.colors_logit, fg.motion_coefs):
        assert torch.isfinite(p).all()

    pts = StaticPoints3D(
        xyz=torch.rand(50, 3),
        normals=torch.nn.functional.normalize(torch.rand(50, 3), dim=-1),
        colors=torch.rand(50, 3),
    )
    bg = init_bg(pts)
    assert isinstance(bg, GaussianField)
    assert bg.num_gaussians == 50
    assert torch.isfinite(bg.get_quats()).all()
    for p in (bg.means, bg.log_scales, bg.logit_opacities, bg.colors_logit):
        assert torch.isfinite(p).all()

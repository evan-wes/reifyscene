"""Unit tests for the native track-lift math ports (dataset-free).

The end-to-end lift against real data is exercised by
``scripts/check_native_lift.py``; here we pin the pure tensor ops that the lift
is built from, plus the lifted-observation dataclasses.
"""

from __future__ import annotations

import numpy as np
import torch

from room4d.lifting.native_track_lift import (
    compute_scene_norm,
    normal_from_depth,
    normalize_coords,
    parse_tapir_track_info,
    rotvec_to_rotmat,
    rt_to_mat4,
)
from room4d.lifting.tracks_3d import StaticPoints3D, Tracks3D


def test_parse_tapir_track_info_buckets() -> None:
    # occlusion/dist are logits; sigmoid(-10)~0 -> visibility~1, confidence~1.
    occ = torch.tensor([-10.0, 10.0, -10.0])   # visible, occluded, visible
    dist = torch.tensor([-10.0, -10.0, 10.0])  # confident, confident, unconfident
    visible, invisible, confidence = parse_tapir_track_info(occ, dist)
    assert visible.tolist() == [True, False, False]
    assert invisible.tolist() == [False, True, False]
    # confidence is zeroed where neither visible nor invisible holds (3rd entry).
    assert confidence[0] > 0.5 and confidence[1] > 0.5 and confidence[2] == 0.0


def test_normalize_coords_corners() -> None:
    coords = torch.tensor([[0.0, 0.0], [9.0, 4.0]])  # corners of a 10x5 image
    out = normalize_coords(coords, h=5, w=10)
    assert torch.allclose(out[0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(out[1], torch.tensor([1.0, 1.0]))


def test_rotvec_to_rotmat_identity_and_z90() -> None:
    assert torch.allclose(rotvec_to_rotmat(torch.zeros(3)), torch.eye(3))
    R = rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.pi / 2]))
    # 90deg about +z maps x->y, y->-x.
    assert torch.allclose(R @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-6)
    assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-6)


def test_rt_to_mat4_block() -> None:
    R = rotvec_to_rotmat(torch.tensor([0.1, -0.2, 0.3]))
    t = torch.tensor([1.0, 2.0, 3.0])
    M = rt_to_mat4(R, t)
    assert M.shape == (4, 4)
    assert torch.allclose(M[:3, :3], R)
    assert torch.allclose(M[:3, 3], t)
    assert torch.allclose(M[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_normal_from_depth_fronto_parallel_plane() -> None:
    # Constant-depth plane under identity extrinsics -> interior normals are ±z.
    H, W = 8, 8
    depth = torch.full((H, W), 2.0)
    K = torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    normals = normal_from_depth(depth, K, torch.eye(4))
    interior = normals[2:H - 2, 2:W - 2]
    assert torch.allclose(interior[..., 0], torch.zeros_like(interior[..., 0]), atol=1e-5)
    assert torch.allclose(interior[..., 1], torch.zeros_like(interior[..., 1]), atol=1e-5)
    assert torch.allclose(interior[..., 2].abs(), torch.ones_like(interior[..., 2]), atol=1e-5)


def test_compute_scene_norm_aligns_up_and_centers() -> None:
    # Identity-rotation cameras: world up (-y of w2c row 1) maps to +z; the
    # transform centers the scene (transfm @ [center, 1] -> origin).
    w2cs = torch.eye(4).repeat(5, 1, 1)
    g = torch.linspace(-1.0, 1.0, 50)
    sample = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 1, 3)
    sample = sample + torch.tensor([10.0, 20.0, 30.0])  # offset center
    norm = compute_scene_norm(w2cs, sample)

    R, t = norm.transfm[:3, :3], norm.transfm[:3, 3]
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
    original_up = torch.tensor([0.0, -1.0, 0.0])
    assert torch.allclose(R @ original_up, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    center = sample.mean((0, 1))
    mapped = norm.transfm @ torch.cat([center, torch.ones(1)])
    assert torch.allclose(mapped[:3], torch.zeros(3), atol=1e-4)
    assert norm.scale > 0.0


def test_tracks3d_check_sizes_and_filter() -> None:
    N, T = 6, 4
    tracks = Tracks3D(
        xyz=torch.zeros(N, T, 3),
        visibles=torch.zeros(N, T, dtype=torch.bool),
        invisibles=torch.zeros(N, T, dtype=torch.bool),
        confidences=torch.zeros(N, T),
        colors=torch.zeros(N, 3),
    )
    assert tracks.check_sizes()
    keep = torch.tensor([True, False, True, False, True, False])
    filtered = tracks.filter_valid(keep)
    assert filtered.num_tracks == 3 and filtered.num_frames == T
    assert filtered.check_sizes()
    # A mismatched field is caught.
    bad = Tracks3D(
        xyz=torch.zeros(N, T, 3),
        visibles=torch.zeros(N, T + 1, dtype=torch.bool),
        invisibles=torch.zeros(N, T, dtype=torch.bool),
        confidences=torch.zeros(N, T),
        colors=torch.zeros(N, 3),
    )
    assert not bad.check_sizes()


def test_static_points3d_check_sizes_and_filter() -> None:
    pts = StaticPoints3D(xyz=torch.zeros(5, 3), normals=torch.zeros(5, 3), colors=torch.zeros(5, 3))
    assert pts.check_sizes()
    kept = pts.filter_valid(torch.tensor([True, True, False, False, True]))
    assert kept.num_points == 3 and kept.check_sizes()
    assert not StaticPoints3D(
        xyz=torch.zeros(5, 3), normals=torch.zeros(4, 3), colors=torch.zeros(5, 3)
    ).check_sizes()

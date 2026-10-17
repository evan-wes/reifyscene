"""Ownership: every Gaussian owned from seeding by the entity it was seeded on
(docs/design/2026-10-17-removal-by-ownership.md).

What these pin is what would go wrong quietly: a seed at a mask's edge given to one side, a
background point taken from a pixel an entity or a dropped tracklet claims, an owner row that
stops following its Gaussian through densification or a removal, the unaddressed owner mixed
into an entity's channel, a loss that counts unlabelled pixels or cannot tell a wrong owner
from a right one, an owned scene trained without its id maps -- and, on CUDA, whether the loss
moves a Gaussian off another entity's pixels and whether the drift check sees one there.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reifyscene.graph.id_maps import BACKGROUND, IGNORE, UNADDRESSED
from reifyscene.graph.renderable_handle import drop_gaussians, owner_handles
from reifyscene.lifting.native_track_lift import (
    CameraRecords,
    FrameStack,
    gate_tracks_by_motion,
    lift_bkgd_points,
    lift_tracks_3d,
    owner_purity,
)
from reifyscene.lifting.tracks_3d import StaticPoints3D, Tracks3D
from reifyscene.render.gaussian4d.densify import DensificationController, DensifyConfig
from reifyscene.render.gaussian4d.dynamic_field import FG_PARAM_NAMES
from reifyscene.render.gaussian4d.dynamic_trainer import DynamicGaussianTrainer, DynamicTrainConfig
from reifyscene.render.gaussian4d.gaussian_field import OWNER_BUFFER
from reifyscene.render.gaussian4d.losses import owner_loss
from reifyscene.render.gaussian4d.motion_init import bundle_to_scene, dump_bundle
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rasterizer tests need gsplat and a CUDA device"
)

A, B = "object:cam_00:sam2:cam_00:0", "human:cam_00:sam2:cam_00:1"
TABLE = [None, A, B]
S, T = 16, 3                       # raster side, frames


def _stack(owners: torch.Tensor, fg: torch.Tensor | None = None) -> FrameStack:
    """``T`` frames of a flat wall at depth 1, seen by a static camera of focal ``S``."""
    K = torch.tensor([[S, 0.0, S / 2], [0.0, S, S / 2], [0.0, 0.0, 1.0]])
    cams = CameraRecords(frame_names=[f"cam_00:{i}" for i in range(T)],
                         time_ids=torch.arange(T), w2cs=torch.eye(4).repeat(T, 1, 1),
                         Ks=K.repeat(T, 1, 1))
    return FrameStack(cameras=cams, depths=torch.ones(T, S, S),
                      rgbs=torch.full((T, S, S, 3), 128, dtype=torch.uint8),
                      fg=torch.ones(T, S, S) if fg is None else fg,
                      valid=torch.ones(T, S, S), owners=owners)


def _halves() -> torch.Tensor:
    """Entity 1 on the left half of every frame, entity 2 on the right."""
    owners = torch.ones(T, S, S, dtype=torch.long)
    owners[:, :, S // 2:] = 2
    return owners


# -- seeding ----------------------------------------------------------------------------------

def test_a_seed_at_a_mask_edge_belongs_to_neither_side() -> None:
    pure = owner_purity(_halves())
    edge = [c for c in range(S) if not pure[0, 5, c]]
    assert edge == [6, 7, 8, 9]                       # within 2 px of the boundary
    ignored = _halves()
    ignored[0, 0, 0] = IGNORE
    assert not owner_purity(ignored)[0, 0, 0]


def test_tracks_are_owned_by_their_query_pixel_and_edge_tracks_are_dropped() -> None:
    stack = _stack(_halves())
    ys, xs = np.mgrid[0:S, 0:S]
    grid = np.stack([xs.ravel(), ys.ravel(), np.full(S * S, -10.0), np.full(S * S, -10.0)], 1)
    tracks = lift_tracks_3d(stack, "", 1, num_samples=T * S * S, rng=np.random.default_rng(0),
                            pair_loader=lambda i, j: grid.astype(np.float32))
    assert tracks.check_sizes()
    assert tracks.num_tracks == T * S * 12            # 4 edge columns of 16 dropped
    assert sorted(set(tracks.owners.tolist())) == [1, 2]
    # each owner is where its track is: the left half of the wall is entity 1
    x = tracks.xyz[:, 0, 0] * S + S / 2               # depth 1, focal S
    assert bool((tracks.owners[x < S / 2] == 1).all())
    assert bool((tracks.owners[x > S / 2] == 2).all())


def test_background_points_come_only_from_background_pixels() -> None:
    owners = _halves() * 0
    owners[:, :, S // 2:] = 2
    owners[:, :, 0] = IGNORE                           # padding
    owners[:, 0, :] = UNADDRESSED                      # a dropped tracklet, not in fg here
    fg = (owners == 2).float()
    pts = lift_bkgd_points(_stack(owners, fg=fg), num_samples=T * S * S,
                           rng=np.random.default_rng(0))
    assert pts.check_sizes() and pts.num_points == T * (S - 1) * (S // 2 - 1)
    assert bool((pts.owners == BACKGROUND).all())


def test_a_static_track_becomes_background_and_keeps_its_owner() -> None:
    moving = torch.stack([torch.tensor([[float(t), 0.0, 0.0] for t in range(T)]),
                          torch.zeros(T, 3)])           # track 0 moves, track 1 does not
    tracks = Tracks3D(xyz=moving, visibles=torch.ones(2, T, dtype=torch.bool),
                      invisibles=torch.zeros(2, T, dtype=torch.bool),
                      confidences=torch.ones(2, T), colors=torch.zeros(2, 3),
                      owners=torch.tensor([1, 2]))
    bg = StaticPoints3D(xyz=torch.zeros(1, 3), normals=torch.zeros(1, 3),
                        colors=torch.zeros(1, 3), owners=torch.tensor([BACKGROUND]))
    fg, merged = gate_tracks_by_motion(tracks, bg, threshold=0.5)
    assert fg.owners.tolist() == [1]
    assert merged.owners.tolist() == [BACKGROUND, 2]


# -- the fields -------------------------------------------------------------------------------

def _bundle(g: int = 6, nb: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    rots = np.zeros((2, T, 6), np.float32)
    rots[:, :, 0] = rots[:, :, 4] = 1.0
    return {
        "fg_means": np.stack([np.linspace(-1, 1, g), np.zeros(g), np.zeros(g)], 1).astype(
            np.float32),
        "fg_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (g, 1)),
        "fg_scales": np.full((g, 3), np.log(0.001), np.float32),
        "fg_colors": np.zeros((g, 3), np.float32),
        "fg_opacities": np.full((g,), 4.0, np.float32),
        "fg_motion_coefs": np.zeros((g, 2), np.float32),
        "fg_owner": np.array([1, 1, 2, 2, UNADDRESSED, 0][:g], np.int32),
        "motion_rots": rots, "motion_transls": np.zeros((2, T, 3), np.float32),
        "bg_means": rng.normal(size=(nb, 3)).astype(np.float32) + 10.0,
        "bg_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (nb, 1)),
        "bg_scales": np.full((nb, 3), np.log(0.001), np.float32),
        "bg_colors": np.zeros((nb, 3), np.float32),
        "bg_opacities": np.full((nb,), 4.0, np.float32),
        "bg_owner": np.array([0, 0, 2, 0][:nb], np.int32),
        "owner_node_ids": np.array(["", A, B]),
        "train_w2cs": np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
        "train_Ks": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
        "train_time_ids": np.arange(T),
        "train_frame_names": np.array([f"cam_00:{i}" for i in range(T)]),
    }


def test_owners_follow_their_gaussian_through_split_clone_and_prune() -> None:
    scene, _ = bundle_to_scene(_bundle(), device="cpu")
    fg = scene.fg
    with torch.no_grad():
        fg.log_scales[2] = np.log(0.1)              # big: 2 (entity B) will split
        fg.logit_opacities[0] = -10.0               # transparent: 0 (entity A) is pruned
    opt = torch.optim.Adam([{"params": [getattr(fg, n)], "name": "fg." + n}
                            for n in FG_PARAM_NAMES])
    ctrl = DensificationController(fg, opt, DensifyConfig(scene_extent=2.0),
                                   param_names=FG_PARAM_NAMES, group_prefix="fg.",
                                   buffer_names=(OWNER_BUFFER,))
    ctrl.grad2d_accum[:] = 0.0
    ctrl.grad2d_accum[[2, 4]] = 1.0                 # 2 splits, 4 (unaddressed) clones
    ctrl.count[:] = 1.0
    stats = ctrl.densify_and_prune(step=1)
    assert (stats["split"], stats["cloned"], stats["pruned"]) == (1, 1, 1)
    assert fg.owner.shape[0] == fg.means.shape[0] == 6 - 1 - 1 + 1 + 2
    assert sorted(fg.owner.tolist()) == sorted([1, 2, UNADDRESSED, 0] + [UNADDRESSED, 2, 2])
    assert "owner" not in dict(fg.named_parameters())       # never optimised


def test_owners_round_trip_through_the_checkpoint_and_a_removal() -> None:
    scene, cams = bundle_to_scene(_bundle(), device="cpu")
    assert scene.has_owners and scene.owner_node_ids == TABLE
    dump = dump_bundle(scene, cams)
    assert dump["fg_owner"].tolist() == [1, 1, 2, 2, UNADDRESSED, 0]
    handles = owner_handles(dump)
    assert set(handles) == {A, B}                   # background and unaddressed are no node's
    assert handles[A][0].tolist() == [0, 1] and handles[B][1].tolist() == [2]
    edited = drop_gaussians(dump, handles[A][0], bg_indices=handles[A][1])
    assert edited["fg_owner"].tolist() == [2, 2, UNADDRESSED, 0]     # rows stay with Gaussians
    assert owner_handles(edited)[A][0].size == 0


def test_the_unaddressed_owner_is_the_last_channel() -> None:
    scene, _ = bundle_to_scene(_bundle(), device="cpu")
    channel, n = scene.owner_channels()
    assert n == 4                                   # background, A, B, unaddressed
    assert channel[:6].tolist() == [1, 1, 2, 2, 3, 0]


def test_a_scene_without_owners_has_none() -> None:
    plain = {k: v for k, v in _bundle().items() if "owner" not in k}
    scene, cams = bundle_to_scene(plain, device="cpu")
    assert not scene.has_owners and "fg_owner" not in dump_bundle(scene, cams)
    with pytest.raises(ValueError, match="owners"):
        owner_handles(plain)


# -- the loss ---------------------------------------------------------------------------------

def test_the_owner_loss_is_zero_for_the_right_owner_and_one_for_another() -> None:
    ids = torch.tensor([[1, 2], [0, IGNORE]])
    known = ids < UNADDRESSED
    right = torch.nn.functional.one_hot(torch.where(known, ids, 0), 4).float()
    assert float(owner_loss(right, ids, known, quantile=1.0)) == 0.0
    wrong = right.clone()
    wrong[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])          # B drawn on A's pixel
    assert float(owner_loss(wrong, ids, known, quantile=1.0)) == pytest.approx(1 / 3)
    wrong[1, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])          # an ignore pixel: no loss
    assert float(owner_loss(wrong, ids, known, quantile=1.0)) == pytest.approx(1 / 3)
    thin = right * 0.5                     # half transparent: half the coverage missing
    assert float(owner_loss(thin, ids, known, quantile=1.0)) == pytest.approx(0.25)
    assert owner_loss(right, ids, torch.zeros_like(known), quantile=1.0) is None


def test_an_owned_scene_is_not_trained_without_its_id_maps() -> None:
    scene, cams = bundle_to_scene(_bundle(), device="cpu")
    cfg = DynamicTrainConfig(iters=1, lambda_depth=0.0, lambda_depth_grad=0.0, lambda_mask=0.0)
    cam = cams["train"][0].render_camera(8, 8)
    with pytest.raises(ValueError, match="id map"):
        DynamicGaussianTrainer(scene, cfg, device="cpu").fit(
            [(cam, torch.zeros(8, 8, 3), 0)])


# -- CUDA: does the loss move Gaussians, and does the drift check see them? -------------------

def _two_blobs():
    """A owns the Gaussian at y=+0.3 and B the one at y=-0.3; the id map gives the right half
    of the image (where both render) to B and the left half to A."""
    b = _bundle(g=2, nb=1)
    b["fg_means"] = np.array([[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]], np.float32)
    b["fg_scales"] = np.full((2, 3), np.log(0.12), np.float32)
    b["fg_owner"] = np.array([1, 2], np.int32)
    b["bg_owner"] = np.array([0], np.int32)
    b["bg_opacities"] = np.full((1,), -20.0, np.float32)
    scene, _ = bundle_to_scene(b, device="cuda")
    cam = RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 96, 64)
    ids = torch.full((64, 96), 2, dtype=torch.long, device="cuda")
    ids[:, :20] = 1
    return scene, cam, ids


def _a_share_on_b(scene, cam, ids) -> float:
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    channel, n = scene.owner_channels()
    with torch.no_grad():
        c = scene.cloud_at(0)
        f = GsplatRasterizer("cuda").rasterize(
            c.means, c.quats, c.scales, c.opacities, c.colors, cam,
            features=torch.nn.functional.one_hot(channel, n).float()).features
    return float(f[..., 1][ids == 2].sum())


@requires_gsplat
def test_the_owner_loss_moves_a_gaussian_off_another_entitys_pixels() -> None:
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    scene, cam, ids = _two_blobs()
    r = GsplatRasterizer("cuda")
    with torch.no_grad():
        c = scene.cloud_at(0)
        gt = r.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, cam).rgb
    before = _a_share_on_b(scene, cam, ids)
    assert before > 50.0
    cfg = DynamicTrainConfig(iters=300, lambda_depth=0.0, lambda_depth_grad=0.0,
                             lambda_mask=0.0, lambda_owner=5.0, owner_drift_every=0,
                             lr_means=1e-2)
    DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r).fit(
        [(cam, gt, 0)], targets=[{"ids": ids}])
    assert _a_share_on_b(scene, cam, ids) < 0.5 * before


@requires_gsplat
def test_the_drift_check_flags_a_gaussian_on_another_entitys_pixels() -> None:
    scene, cam, ids = _two_blobs()
    cfg = DynamicTrainConfig(iters=1, lambda_depth=0.0, lambda_depth_grad=0.0, lambda_mask=0.0)
    drift = DynamicGaussianTrainer(scene, cfg, device="cuda").owner_drift([(cam, 0, ids)])
    assert drift["flagged"] >= 1 and drift["pairs"].get("1->2", 0) >= 1

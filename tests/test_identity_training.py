"""Per-Gaussian identity training (I2 of the per-Gaussian identity scope).

What these pin is what would go wrong quietly: identity rows that stop following their
Gaussians through densification, fg/bg identity concatenated out of order, ignore pixels
leaking into the loss, a checkpoint that drops the classifier, and -- on CUDA -- whether the
loss can actually separate two things at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reifyscene.graph.id_maps import IGNORE
from reifyscene.render.gaussian4d.densify import DensificationController, DensifyConfig
from reifyscene.render.gaussian4d.dynamic_field import FG_PARAM_NAMES
from reifyscene.render.gaussian4d.dynamic_trainer import (
    DynamicGaussianTrainer,
    DynamicTrainConfig,
    knn_indices,
)
from reifyscene.render.gaussian4d.gaussian_field import IDENTITY_PARAM
from reifyscene.render.gaussian4d.motion_init import bundle_to_scene, dump_bundle
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rasterizer tests need gsplat and a CUDA device"
)

G, NB, T, K = 6, 4, 3, 2
NODES = [None, "object:cam_00:sam2:cam_00:0", "human:cam_00:sam2:cam_00:1"]


def _bundle(g: int = G, nb: int = NB, *, means=None, scale: float = 0.05) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    rots = np.zeros((K, T, 6), np.float32)
    rots[:, :, 0] = 1.0
    rots[:, :, 4] = 1.0            # 6D identity rotation: a static foreground
    fg_means = rng.normal(size=(g, 3)).astype(np.float32) if means is None else means
    return {
        "fg_means": fg_means,
        "fg_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (g, 1)),
        "fg_scales": np.full((g, 3), np.log(scale), np.float32),
        "fg_colors": np.zeros((g, 3), np.float32),
        "fg_opacities": np.full((g,), 4.0, np.float32),
        "fg_motion_coefs": np.zeros((g, K), np.float32),
        "motion_rots": rots,
        "motion_transls": np.zeros((K, T, 3), np.float32),
        "bg_means": rng.normal(size=(nb, 3)).astype(np.float32) + 10.0,
        "bg_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (nb, 1)),
        "bg_scales": np.full((nb, 3), np.log(scale), np.float32),
        "bg_colors": np.zeros((nb, 3), np.float32),
        "bg_opacities": np.full((nb,), 4.0, np.float32),
        "train_w2cs": np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
        "train_Ks": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
        "train_time_ids": np.arange(T),
        "train_frame_names": np.array([f"cam_00:{i}" for i in range(T)]),
    }


def _scene(device: str = "cpu", **kw):
    scene, cams = bundle_to_scene(_bundle(**kw), device=device)
    scene.enable_identity(20, NODES, seed=0)
    return scene, cams


# -- the scene --------------------------------------------------------------------------------

def test_identity_is_on_both_fields_and_the_classifier_matches_the_classes() -> None:
    scene, _ = _scene()
    assert scene.fg.identity.shape == (G, 20)
    assert scene.bg.identity.shape == (NB, 20)
    assert scene.identity_classifier.in_features == 20
    assert scene.identity_classifier.out_features == len(NODES)


def test_cloud_features_are_foreground_then_background_like_every_other_param() -> None:
    scene, _ = _scene()
    cloud = scene.cloud_at(1)
    assert cloud.features.shape == (G + NB, 20)
    assert torch.equal(cloud.features[:G], scene.fg.identity)
    assert torch.equal(cloud.features[G:], scene.bg.identity)
    assert torch.equal(cloud.means[G:], scene.bg.means)       # same order as the geometry


def test_identity_on_one_field_only_is_refused() -> None:
    scene, _ = bundle_to_scene(_bundle(), device="cpu")
    scene.fg.init_identity(20)
    with pytest.raises(ValueError, match="one field"):
        scene.cloud_at(0)


def test_a_scene_without_identity_renders_no_features() -> None:
    scene, _ = bundle_to_scene(_bundle(), device="cpu")
    assert not scene.has_identity
    assert scene.cloud_at(0).features is None


# -- densification ----------------------------------------------------------------------------

def test_identity_follows_its_gaussian_through_split_clone_and_prune() -> None:
    """The densifier inherits any listed param from the parent. Pin that identity is listed
    and that rows land where their Gaussians do -- a shifted row would silently hand one
    entity's identity to another's geometry."""
    means = np.stack([np.linspace(-1, 1, G), np.zeros(G), np.zeros(G)], 1).astype(np.float32)
    scene, _ = _scene(means=means, scale=0.001)
    fg = scene.fg
    with torch.no_grad():
        fg.identity.copy_(torch.arange(G, dtype=torch.float32)[:, None].repeat(1, 20))
        fg.log_scales[1] = np.log(0.1)              # big: 1 will split
        fg.logit_opacities[4] = -10.0               # transparent: 4 will be pruned
    opt = torch.optim.Adam([{"params": [getattr(fg, n)], "name": "fg." + n}
                            for n in FG_PARAM_NAMES + (IDENTITY_PARAM,)])
    ctrl = DensificationController(fg, opt, DensifyConfig(scene_extent=2.0),
                                   param_names=FG_PARAM_NAMES + (IDENTITY_PARAM,),
                                   group_prefix="fg.")
    ctrl.grad2d_accum[:] = 0.0
    ctrl.grad2d_accum[[1, 3]] = 1.0                 # 1 splits (big), 3 clones (small)
    ctrl.count[:] = 1.0
    stats = ctrl.densify_and_prune(step=1)
    assert (stats["split"], stats["cloned"], stats["pruned"]) == (1, 1, 1)

    tags = fg.identity[:, 0].detach().round().long().tolist()
    assert fg.identity.shape[0] == fg.means.shape[0] == G - 1 - 1 + 1 + 2
    assert sorted(tags) == sorted([0, 2, 3, 5] + [3] + [1, 1])
    assert all(opt.param_groups[i]["params"][0] is getattr(fg, g["name"][3:])
               for i, g in enumerate(opt.param_groups))


# -- the loss ---------------------------------------------------------------------------------

def _photo_cfg(**kw) -> DynamicTrainConfig:
    """These scenes have no depth or masks: identity is trained on photos alone."""
    return DynamicTrainConfig(lambda_depth=0.0, lambda_depth_grad=0.0, lambda_mask=0.0, **kw)


def _trainer(scene):
    return DynamicGaussianTrainer(scene, _photo_cfg(iters=1), device="cpu")


def test_ignore_pixels_carry_no_loss_and_no_gradient() -> None:
    scene, _ = _scene()
    tr = _trainer(scene)
    feats = torch.randn((4, 5, 20), requires_grad=True)
    id_map = torch.full((4, 5), IGNORE, dtype=torch.long)
    id_map[0, 0], id_map[1, 1] = 1, 2
    tr.identity_loss(feats, id_map).backward()
    touched = feats.grad.abs().sum(-1) > 0
    assert touched[0, 0] and touched[1, 1]
    assert int(touched.sum()) == 2


def test_the_loss_is_normalised_by_log_num_classes() -> None:
    scene, _ = _scene()
    tr = _trainer(scene)
    with torch.no_grad():
        scene.identity_classifier.weight.zero_()
        scene.identity_classifier.bias.zero_()     # uniform prediction: CE = log(C)
    loss = tr.identity_loss(torch.zeros((2, 2, 20)), torch.ones((2, 2), dtype=torch.long))
    assert abs(float(loss.detach()) - 1.0) < 1e-6


def test_a_frame_with_nothing_labelled_contributes_nothing() -> None:
    scene, _ = _scene()
    tr = _trainer(scene)
    assert tr.identity_loss(torch.zeros((2, 2, 20)), torch.full((2, 2), IGNORE)) is None


def test_identity_without_any_id_map_is_refused() -> None:
    scene, _ = _scene()
    cam = RenderCamera(viewmat=np.eye(4, dtype=np.float32), K=np.eye(3, dtype=np.float32),
                       width=4, height=4)
    with pytest.raises(ValueError, match="no training view has an id map"):
        _trainer(scene).fit([(cam, torch.zeros((4, 4, 3)), 0)])


def test_identity_params_have_their_own_optimizer_groups() -> None:
    scene, _ = _scene()
    names = {g["name"] for g in _trainer(scene).optimizer.param_groups}
    assert {"fg.identity", "bg.identity", "identity_classifier"} <= names


# -- the checkpoint ---------------------------------------------------------------------------

def test_identity_round_trips_through_the_checkpoint() -> None:
    scene, cams = _scene()
    loaded, _ = bundle_to_scene(dump_bundle(scene, cams), device="cpu")
    assert loaded.has_identity
    assert torch.equal(loaded.fg.identity, scene.fg.identity)
    assert torch.equal(loaded.bg.identity, scene.bg.identity)
    assert torch.equal(loaded.identity_classifier.weight, scene.identity_classifier.weight)
    assert loaded.identity_node_ids == NODES
    assert torch.allclose(loaded.gaussian_identity_probs(), scene.gaussian_identity_probs())


def test_a_checkpoint_without_identity_is_unchanged() -> None:
    scene, cams = bundle_to_scene(_bundle(), device="cpu")
    dump = dump_bundle(scene, cams)
    assert not any("identity" in k for k in dump)
    assert not bundle_to_scene(dump, device="cpu")[0].has_identity


# -- CUDA: can the loss separate two things at all? -------------------------------------------

@requires_gsplat
def test_one_batch_overfit_separates_two_gaussians() -> None:
    """Two blobs, each labelled its own class where it is opaque. The done-when for I2:
    classification loss to about zero, and each Gaussian's own vector classified correctly."""
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    means = np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], np.float32)
    scene, _ = bundle_to_scene(_bundle(g=2, nb=1, means=means, scale=0.12), device="cuda")
    with torch.no_grad():
        scene.bg.logit_opacities.fill_(-20.0)      # keep the background out of the picture
    scene.enable_identity(20, NODES, seed=0)
    cam = RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 96, 64)

    r = GsplatRasterizer("cuda")
    id_map = torch.full((64, 96), IGNORE, dtype=torch.long, device="cuda")
    with torch.no_grad():
        for k in range(2):
            c = scene.cloud_at(0)
            op = torch.zeros_like(c.opacities)
            op[k] = c.opacities[k]
            alpha = r.rasterize(c.means, c.quats, c.scales, op, c.colors, cam).alpha
            id_map[alpha > 0.5] = k + 1
        gt = r.rasterize(*[getattr(scene.cloud_at(0), n) for n in
                           ("means", "quats", "scales", "opacities", "colors")], cam).rgb
    assert int((id_map == 1).sum()) > 20 and int((id_map == 2).sum()) > 20

    logs: list[dict] = []
    # The 3D term is off: with two Gaussians each is the other's only neighbour, and I3 would
    # rightly pull them together. This pins the 2D loss alone.
    cfg = _photo_cfg(iters=400, lr_identity=2e-2, lr_identity_classifier=2e-2,
                             log_every=50, lambda_identity_3d=0.0)
    DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r).fit(
        [(cam, gt, 0, id_map)], on_log=logs.append
    )
    assert logs[-1]["identity_loss"] < 0.05, logs[-1]
    assert scene.gaussian_identity_probs()[:2].argmax(-1).tolist() == [1, 2]


@requires_gsplat
@pytest.mark.parametrize("detach", [True, False])
def test_detached_identity_reads_geometry_and_never_writes_it(detach: bool) -> None:
    """Detached (the default): the identity loss reaches the identity vectors and the
    classifier, and not one geometric parameter. Coupled: it reaches geometry too -- which is
    what let pseudo-label errors cost 1.2 dB on lab-coat."""
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    means = np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], np.float32)
    scene, _ = bundle_to_scene(_bundle(g=2, nb=1, means=means, scale=0.12), device="cuda")
    scene.enable_identity(20, NODES, seed=0)
    cam = RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 96, 64)
    cfg = _photo_cfg(iters=1, identity_detach_geometry=detach)
    tr = DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=GsplatRasterizer("cuda"))
    id_map = torch.full((64, 96), 1, dtype=torch.long, device="cuda")
    id_map[:, 48:] = 2

    cloud = scene.cloud_at(0)
    out = tr.rasterizer.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities,
                                  cloud.colors, cam,
                                  features=None if detach else cloud.features)
    tr.identity_term(cloud, cam, id_map, out).backward()

    assert scene.fg.identity.grad.abs().sum() > 0
    assert scene.identity_classifier.weight.grad.abs().sum() > 0
    geometry = [scene.fg.means, scene.fg.log_scales, scene.fg.quats, scene.fg.logit_opacities,
                scene.fg.motion_coefs, scene.motion_bases.transls]
    touched = [p.grad is not None and bool(p.grad.abs().sum() > 0) for p in geometry]
    assert not any(touched) if detach else any(touched)


# -- I3: 3D neighbour consistency -------------------------------------------------------------

def test_knn_matches_brute_force_across_chunks_and_never_returns_self() -> None:
    g = torch.Generator().manual_seed(0)
    pts = torch.randn((500, 3), generator=g)
    idx = torch.randperm(500, generator=g)[:40]
    got = knn_indices(pts[idx], pts, 5, exclude=idx, chunk=64)       # 8 chunks
    d = torch.cdist(pts[idx], pts)
    d[torch.arange(40), idx] = float("inf")
    want = d.topk(5, largest=False).indices
    assert torch.equal(torch.sort(got, 1).values, torch.sort(want, 1).values)
    assert not (got == idx[:, None]).any()


def _clusters(n_per: int = 30):
    """Two tight, well-separated clusters per field."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn((n_per, 3), generator=g) * 0.01
    means = torch.cat([a - 1.0, a + 1.0]).numpy().astype(np.float32)
    scene, _ = bundle_to_scene(_bundle(g=2 * n_per, nb=2 * n_per, means=means), device="cpu")
    with torch.no_grad():
        scene.bg.means.copy_(torch.as_tensor(means) + 10.0)
    scene.enable_identity(20, NODES, seed=0)
    return scene


def test_consistency_is_zero_when_neighbours_agree() -> None:
    scene = _clusters()
    with torch.no_grad():
        scene.fg.identity.zero_()
        scene.bg.identity.zero_()
    loss = _trainer(scene).identity_consistency_loss(torch.Generator().manual_seed(0))
    assert float(loss.detach()) < 1e-8


def test_consistency_reaches_identity_and_never_geometry() -> None:
    scene = _clusters()
    _trainer(scene).identity_consistency_loss(torch.Generator().manual_seed(0)).backward()
    assert scene.fg.identity.grad.abs().sum() > 0 and scene.bg.identity.grad.abs().sum() > 0
    assert scene.identity_classifier.weight.grad.abs().sum() > 0
    for p in (scene.fg.means, scene.bg.means, scene.fg.log_scales, scene.fg.logit_opacities):
        assert p.grad is None or not p.grad.any()


def test_consistency_raises_agreement_within_clusters_not_across() -> None:
    """Noisy identity inside each of two separated clusters: optimising the 3D term alone
    makes each cluster agree with itself, and does not merge the clusters."""
    scene = _clusters()
    n = scene.fg.num_gaussians
    with torch.no_grad():
        head = scene.identity_classifier
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1, 0], head.weight[2, 1] = 5.0, 5.0            # dim 0 -> class 1, dim 1 -> 2
        ident = torch.zeros((n, 20))
        ident[: n // 2, 0] = 1.0
        ident[n // 2:, 1] = 1.0
        flip = torch.randperm(n, generator=torch.Generator().manual_seed(2))[: n // 5]
        ident[flip] = ident[flip].flip(0)                          # 20% noise, both ways
        scene.fg.identity.copy_(ident)
    cfg = _photo_cfg(iters=1, identity_3d_samples=n, lr_identity=5e-2)
    tr = _trainer(scene)
    tr.cfg = cfg
    opt = torch.optim.Adam([scene.fg.identity, scene.bg.identity], lr=5e-2)
    gen = torch.Generator().manual_seed(0)

    def agreement() -> float:
        arg = torch.softmax(head(scene.fg.identity), -1).argmax(-1)
        nn = knn_indices(scene.fg.means.detach(), scene.fg.means.detach(), 5,
                         exclude=torch.arange(n))
        return float((arg[nn] == arg[:, None]).float().mean())

    before = agreement()
    for _ in range(200):
        opt.zero_grad()
        tr.identity_consistency_loss(gen).backward()
        opt.step()
    arg = torch.softmax(head(scene.fg.identity), -1).argmax(-1)
    assert agreement() > before + 0.1
    assert set(arg[: n // 2].tolist()) != set(arg[n // 2:].tolist())   # clusters not merged


def test_the_3d_term_defaults_are_gaussian_groupings() -> None:
    """k=5, lambda=2, every 2 steps: their reg3d defaults. Their tuning only carries over
    because the KL is divided by their class count (200), not ours (4-24 per scene)."""
    cfg = _photo_cfg()
    assert (cfg.identity_3d_k, cfg.lambda_identity_3d, cfg.identity_3d_every) == (5, 2.0, 2)
    assert cfg.identity_3d_norm == 200.0


def test_the_3d_term_strength_does_not_depend_on_the_scene_class_count() -> None:
    """Same neighbours, same distributions over the first classes: a scene with more
    (unused) classes must not be smoothed more gently -- or, as before, fewer more harshly."""
    losses = []
    for extra in (0, 20):
        scene = _clusters()
        nodes = NODES + [f"object:cam_00:sam2:cam_00:{9 + i}" for i in range(extra)]
        scene.enable_identity(20, nodes, seed=0)
        with torch.no_grad():
            head = scene.identity_classifier
            head.weight.zero_()
            head.bias.fill_(-1e4)
            head.bias[:3] = 0.0
            head.weight[1, 0], head.weight[2, 1] = 5.0, 5.0
            ident = torch.zeros_like(scene.fg.identity)
            ident[::2, 0] = 1.0
            ident[1::2, 1] = 1.0
            scene.fg.identity.copy_(ident)
            scene.bg.identity.copy_(ident)
        losses.append(float(_trainer(scene).identity_consistency_loss(
            torch.Generator().manual_seed(0)).detach()))
    assert losses[0] > 0 and abs(losses[0] - losses[1]) < 1e-6 * losses[0]


# -- training-time identity trace -------------------------------------------------------------

def _two_blob_views():
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    means = np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], np.float32)
    scene, _ = bundle_to_scene(_bundle(g=2, nb=1, means=means, scale=0.12), device="cuda")
    with torch.no_grad():
        scene.bg.logit_opacities.fill_(-20.0)
    scene.enable_identity(20, NODES, seed=0)
    cam = RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 96, 64)
    r = GsplatRasterizer("cuda")
    id_map = torch.full((64, 96), IGNORE, dtype=torch.long, device="cuda")
    with torch.no_grad():
        c = scene.cloud_at(0)
        for k in range(2):
            op = torch.zeros_like(c.opacities)
            op[k] = c.opacities[k]
            id_map[r.rasterize(c.means, c.quats, c.scales, op, c.colors, cam).alpha > 0.5] = k + 1
        gt = r.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, cam).rgb
    return scene, r, [(cam, gt, 0, id_map)]


@requires_gsplat
def test_the_trace_records_fit_stability_supervision_and_selection() -> None:
    scene, r, views = _two_blob_views()
    cfg = _photo_cfg(iters=300, lr_identity=2e-2, lr_identity_classifier=2e-2,
                             lambda_identity_3d=0.0, identity_trace_every=100)
    records, snaps = [], []
    DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r).fit(
        views, on_trace=records.append, on_snapshot=lambda s, sc: snaps.append(s))
    assert [rec["step"] for rec in records] == [1, 100, 200, 300]
    first, last = records[0], records[-1]
    # identity learns its labels: 2D fit rises, entropy falls
    assert last["iou_entities_mean"] > first["iou_entities_mean"]
    assert last["iou_entities_mean"] > 0.8
    assert last["entropy_fg"] < first["entropy_fg"]
    # both visible blobs get supervision; the invisible background Gaussian gets none
    assert last["unsupervised_frac_fg"] == 0.0 and last["unsupervised_frac_bg"] == 1.0
    # no densification: per-Gaussian flips are comparable, and settle
    assert first["flip_frac_fg"] is None and last["flip_frac_fg"] == 0.0
    # each blob's own Gaussians select it: both scorings agree on a clean two-blob scene
    assert last["sel_entities"] == 2
    assert last["sel_visible_precision_median"] > 0.9 and last["sel_alone_recall_median"] > 0.8
    assert snaps == []


@requires_gsplat
def test_snapshots_fire_at_the_requested_steps() -> None:
    scene, r, views = _two_blob_views()
    cfg = _photo_cfg(iters=30, lambda_identity_3d=0.0, snapshot_steps=(10, 30))
    snaps = []
    DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r).fit(
        views, on_snapshot=lambda s, sc: snaps.append(s))
    assert snaps == [10, 30]


# -- identity reset ---------------------------------------------------------------------------

def test_reset_zeroes_identity_and_its_momentum_and_keeps_the_head() -> None:
    scene, _ = _scene()
    tr = _trainer(scene)
    for g in tr.optimizer.param_groups:           # give every param some Adam state
        for p in g["params"]:
            p.grad = torch.ones_like(p)
    tr.optimizer.step()
    head_w = scene.identity_classifier.weight.detach().clone()
    tr.reset_identity(step=7)
    assert not scene.fg.identity.any() and not scene.bg.identity.any()
    for g in tr.optimizer.param_groups:
        st = tr.optimizer.state[g["params"][0]]
        cleared = not st["exp_avg"].any() and not st["exp_avg_sq"].any()
        assert cleared == (g["name"] in ("fg.identity", "bg.identity")), g["name"]
    assert torch.equal(scene.identity_classifier.weight, head_w)
    assert tr.last_identity_reset == 7


def test_reset_schedule_leaves_a_recovery_window() -> None:
    scene, _ = _scene()
    tr = DynamicGaussianTrainer(scene, _photo_cfg(iters=5000, identity_reset_every=1000),
                                device="cpu")
    assert [s for s in range(1, 5001) if tr._identity_reset_due(s)] == [1000, 2000, 3000]
    tr.cfg.identity_reset_until = 1500
    assert [s for s in range(1, 5001) if tr._identity_reset_due(s)] == [1000]


@requires_gsplat
@pytest.mark.parametrize("reset", [False, True])
def test_a_hidden_gaussian_does_not_earn_its_identity_back(reset: bool) -> None:
    """Two visible blobs and a third Gaussian no training view sees, carrying a stale label.

    Without a reset nothing ever touches its vector, so its class is whatever the (still
    training) classifier happens to map that unchecked vector to. With one, the visible blobs
    earn their classes back from the pixels and the hidden one is left at zero: the
    classifier's prior, which in a scene with labelled background is background."""
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    # the third Gaussian sits behind the camera (at x=+2, looking at the origin): never seen
    means = np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0], [6.0, 0.0, 0.0]], np.float32)
    scene, _ = bundle_to_scene(_bundle(g=3, nb=1, means=means, scale=0.12), device="cuda")
    with torch.no_grad():
        scene.bg.logit_opacities.fill_(-20.0)
    scene.enable_identity(20, NODES, seed=0)
    cam = RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 96, 64)
    r = GsplatRasterizer("cuda")
    id_map = torch.zeros((64, 96), dtype=torch.long, device="cuda")     # background labelled
    with torch.no_grad():
        c = scene.cloud_at(0)
        for k in range(2):
            op = torch.zeros_like(c.opacities)
            op[k] = c.opacities[k]
            id_map[r.rasterize(c.means, c.quats, c.scales, op, c.colors, cam).alpha > 0.3] = k + 1
        gt = r.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, cam).rgb
    cfg = _photo_cfg(iters=600, lr_identity=2e-2, lr_identity_classifier=2e-2,
                             lambda_identity_3d=0.0,
                             identity_reset_every=200 if reset else 0, identity_reset_until=200)
    tr = DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r)
    with torch.no_grad():                       # a stale, confident label on the hidden one
        tr.scene.fg.identity[2] = 0.0
        tr.scene.fg.identity[2, :] = 3.0 * tr.scene.identity_classifier.weight[1].sign()
    stale = tr.scene.fg.identity[2].detach().clone()
    tr.fit([(cam, gt, 0, id_map)])
    probs = tr.scene.gaussian_identity_probs()
    assert probs[:2].argmax(-1).tolist() == [1, 2]          # visible blobs earn their class
    if reset:
        assert not tr.scene.fg.identity[2].any()            # never re-earned
        assert int(probs[2].argmax()) == 0                  # reads as the prior: background
        assert float(probs[2, 1:].max()) < 0.5              # no entity claims it
    else:
        assert torch.equal(tr.scene.fg.identity[2], stale)  # nothing ever checked it


# -- training signal and schedule -------------------------------------------------------------

@requires_gsplat
def test_scales_lr_decays_to_its_final_ratio_and_batches_train() -> None:
    scene, r, views = _two_blob_views()
    views = views * 4                                   # four copies: a batch of distinct picks
    cfg = _photo_cfg(iters=20, lambda_identity_3d=0.0, views_per_step=3,
                             lr_scales_final_ratio=0.1)
    tr = DynamicGaussianTrainer(scene, cfg, device="cuda", rasterizer=r)
    logs: list[dict] = []
    tr.fit(views, on_log=logs.append)
    lrs = {g["name"]: g["lr"] for g in tr.optimizer.param_groups}
    assert abs(lrs["fg.log_scales"] - 0.1 * cfg.lr_scales) < 1e-12
    assert lrs["fg.means"] == cfg.lr_means                     # only scales decay, as in SoM
    assert logs and all(np.isfinite(r_["loss"]) for r_ in logs)

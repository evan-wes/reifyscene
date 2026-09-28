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
from reifyscene.render.gaussian4d.dynamic_trainer import DynamicGaussianTrainer, DynamicTrainConfig
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

def _trainer(scene):
    return DynamicGaussianTrainer(scene, DynamicTrainConfig(iters=1), device="cpu")


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
    cfg = DynamicTrainConfig(iters=400, lr_identity=2e-2, lr_identity_classifier=2e-2, log_every=50)
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
    cfg = DynamicTrainConfig(iters=1, identity_detach_geometry=detach)
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

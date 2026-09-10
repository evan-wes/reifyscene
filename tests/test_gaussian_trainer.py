from __future__ import annotations

import torch

from reifyscene.render.gaussian4d.densify import (
    DensificationController,
    DensifyConfig,
    compute_scene_extent,
)
from reifyscene.render.gaussian4d.gaussian_field import PARAM_NAMES, GaussianField, logit
from reifyscene.render.gaussian4d.gaussian_trainer import image_psnr, image_ssim, ssim
from reifyscene.render.gaussian4d.rasterizer import GaussianCloud


def _make_field(n: int = 12, scale: float = 0.005, opacity: float = 0.9) -> GaussianField:
    g = torch.Generator().manual_seed(0)
    means = torch.rand((n, 3), generator=g)
    quats = torch.zeros((n, 4))
    quats[:, 0] = 1.0
    log_scales = torch.log(torch.full((n, 3), float(scale)))
    logit_opac = logit(torch.full((n,), float(opacity)))
    colors_logit = logit(torch.rand((n, 3), generator=g).clamp(0.05, 0.95))
    return GaussianField(means, quats, log_scales, logit_opac, colors_logit)


def _make_optimizer(field: GaussianField) -> torch.optim.Optimizer:
    opt = torch.optim.Adam(
        [{"params": [field.get_param(name)], "lr": 1e-3, "name": name} for name in PARAM_NAMES],
        eps=1e-15,
    )
    # One step so Adam moment state exists (densification surgery must follow it).
    loss = sum(p.sum() for p in field.parameters())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return opt


def _assert_optimizer_consistent(field: GaussianField, optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        param = group["params"][0]
        assert param is field.get_param(group["name"])  # field + optimizer share the param
        state = optimizer.state.get(param)
        if state:
            assert state["exp_avg"].shape == param.shape
            assert state["exp_avg_sq"].shape == param.shape
    # optimizer can still take a step on the resized params
    loss = sum(p.sum() for p in field.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# -- field ----------------------------------------------------------------

def test_gaussian_field_activations_roundtrip() -> None:
    field = _make_field(n=8, scale=0.01, opacity=0.8)
    assert torch.allclose(field.get_scales(), torch.full((8, 3), 0.01), atol=1e-6)
    assert torch.allclose(field.get_opacities(), torch.full((8,), 0.8), atol=1e-5)
    assert torch.allclose(field.get_quats().norm(dim=-1), torch.ones(8), atol=1e-6)
    cloud = field.to_cloud()
    assert isinstance(cloud, GaussianCloud)
    assert len(cloud) == 8


# -- metrics --------------------------------------------------------------

def test_ssim_and_psnr_on_identical_images() -> None:
    img = torch.rand((16, 16, 3))
    assert image_psnr(img, img) > 80.0
    assert image_ssim(img, img) > 0.99
    noisy = (img + 0.2).clamp(0, 1)
    assert image_psnr(noisy, img) < image_psnr(img, img)


def test_compute_scene_extent_positive() -> None:
    means = torch.randn((1000, 3))
    assert compute_scene_extent(means) > 0.0


# -- densification optimizer surgery (CPU) --------------------------------

def test_densify_clone_grows_field_and_optimizer() -> None:
    field = _make_field(n=12, scale=0.005)  # small -> clone (not split)
    optimizer = _make_optimizer(field)
    cfg = DensifyConfig(grad2d_threshold=0.0, scene_extent=1.0)  # grad>=0 -> all grow
    ctrl = DensificationController(field, optimizer, cfg, scene_extent=1.0)
    ctrl.grad2d_accum = torch.ones(field.num_gaussians)
    ctrl.count = torch.ones(field.num_gaussians)

    stats = ctrl.densify_and_prune(step=600)
    assert stats["cloned"] == 12 and stats["split"] == 0 and stats["pruned"] == 0
    assert field.num_gaussians == 24
    _assert_optimizer_consistent(field, optimizer)


def test_densify_split_consumes_parents() -> None:
    field = _make_field(n=10, scale=0.5)  # large -> split; ext=10 keeps children
    optimizer = _make_optimizer(field)
    cfg = DensifyConfig(grad2d_threshold=0.0, scene_extent=10.0, split_count=2)
    ctrl = DensificationController(field, optimizer, cfg, scene_extent=10.0)
    ctrl.grad2d_accum = torch.ones(field.num_gaussians)
    ctrl.count = torch.ones(field.num_gaussians)

    stats = ctrl.densify_and_prune(step=600)
    assert stats["split"] == 10 and stats["cloned"] == 0
    # 10 parents removed, 10*2 children appended.
    assert field.num_gaussians == 20
    _assert_optimizer_consistent(field, optimizer)


def test_densify_prunes_low_opacity() -> None:
    field = _make_field(n=12, scale=0.005, opacity=0.9)
    # Drive a few opacities below the prune threshold.
    with torch.no_grad():
        field.logit_opacities[:4] = logit(torch.tensor(1e-4))
    optimizer = _make_optimizer(field)
    cfg = DensifyConfig(grad2d_threshold=1e9, prune_opacity=0.005, scene_extent=1.0)  # no growth
    ctrl = DensificationController(field, optimizer, cfg, scene_extent=1.0)
    ctrl.grad2d_accum = torch.zeros(field.num_gaussians)
    ctrl.count = torch.ones(field.num_gaussians)

    stats = ctrl.densify_and_prune(step=600)
    assert stats["pruned"] == 4 and stats["cloned"] == 0
    assert field.num_gaussians == 8
    _assert_optimizer_consistent(field, optimizer)


def test_max_gaussians_cap_pauses_growth() -> None:
    field = _make_field(n=12, scale=0.005)
    optimizer = _make_optimizer(field)
    cfg = DensifyConfig(grad2d_threshold=0.0, scene_extent=1.0, max_gaussians=12)
    ctrl = DensificationController(field, optimizer, cfg, scene_extent=1.0)
    ctrl.grad2d_accum = torch.ones(field.num_gaussians)
    ctrl.count = torch.ones(field.num_gaussians)

    stats = ctrl.densify_and_prune(step=600)
    assert stats["cloned"] == 0 and stats["split"] == 0  # capped
    assert field.num_gaussians == 12

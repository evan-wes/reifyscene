"""N-channel features through the rasterizer (I1 of the per-Gaussian identity scope).

Identity reaches the image by riding in the colour channels of the same gsplat call, so what
these pin is: a feature composites with exactly RGB's weights, adding one changes nothing
about RGB or depth, gradients reach the per-Gaussian feature, and the plumbing refuses the
shapes that would otherwise misalign silently (a cloud half with features, half without).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from reifyscene.api.run_outputs import RunOutputs
from reifyscene.render.gaussian4d.rasterizer import (
    GaussianCloud,
    GsplatRasterizer,
    RenderCamera,
    gsplat_available,
)

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rasterizer tests need gsplat and a CUDA device"
)

REPO = Path(__file__).resolve().parents[1]
BLACKSWAN = REPO.parent / "runs" / "slice_blackswan"     # a real trained run, if on disk


def _cpu_cloud(n: int, d: int | None) -> GaussianCloud:
    return GaussianCloud(
        means=torch.zeros(n, 3),
        quats=torch.zeros(n, 4),
        scales=torch.zeros(n, 3),
        opacities=torch.zeros(n),
        colors=torch.zeros(n, 3),
        features=None if d is None else torch.arange(n * d, dtype=torch.float32).reshape(n, d),
    )


def _camera(width: int = 128, height: int = 96) -> RenderCamera:
    return RenderCamera.from_look_at([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                     width, height)


# -- CPU: the plumbing ------------------------------------------------------------------------

def test_concat_carries_features_in_order() -> None:
    a, b = _cpu_cloud(2, 5), _cpu_cloud(3, 5)
    merged = GaussianCloud.concat([a, b], device="cpu")
    assert merged.features.shape == (5, 5)
    assert torch.equal(merged.features[:2], a.features)
    assert torch.equal(merged.features[2:], b.features)


def test_concat_without_features_stays_without() -> None:
    merged = GaussianCloud.concat([_cpu_cloud(2, None), _cpu_cloud(1, None)], device="cpu")
    assert merged.features is None


def test_concat_refuses_a_cloud_missing_features() -> None:
    """Zero-filling would assert 'no identity' for Gaussians that were never given one."""
    with pytest.raises(ValueError, match="feature widths"):
        GaussianCloud.concat([_cpu_cloud(2, 5), _cpu_cloud(1, None)], device="cpu")


def test_concat_refuses_differing_widths() -> None:
    with pytest.raises(ValueError, match="feature widths"):
        GaussianCloud.concat([_cpu_cloud(2, 5), _cpu_cloud(1, 4)], device="cpu")


# -- CUDA: the render ------------------------------------------------------------------------

def _blob_cloud(features: np.ndarray | None, **kw) -> GaussianCloud:
    return GaussianCloud.from_numpy(
        means=kw.get("means", np.zeros((1, 3), np.float32)),
        scales=kw.get("scales", np.full((1, 3), 0.1, np.float32)),
        colors=kw.get("colors", np.asarray([[0.8, 0.2, 0.1]], np.float32)),
        opacities=kw.get("opacities", np.asarray([0.95], np.float32)),
        device="cuda",
        features=features,
    )


@requires_gsplat
@pytest.mark.parametrize("d", [1, 13, 16, 20])
def test_a_known_feature_renders_as_itself_times_alpha(d: int) -> None:
    """One Gaussian: every pixel's feature is its vector weighted by that pixel's alpha.

    D=16 pads inside gsplat (3 + 16 + 1 = 20 channels); D=13 and 20 do not. Both must agree.
    """
    v = np.linspace(-2.0, 3.0, d, dtype=np.float32)[None]
    out = GsplatRasterizer("cuda").render(_blob_cloud(v), _camera())
    assert out["features"].shape == (96, 128, d)
    expected = out["alpha"][..., None] * v[0]
    assert np.allclose(out["features"], expected, atol=1e-5)


@requires_gsplat
def test_features_composite_front_to_back_like_rgb() -> None:
    """Two overlapping Gaussians: the feature channel carrying the colour's value must equal
    the rendered colour. Same weights, same order, same pass."""
    colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    cloud = _blob_cloud(
        colors.copy(),                                     # features := colours
        means=np.asarray([[0.5, 0.0, 0.0], [-0.5, 0.05, 0.0]], np.float32),
        scales=np.full((2, 3), 0.12, np.float32),
        colors=colors,
        opacities=np.asarray([0.6, 0.9], np.float32),
    )
    out = GsplatRasterizer("cuda").render(cloud, _camera())
    assert np.allclose(out["features"], out["rgb"], atol=1e-5)
    centre = out["features"][48, 64]
    assert centre[0] > centre[1] > 0.0                     # front Gaussian dominates, back shows


@requires_gsplat
def test_adding_features_leaves_rgb_depth_and_alpha_untouched() -> None:
    base = _blob_cloud(None)
    with_feat = _blob_cloud(np.random.default_rng(0).normal(size=(1, 20)).astype(np.float32))
    r = GsplatRasterizer("cuda")
    a, b = r.render(base, _camera()), r.render(with_feat, _camera())
    assert "features" not in a
    for key in ("rgb", "alpha", "depth"):
        assert np.array_equal(a[key], b[key]), key


@requires_gsplat
def test_the_feature_gradient_reaches_the_gaussian() -> None:
    feats = torch.zeros((1, 20), device="cuda", requires_grad=True)
    cloud = _blob_cloud(None)
    out = GsplatRasterizer("cuda").rasterize(
        cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, _camera(),
        features=feats,
    )
    out.features[..., 7].sum().backward()
    grad = feats.grad[0].cpu().numpy()
    assert grad[7] > 0.0
    assert np.count_nonzero(grad) == 1


@requires_gsplat
def test_a_misshapen_feature_tensor_is_refused() -> None:
    cloud = _blob_cloud(None)
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        GsplatRasterizer("cuda").rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, _camera(),
            features=torch.zeros((2, 4), device="cuda"),
        )


@requires_gsplat
def test_an_empty_cloud_with_features_returns_blank_features() -> None:
    empty = GaussianCloud.empty("cuda")
    empty.features = torch.zeros((0, 20), device="cuda")
    out = GsplatRasterizer("cuda").render(empty, _camera(64, 48))
    assert out["features"].shape == (48, 64, 20) and not out["features"].any()


@requires_gsplat
@pytest.mark.skipif(not RunOutputs.is_run(BLACKSWAN), reason="needs the slice_blackswan run")
def test_rgb_is_unchanged_for_an_existing_checkpoint() -> None:
    """The done-when for I1: a real trained scene renders identically with a feature attached."""
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    run = RunOutputs.open(BLACKSWAN)
    data = run.checkpoint().load()
    scene, _ = bundle_to_scene(data, device="cuda")
    r = GsplatRasterizer("cuda")
    for i in (0, len(data["train_time_ids"]) // 2):
        K = np.asarray(data["train_Ks"][i], np.float32)
        w, h = run.streams.only().render_raster
        cam = RenderCamera(viewmat=np.asarray(data["train_w2cs"][i], np.float32), K=K,
                           width=w, height=h)
        with torch.no_grad():
            cloud = scene.cloud_at(int(data["train_time_ids"][i]))
            feats = torch.randn((len(cloud), 20), device="cuda")
            plain = r.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities,
                                cloud.colors, cam)
            wide = r.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities,
                               cloud.colors, cam, features=feats)
        assert plain.rgb.abs().sum() > 0
        assert torch.equal(plain.rgb, wide.rgb)
        assert torch.equal(plain.depth, wide.depth)
        assert torch.equal(plain.alpha, wide.alpha)
        assert wide.features.shape == (h, w, 20)

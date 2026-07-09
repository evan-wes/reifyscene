from __future__ import annotations

import numpy as np
import pytest

from room4d.common.types import CameraIntrinsics
from room4d.render.gaussian4d.rasterizer import (
    GaussianCloud,
    GsplatRasterizer,
    Rasterizer,
    RenderCamera,
    build_rasterizer,
    default_intrinsics,
    gsplat_available,
    look_at_viewmat,
    project_to_pixels,
)

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rasterizer tests need gsplat and a CUDA device"
)


def _camera(eye, target, up=(0.0, 0.0, 1.0)) -> dict:
    return {"eye": list(eye), "target": list(target), "up": list(up)}


def _render_camera(eye, target, width, height, up=(0.0, 0.0, 1.0)) -> RenderCamera:
    return RenderCamera.from_look_at(list(eye), list(target), list(up), width, height)


def test_look_at_viewmat_maps_target_to_optical_axis() -> None:
    eye = np.asarray([2.0, 1.0, 0.5], dtype=np.float32)
    target = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    viewmat = look_at_viewmat(eye, target, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    cam = viewmat[:3, :3] @ target + viewmat[:3, 3]
    # The look-at target sits on the +z optical axis at the eye-target distance.
    assert abs(cam[0]) < 1e-5 and abs(cam[1]) < 1e-5
    assert abs(cam[2] - np.linalg.norm(eye - target)) < 1e-5
    # Rotation block is orthonormal.
    R = viewmat[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)


def test_project_to_pixels_centers_the_target() -> None:
    camera = _camera([3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    xy, valid = project_to_pixels(np.zeros((1, 3), dtype=np.float32), camera, 200, 100)
    assert bool(valid[0])
    assert abs(xy[0, 0] - 100.0) < 1e-3
    assert abs(xy[0, 1] - 50.0) < 1e-3


def test_project_to_pixels_rejects_points_behind_camera() -> None:
    camera = _camera([3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    xy, valid = project_to_pixels(np.asarray([[6.0, 0.0, 0.0]], dtype=np.float32), camera, 200, 100)
    assert not bool(valid[0])


def test_default_intrinsics_matches_legacy_framing() -> None:
    K = default_intrinsics(800, 450)
    assert K[0, 0] == K[1, 1] == 0.75 * 800
    assert K[0, 2] == 400.0 and K[1, 2] == 225.0


@requires_gsplat
def test_render_single_gaussian_centered_blob() -> None:
    cloud = GaussianCloud.from_numpy(
        means=np.zeros((1, 3), dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
        opacities=np.asarray([0.95], dtype=np.float32),
        device="cuda",
    )
    out = GsplatRasterizer("cuda").render(cloud, _render_camera([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 128, 96))
    assert out["rgb"].shape == (96, 128, 3)
    assert out["alpha"].shape == (96, 128)
    assert out["depth"].shape == (96, 128)
    peak = np.unravel_index(np.argmax(out["alpha"]), out["alpha"].shape)
    assert abs(peak[0] - 48) <= 1 and abs(peak[1] - 64) <= 1
    assert out["rgb"][peak][0] > 0.5 and out["rgb"][peak][0] > out["rgb"][peak][1]
    assert abs(out["depth"][peak] - 2.0) < 0.05
    assert np.isinf(out["depth"][0, 0])


@requires_gsplat
def test_render_empty_cloud_returns_blank_canvases() -> None:
    out = GsplatRasterizer("cuda").render(
        GaussianCloud.empty("cuda"), _render_camera([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 64, 48)
    )
    assert float(out["rgb"].sum()) == 0.0
    assert float(out["alpha"].sum()) == 0.0
    assert np.all(np.isinf(out["depth"]))


@requires_gsplat
def test_cloud_concat_preserves_counts_and_renders() -> None:
    a = GaussianCloud.from_numpy(
        means=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), 0.05, dtype=np.float32),
        colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
        opacities=np.asarray([0.9], dtype=np.float32),
        device="cuda",
    )
    b = GaussianCloud.from_numpy(
        means=np.asarray([[0.0, 0.4, 0.0], [0.0, -0.4, 0.0]], dtype=np.float32),
        scales=np.full((2, 3), 0.05, dtype=np.float32),
        colors=np.asarray([[0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        opacities=np.asarray([0.9, 0.9], dtype=np.float32),
        device="cuda",
    )
    merged = GaussianCloud.concat([a, b, GaussianCloud.empty("cuda")], device="cuda")
    assert len(merged) == 3
    out = GsplatRasterizer("cuda").render(merged, _render_camera([2.5, 0.0, 0.0], [0.0, 0.0, 0.0], 128, 96))
    # All three colors must land on the canvas.
    assert float(out["rgb"][..., 0].max()) > 0.3
    assert float(out["rgb"][..., 1].max()) > 0.3
    assert float(out["rgb"][..., 2].max()) > 0.3


@requires_gsplat
def test_full_render_occludes_by_depth() -> None:
    # A red gaussian sits in front of a green one along the view ray; the
    # composited center pixel must be red-dominant.
    cloud = GaussianCloud.from_numpy(
        means=np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((2, 3), 0.08, dtype=np.float32),
        colors=np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        opacities=np.asarray([0.99, 0.99], dtype=np.float32),
        device="cuda",
    )
    out = GsplatRasterizer("cuda").render(cloud, _render_camera([3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 128, 96))
    center = out["rgb"][48, 64]
    assert center[0] > center[1] * 2.0


def test_render_camera_from_world_cam_projects_known_point() -> None:
    # Camera at world [1, 0, 0], identity rotation (OpenCV: +z forward into scene).
    # A world point straight ahead on the camera's +z axis must land on the
    # principal point — the coordinate-convention guard for from_world_cam.
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, 3] = [1.0, 0.0, 0.0]
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=64.0, cy=48.0)
    cam = RenderCamera.from_world_cam(T_world_cam, intr, width=128, height=96)
    assert cam.width == 128 and cam.height == 96
    assert np.allclose(cam.K, [[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]], atol=1e-5)
    world_pt = np.asarray([1.0, 0.0, 5.0, 1.0], dtype=np.float32)
    cam_pt = cam.viewmat @ world_pt
    assert np.allclose(cam_pt[:3], [0.0, 0.0, 5.0], atol=1e-5)
    px = cam.K @ cam_pt[:3]
    px = px[:2] / px[2]
    assert np.allclose(px, [64.0, 48.0], atol=1e-4)


def test_build_rasterizer_selects_backend() -> None:
    rasterizer = build_rasterizer({"device": "cpu"})
    assert isinstance(rasterizer, GsplatRasterizer)
    assert isinstance(rasterizer, Rasterizer)
    assert rasterizer.backend_name == "gsplat"
    assert rasterizer.device == "cpu"
    with pytest.raises(ValueError):
        build_rasterizer({"rasterizer_backend": "does-not-exist"})

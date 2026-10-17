"""Unit tests for the render-path geometry-provider seam (step 12, PR 2).

Dataset-free: the lossless equivalence of ``DyCheckGeometryProvider`` against the
old direct loaders is proven on real data in the PR validation; here we pin the
ABC contract and that the DyCheck provider delegates to the existing loaders with
the right arguments (so the seam is behavior-preserving by construction).
"""

from __future__ import annotations

import pytest

import reifyscene.lifting.native_track_lift as ntl
from reifyscene.lifting.geometry_provider import DyCheckGeometryProvider, GeometryProvider


def test_geometry_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        GeometryProvider()  # type: ignore[abstract]

    class _Fake(GeometryProvider):
        def cameras(self, split, *, device="cpu"):
            return ("cameras", split, device)

        def frame_stack(self, split, *, device="cpu"):
            return ("stack", split, device)

    fake = _Fake()
    assert isinstance(fake, GeometryProvider)
    assert fake.cameras("val", device="cuda") == ("cameras", "val", "cuda")


def test_dycheck_provider_stores_config_and_backend() -> None:
    prov = DyCheckGeometryProvider("/data/seq", factor=2, depth_type="lidar", camera_type="original")
    assert prov.dataset_root == "/data/seq"
    assert prov.factor == 2
    assert prov.depth_type == "lidar"
    assert prov.camera_type == "original"
    backend = prov._backend("train")
    # The DatasetGeometryInitializer is configured from the provider's fields.
    assert backend.cfg["dataset_root"] == "/data/seq"
    assert backend.cfg["factor"] == 2
    assert backend.cfg["split"] == "train"


def test_dycheck_provider_delegates_to_loaders(monkeypatch) -> None:
    """The provider must call the existing loaders with the exact same args the
    old direct path used — that is what makes the seam lossless."""
    calls: dict[str, tuple] = {}

    def fake_load_cameras(backend, split, device, dataset_root, factor, camera_type):
        calls["cameras"] = (split, device, dataset_root, factor, camera_type)
        return "CAMS"

    def fake_load_frame_stack(backend, split, device, dataset_root, factor, depth_type, camera_type):
        calls["stack"] = (split, device, dataset_root, factor, depth_type, camera_type)
        return "STACK"

    monkeypatch.setattr(ntl, "_load_cameras", fake_load_cameras)
    monkeypatch.setattr(ntl, "_load_frame_stack", fake_load_frame_stack)

    prov = DyCheckGeometryProvider("/data/seq", factor=1, depth_type="depth_anything_colmap", camera_type="refined")
    assert prov.cameras("val", device="cpu") == "CAMS"
    assert prov.frame_stack("train", device="cuda") == "STACK"
    assert calls["cameras"] == ("val", "cpu", "/data/seq", 1, "refined")
    assert calls["stack"] == ("train", "cuda", "/data/seq", 1, "depth_anything_colmap", "refined")


def test_build_native_init_inputs_accepts_provider() -> None:
    import inspect

    sig = inspect.signature(ntl.build_native_init_inputs)
    assert "provider" in sig.parameters
    assert sig.parameters["provider"].default is None

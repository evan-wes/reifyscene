"""Scene stage: the self-contained renderable SceneArtifact + render backend.

The scene assembler turns a reconstruction into a SELF-CONTAINED renderable
scene (dense seed + persisted train views + render trajectory); the render
backend consumes ONLY that artifact (seed -> trained GaussianField -> rendered
trajectory + checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from room4d.render.gaussian4d.rasterizer import gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rendering tests need gsplat and a CUDA device"
)

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.scene.scene_artifact import SceneArtifact

H = W = 16


def _write_frame(path: Path) -> str:
    yy, xx = np.mgrid[0:H, 0:W]
    rgb = np.stack(
        [(xx * 16).astype(np.uint8), (yy * 16).astype(np.uint8), np.full((H, W), 128, np.uint8)],
        axis=-1,
    )
    Image.fromarray(rgb).save(path)
    return str(path)


def _synthetic_geometry(tmp_path: Path) -> GeometryArtifact:
    """Two posed frames with constant-depth maps + GT images on disk."""
    frames = [FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) * 0.1) for i in range(2)]
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    debug_image_paths: dict[str, str] = {}
    depthmaps: dict[str, dict] = {}
    for fr in frames:
        key = f"cam_00:{fr.frame_idx}"
        debug_image_paths[key] = _write_frame(gt_dir / f"{fr.frame_idx}.png")
        depthmaps[key] = {"depth": np.full((H, W), 2.0, dtype=np.float32), "confidence": None}
    intr = CameraIntrinsics(60.0, 60.0, W / 2.0, H / 2.0)
    states = [
        TimedCameraState("cam_00", float(fr.frame_idx) * 0.1, intr, CameraPose(np.eye(4, dtype=np.float32)))
        for fr in frames
    ]
    return GeometryArtifact(
        run_id="scene",
        stage_name="geometry",
        save_dir=tmp_path / "geometry",
        selected_frames=frames,
        intrinsics_by_stream={"cam_00": intr},
        camera_states_by_stream={"cam_00": states},
        depthmaps_by_frame=depthmaps,
        metadata={"debug_image_paths": debug_image_paths},
    )


def test_assembler_builds_self_contained_renderable_scene(tmp_path: Path) -> None:
    geom = _synthetic_geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=6).assemble(
        run_id="scene", save_dir=tmp_path / "scene", geometry_artifact=geom
    )
    assert scene.kind == "static"
    # both frames fully valid (depth 2.0 > near), pixel_stride 1 -> H*W per frame
    assert scene.render_seed["num_gaussians"] == H * W * 2
    assert np.asarray(scene.render_seed["means_xyz"]).shape == (H * W * 2, 3)
    assert len(scene.train_views) == 2
    view = scene.train_views[0]
    assert set(view) >= {"frame_key", "time_id", "t_world_cam", "intrinsics", "width", "height", "image_path"}
    assert view["width"] == W and view["height"] == H
    assert np.asarray(view["t_world_cam"]).shape == (4, 4)
    assert len(scene.render_trajectory) == 6


def test_assembler_tolerates_legacy_scene_config_keys() -> None:
    # legacy ``scene:`` config fields from the retired toy must not raise
    SingleVideoSceneAssembler(max_static_points_per_frame=8, max_static_points_total=16, timeline_source="geometry")


def test_scene_artifact_roundtrip_persists_gt_frames(tmp_path: Path) -> None:
    geom = _synthetic_geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=4).assemble(
        run_id="scene", save_dir=tmp_path / "scene", geometry_artifact=geom
    )
    scene.save()
    loaded = SceneArtifact.load(tmp_path / "scene")
    assert loaded.kind == "static"
    assert loaded.render_seed["num_gaussians"] == scene.render_seed["num_gaussians"]
    assert np.asarray(loaded.render_seed["means_xyz"]).shape == (scene.render_seed["num_gaussians"], 3)
    assert len(loaded.train_views) == 2
    # GT frame persisted into the artifact (image_file), resolvable on disk
    assert Path(loaded.train_views[0]["image_file"]).exists()
    assert len(loaded.render_trajectory) == 4
    summary = loaded.summary()
    assert summary["num_train_views"] == 2 and summary["kind"] == "static"


@requires_gsplat
def test_render_backend_trains_and_renders(tmp_path: Path) -> None:
    from room4d.render.gaussian4d.render_backend import GaussianRenderBackend

    geom = _synthetic_geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=3).assemble(
        run_id="scene", save_dir=tmp_path / "scene", geometry_artifact=geom
    )
    backend = GaussianRenderBackend(
        {"device": "cuda", "iters": 10, "trainer": {"log_every": 5}, "densify": {}}
    )
    art = backend.fit(scene, save_dir=tmp_path / "render")
    assert art.scene_state["num_gaussians"] > 0
    assert art.scene_state["num_train_views"] == 2
    assert art.scene_state["train_view_psnr"] > 0.0
    assert (Path(art.save_dir) / art.scene_state["checkpoint"]).exists()
    flythrough = art.path_index.get("flythrough", [])
    assert len(flythrough) == 3 and Path(flythrough[0]).exists()


@requires_gsplat
def test_render_backend_rejects_dynamic_until_f0b(tmp_path: Path) -> None:
    from room4d.render.gaussian4d.render_backend import GaussianRenderBackend

    geom = _synthetic_geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=2).assemble(
        run_id="scene", save_dir=tmp_path / "scene", geometry_artifact=geom
    )
    scene.kind = "dynamic"
    with pytest.raises(NotImplementedError):
        GaussianRenderBackend({"device": "cuda"}).fit(scene, save_dir=tmp_path / "render")

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

from reifyscene.render.gaussian4d.rasterizer import gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rendering tests need gsplat and a CUDA device"
)

from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.scene.assembler import SingleVideoSceneAssembler
from reifyscene.scene.scene_artifact import SceneArtifact

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


def test_dynamic_scene_artifact_bundle_roundtrip(tmp_path: Path) -> None:
    """A built DynamicScene serializes into the SceneArtifact npz and reloads unchanged.

    CPU-only: exercises dump_bundle -> SceneArtifact.save/load -> bundle_to_scene
    without gsplat (no rasterization), validating the F0b serialization seam.
    """
    import torch

    from reifyscene.common.io import read_json
    from reifyscene.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
    from reifyscene.render.gaussian4d.gaussian_field import GaussianField, logit
    from reifyscene.render.gaussian4d.motion import MotionBases
    from reifyscene.render.gaussian4d.motion_init import BorrowedCamera, bundle_to_scene, dump_bundle

    n_fg, n_bg, k, t = 6, 4, 2, 5
    fg_q = torch.zeros(n_fg, 4)
    fg_q[:, 0] = 1.0
    fg = DynamicGaussianField(
        means=torch.randn(n_fg, 3), quats=fg_q, log_scales=torch.zeros(n_fg, 3),
        logit_opacities=logit(torch.full((n_fg,), 0.9)),
        colors_logit=logit(torch.rand(n_fg, 3).clamp(0.1, 0.9)),
        motion_coefs=torch.randn(n_fg, k),
    )
    bg_q = torch.zeros(n_bg, 4)
    bg_q[:, 0] = 1.0
    bg = GaussianField(
        means=torch.randn(n_bg, 3), quats=bg_q, log_scales=torch.zeros(n_bg, 3),
        logit_opacities=logit(torch.full((n_bg,), 0.8)),
        colors_logit=logit(torch.rand(n_bg, 3).clamp(0.1, 0.9)),
    )
    bases = MotionBases(rots=torch.randn(k, t, 6), transls=torch.randn(k, t, 3))
    scene = DynamicScene(fg, bases, bg)
    cams = {
        "train": [
            BorrowedCamera(f"cam_00:{i}", i, np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32))
            for i in range(t)
        ]
    }

    art = SceneArtifact(
        run_id="scene", stage_name="scene", save_dir=tmp_path / "scene",
        kind="dynamic", dynamic_bundle=dump_bundle(scene, cams),
    )
    art.save()

    # Heavy gaussian arrays live in the npz, not inlined into artifact.json.
    raw = read_json(tmp_path / "scene" / "artifact.json")
    assert raw["dynamic_bundle"]["npz_file"] == "dynamic/scene_bundle.npz"
    assert raw["dynamic_bundle"]["num_fg"] == n_fg and raw["dynamic_bundle"]["num_bg"] == n_bg
    assert (tmp_path / "scene" / "dynamic" / "scene_bundle.npz").exists()

    loaded = SceneArtifact.load(tmp_path / "scene")
    assert loaded.kind == "dynamic"
    scene2, cams2 = bundle_to_scene(loaded.dynamic_bundle, device="cpu")
    assert scene2.fg.num_gaussians == n_fg and scene2.bg.num_gaussians == n_bg
    assert scene2.num_frames == t and scene2.fg.num_bases == k
    assert len(cams2["train"]) == t and not cams2["val"]
    assert cams2["train"][0].time_id == 0 and cams2["train"][0].frame_name == "cam_00:0"
    assert torch.allclose(scene2.fg.means, fg.means, atol=1e-6)
    assert torch.allclose(scene2.fg.motion_coefs, fg.motion_coefs, atol=1e-6)
    assert torch.allclose(scene2.motion_bases.transls, bases.transls, atol=1e-6)
    # The materialized cloud at a frame is identical after the round-trip.
    assert torch.allclose(scene2.cloud_at(2).means, scene.cloud_at(2).means, atol=1e-5)


def test_dynamic_fg_mask_union_matches_provider_keys() -> None:
    """Segment instance masks union per frame into ArtifactGeometryProvider keys."""
    from reifyscene.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation

    fr0 = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    fr1 = FrameRef(stream_id="cam_00", frame_idx=1, timestamp_sec=0.1)
    # frame 0: two instances whose masks should OR together; frame 1: one instance.
    m_a = np.zeros((4, 4), bool); m_a[0, :] = True
    m_b = np.zeros((4, 4), bool); m_b[1, :] = True
    m_c = np.zeros((4, 4), bool); m_c[2, :] = True
    art = InstanceArtifact(
        run_id="r", stage_name="segment", save_dir=Path("."),
        observations=[
            InstanceObservation("i0", "t0", fr0, kind="object"),
            InstanceObservation("i1", "t1", fr0, kind="human"),
            InstanceObservation("i2", "t0", fr1, kind="object"),
        ],
        mask_arrays={"i0": m_a, "i1": m_b, "i2": m_c},
    )
    art.rebuild_indices()

    fg = SingleVideoSceneAssembler(kind="dynamic")._fg_mask_by_frame(art)
    assert set(fg) == {"cam_00:0", "cam_00:1"}  # provider-style frame keys
    assert np.array_equal(fg["cam_00:0"], (m_a | m_b).astype(np.float32))  # union of both
    assert np.array_equal(fg["cam_00:1"], m_c.astype(np.float32))
    assert SingleVideoSceneAssembler(kind="dynamic")._fg_mask_by_frame(None) == {}


def test_dynamic_tracker_defaults_cap_query_work() -> None:
    # The dynamic scene path coarsens the grid and budgets the all-pairs tracker
    # (the O(T^2) scene-assembly hotspot); the standalone tracker keeps SoM's
    # stride-4/no-budget defaults for shipped-track parity.
    asm = SingleVideoSceneAssembler(kind="dynamic", dynamic_num_fg=40_000)
    assert asm.tracker_cfg["grid_size"] == 8
    assert asm.tracker_cfg["max_total_queries"] == 3 * 40_000
    # Explicit tracker config still wins over the defaults.
    override = SingleVideoSceneAssembler(kind="dynamic", tracker={"grid_size": 4})
    assert override.tracker_cfg["grid_size"] == 4


@requires_gsplat
def test_render_backend_trains_and_renders(tmp_path: Path) -> None:
    from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend

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


def test_render_backend_dynamic_empty_bundle_skips(tmp_path: Path) -> None:
    """Dynamic dispatch with an empty bundle emits an empty render, not a crash (CPU)."""
    from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend

    geom = _synthetic_geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=2).assemble(
        run_id="scene", save_dir=tmp_path / "scene", geometry_artifact=geom
    )
    scene.kind = "dynamic"  # static-assembled -> empty dynamic_bundle, dispatch short-circuits
    art = GaussianRenderBackend({"device": "cpu"}).fit(scene, save_dir=tmp_path / "render")
    assert art.scene_state["kind"] == "dynamic" and art.scene_state["empty"] is True


@requires_gsplat
def test_render_backend_dynamic_trains_and_checkpoints(tmp_path: Path) -> None:
    """End-to-end dynamic render smoke: bundle -> DynamicGaussianTrainer -> checkpoint."""
    import torch

    from reifyscene.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
    from reifyscene.render.gaussian4d.gaussian_field import GaussianField, logit
    from reifyscene.render.gaussian4d.motion import MotionBases
    from reifyscene.render.gaussian4d.motion_init import BorrowedCamera, dump_bundle
    from reifyscene.render.gaussian4d.rasterizer import look_at_viewmat
    from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend

    rng = np.random.default_rng(0)
    n_fg, n_bg, k, t = 30, 30, 2, 3

    def _field(n: int, dynamic: bool):
        q = torch.zeros(n, 4)
        q[:, 0] = 1.0
        kw = dict(
            means=torch.as_tensor(rng.normal(0, 0.3, (n, 3)), dtype=torch.float32), quats=q,
            log_scales=torch.full((n, 3), float(np.log(0.05))),
            logit_opacities=logit(torch.full((n,), 0.9)),
            colors_logit=logit(torch.as_tensor(rng.uniform(0.2, 0.8, (n, 3)), dtype=torch.float32)),
        )
        return DynamicGaussianField(motion_coefs=torch.zeros(n, k), **kw) if dynamic else GaussianField(**kw)

    bases = MotionBases(rots=torch.zeros(k, t, 6), transls=torch.zeros(k, t, 3))
    bases.rots.data[..., 0] = 1.0
    bases.rots.data[..., 4] = 1.0  # 6D identity
    scene = DynamicScene(_field(n_fg, True), bases, _field(n_bg, False))

    w2c = np.asarray(
        look_at_viewmat(np.array([0, 0, 3], np.float32), np.zeros(3, np.float32), np.array([0, 1, 0], np.float32)),
        dtype=np.float32,
    )
    K = np.array([[16.0, 0, W / 2], [0, 16.0, H / 2], [0, 0, 1]], dtype=np.float32)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    cams, views = [], []
    for i in range(t):
        cams.append(BorrowedCamera(f"f{i}", i, w2c, K))
        views.append({
            "frame_key": f"f{i}", "time_id": i,
            "t_world_cam": np.linalg.inv(w2c).astype(np.float32),
            "intrinsics": {"fx": 16.0, "fy": 16.0, "cx": W / 2, "cy": H / 2},
            "width": W, "height": H, "image_path": _write_frame(gt_dir / f"{i}.png"),
        })

    art = SceneArtifact(
        run_id="scene", stage_name="scene", save_dir=tmp_path / "scene",
        kind="dynamic", dynamic_bundle=dump_bundle(scene, {"train": cams}), train_views=views,
    )
    backend = GaussianRenderBackend({"device": "cuda", "iters": 3, "dynamic": {"log_every": 2}, "densify": {"enabled": False}})
    out = backend.fit(art, save_dir=tmp_path / "render")
    assert out.scene_state["kind"] == "dynamic"
    assert out.scene_state["num_train_views"] == t and out.scene_state["num_fg"] == n_fg
    assert (Path(out.save_dir) / out.scene_state["checkpoint"]).exists()

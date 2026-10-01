"""Every place a photo, mask or track position meets geometry pixels goes through the transform.

The synthetic geometry here is a 20x40 frame whose geometry pixels are its middle
20x20 (a crop, like VGGT's centre crop), with a red top band and a blue bottom band
the crop removes. Nothing downstream may see red or blue, stretch the frame, or
lift a point from a band the geometry never saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import Crop, ImageTransform, Resize, transform_record
from reifyscene.lifting.backproject import TrackLifter3D
from reifyscene.lifting.geometry_provider import ArtifactGeometryProvider
from reifyscene.lifting.lift import lift_pixels
from reifyscene.optimize.solver import JointSpatiotemporalSolver
from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend
from reifyscene.scene.assembler import SingleVideoSceneAssembler
from reifyscene.scene.scene_artifact import SceneArtifact
from reifyscene.tracking.track_artifact import TrackArtifact

W, H = 20, 40  # ingest frame
GW, GH = 20, 20  # geometry pixels: rows 10..29 of the frame
CROP = ImageTransform((W, H), (Crop((0, 10, 20, 30)),))
RED, BLUE = (255, 0, 0), (0, 0, 255)


def _frame() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([np.full((H, W), 60), (xx * 10) % 200 + 20, (yy * 5) % 200 + 20], -1).astype(
        np.uint8
    )
    img[:10] = RED
    img[30:] = BLUE
    return img


def _geometry(
    tmp_path: Path, *, n: int = 2, record: bool = True
) -> GeometryArtifact:
    frames = [FrameRef("cam_00", i, 0.1 * i) for i in range(n)]
    photos, depthmaps = {}, {}
    (tmp_path / "gt").mkdir(parents=True, exist_ok=True)
    for fr in frames:
        key = f"cam_00:{fr.frame_idx}"
        path = tmp_path / "gt" / f"{fr.frame_idx}.png"
        Image.fromarray(_frame()).save(path)
        photos[key] = str(path)
        depthmaps[key] = {
            "depth": np.full((GH, GW), 2.0, np.float32),
            "confidence": np.full((GH, GW), 2.0, np.float32),
        }
    intr = CameraIntrinsics(30.0, 30.0, GW / 2.0, GH / 2.0)
    states = [
        TimedCameraState("cam_00", 0.1 * i, intr, CameraPose(np.eye(4, dtype=np.float32)))
        for i in range(n)
    ]
    metadata = {"debug_image_paths": photos}
    if record:
        metadata["image_transforms"] = {"cam_00": transform_record(CROP, recorded_by="test")}
    return GeometryArtifact(
        run_id="t",
        stage_name="geometry",
        save_dir=tmp_path / "geometry",
        selected_frames=frames,
        intrinsics_by_stream={"cam_00": intr},
        camera_states_by_stream={"cam_00": states},
        depthmaps_by_frame=depthmaps,
        metadata=metadata,
    )


def _ingest(tmp_path: Path) -> SimpleNamespace:
    """The ingest stream of the fixture's frames (the photos every stage reads)."""
    from reifyscene.ingest.decoder import VideoDecoder

    return SimpleNamespace(streams=[VideoDecoder().decode(tmp_path / "gt", stream_id="cam_00")])


def _streams(geom: GeometryArtifact, tmp_path: Path):
    from reifyscene.ingest.streams import RunStreams

    return RunStreams.from_artifacts(_ingest(tmp_path), geom)


def _no_bands(img: np.ndarray) -> bool:
    flat = img.reshape(-1, 3)
    return not ((flat == RED).all(1).any() or (flat == BLUE).all(1).any())


# -- scene: static train views + seed ---------------------------------------------
def test_static_train_photos_and_seed_are_the_crop(tmp_path: Path) -> None:
    geom = _geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=2).assemble(
        run_id="t", save_dir=tmp_path / "scene", geometry_artifact=geom,
        streams=_streams(geom, tmp_path),
    )
    view = scene.train_views[0]
    assert (view["width"], view["height"]) == (GW, GH)
    photo = np.asarray(Image.open(view["image_path"]))
    assert np.array_equal(photo, _frame()[10:30])
    # the photo sits where the scene keeps it, so in-memory and reloaded scenes train on it
    assert Path(view["image_path"]).parent == (tmp_path / "scene" / "train_views").resolve()
    assert _no_bands(np.asarray(scene.render_seed["colors_rgb"]))
    assert scene.metadata["image_transforms"]["cam_00"]["source_size"] == [W, H]

    scene.save()
    loaded = SceneArtifact.load(tmp_path / "scene")
    assert np.array_equal(np.asarray(Image.open(loaded.train_views[0]["image_file"])), photo)


def test_a_frame_the_stream_does_not_have_raises(tmp_path: Path) -> None:
    from reifyscene.ingest.streams import StreamError

    geom = _geometry(tmp_path)
    streams = _streams(geom, tmp_path)
    (tmp_path / "gt" / "1.png").unlink()        # the stream recorded it; it is gone
    with pytest.raises(StreamError, match="frame 1"):
        SingleVideoSceneAssembler().assemble(
            run_id="t", save_dir=tmp_path / "scene", geometry_artifact=geom, streams=streams)


def test_geometry_without_depth_builds_an_empty_scene(tmp_path: Path) -> None:
    geom = _geometry(tmp_path)
    geom.depthmaps_by_frame = {}  # heuristic geometry: cameras, no depth
    scene = SingleVideoSceneAssembler().assemble(
        run_id="t", save_dir=tmp_path / "scene", geometry_artifact=geom,
        streams=_streams(geom, tmp_path))
    assert scene.train_views == [] and scene.render_seed["num_gaussians"] == 0


def test_seed_refuses_a_photo_not_in_depth_pixels() -> None:
    from reifyscene.render.gaussian4d.seed import _backproject_view

    with pytest.raises(ValueError, match="put the photo into the geometry's pixels"):
        _backproject_view(
            np.ones((GH, GW)),
            _frame(),
            CameraIntrinsics(30, 30, 10, 10),
            np.eye(4),
            pixel_stride=1,
            depth_near=1e-3,
            depth_far=None,
            scale_mult=1.0,
            scale_min=1e-5,
            scale_max=1.0,
        )


# -- dynamic: train views + provider ----------------------------------------------
def test_dynamic_train_photos_are_the_crop(tmp_path: Path) -> None:
    geom = _geometry(tmp_path)
    cams = [
        SimpleNamespace(
            frame_name=f"cam_00:{i}",
            time_id=i,
            w2c=np.eye(4),
            K=np.array([[30, 0, 10], [0, 30, 10], [0, 0, 1.0]]),
        )
        for i in range(2)
    ]
    asm = SingleVideoSceneAssembler()
    asm._streams = _streams(geom, tmp_path)
    views = asm._dynamic_train_views(cams, GH, GW, geom, tmp_path / "scene")
    assert [
        np.array_equal(np.asarray(Image.open(v["image_path"])), _frame()[10:30]) for v in views
    ] == [True, True]


def test_provider_crops_rgb_and_foreground(tmp_path: Path) -> None:
    geom = _geometry(tmp_path)
    fg = np.zeros((H, W), np.float32)
    fg[:10] = 1.0  # a "foreground" in the band the geometry never saw
    fg[15:18, 5:8] = 1.0  # and one inside it
    stack = ArtifactGeometryProvider(geom, stream=_streams(geom, tmp_path).only(),
                                     fg_mask_by_frame={"cam_00:0": fg}).frame_stack("train")
    rgb = stack.rgbs.numpy()
    assert rgb.shape == (2, GH, GW, 3) and np.array_equal(rgb[0], _frame()[10:30])
    assert np.array_equal(stack.fg.numpy()[0], fg[10:30])
    assert stack.cameras.time_ids.tolist() == [0, 1]


def test_provider_raises_on_a_missing_photo(tmp_path: Path) -> None:
    from reifyscene.ingest.streams import StreamError

    geom = _geometry(tmp_path)
    stream = _streams(geom, tmp_path).only()
    (tmp_path / "gt" / "1.png").unlink()
    with pytest.raises(StreamError, match="frame 1"):
        ArtifactGeometryProvider(geom, stream=stream).frame_stack("train")


# -- render backend -----------------------------------------------------------------
def test_load_gt_refuses_to_squash(tmp_path: Path) -> None:
    path = tmp_path / "p.png"
    Image.fromarray(_frame()).save(path)
    backend = GaussianRenderBackend({"device": "cpu"})
    with pytest.raises(ValueError, match="is 20x40 but its camera is 20x20"):
        backend._load_gt(str(path), GW, GH)
    assert backend._load_gt(str(path), W, H).shape == (H, W, 3)
    with pytest.raises(FileNotFoundError):
        backend._train_views(
            SimpleNamespace(
                train_views=[
                    {
                        "frame_key": "cam_00:0",
                        "t_world_cam": np.eye(4),
                        "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1},
                        "width": 2,
                        "height": 2,
                    }
                ]
            )
        )


# -- scene artifact persistence -------------------------------------------------------
def test_resaving_a_scene_copies_its_photos_and_writes_relative_paths(tmp_path: Path) -> None:
    geom = _geometry(tmp_path)
    scene = SingleVideoSceneAssembler(num_render_cameras=2).assemble(
        run_id="t", save_dir=tmp_path / "run1" / "scene", geometry_artifact=geom,
        streams=_streams(geom, tmp_path),
    )
    scene.save()
    loaded = SceneArtifact.load(tmp_path / "run1" / "scene")
    loaded.save_dir = tmp_path / "run2" / "scene"
    loaded.save()
    raw = json.loads((tmp_path / "run2" / "scene" / "artifact.json").read_text())
    for view in raw["train_views"]:
        assert not Path(view["image_file"]).is_absolute() and view["image_file"].startswith(
            "train_views/"
        )
        assert (tmp_path / "run2" / "scene" / view["image_file"]).exists()
    # run2 no longer depends on run1's files
    for f in (tmp_path / "run1" / "scene" / "train_views").iterdir():
        f.unlink()
    reloaded = SceneArtifact.load(tmp_path / "run2" / "scene")
    assert all(Path(v["image_file"]).exists() for v in reloaded.train_views)


# -- the one lift -------------------------------------------------------------------------
def test_lift_pixels_recovers_points_and_drops_the_cropped_band() -> None:
    K = np.array([[30.0, 0, 10.0], [0, 30.0, 10.0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = [0.5, -0.2, 1.0]
    depth = np.full((GH, GW), 2.0, np.float32)
    geo_px = np.array([[4.0, 3.0], [15.0, 17.0]])
    src_px = geo_px + [0.0, 10.0]  # ingest-frame positions of those geometry pixels
    xyz, valid = lift_pixels(np.vstack([src_px, [[5.0, 2.0], [5.0, 35.0]]]), CROP, depth, K, T)
    expected = (
        np.stack([(geo_px[:, 0] - 10) / 30 * 2, (geo_px[:, 1] - 10) / 30 * 2, np.full(2, 2.0)], -1)
        + T[:3, 3]
    )
    assert valid.tolist() == [True, True, False, False]
    assert np.allclose(xyz[:2], expected, atol=1e-5)
    assert np.isnan(xyz[2:]).all()
    # a depth hole or a low confidence is no point either
    depth[3, 4] = np.nan
    conf = np.full((GH, GW), 2.0)
    conf[17, 15] = 0.5
    _, valid = lift_pixels(src_px, CROP, depth, K, T, confidence=conf, conf_threshold=1.0)
    assert valid.tolist() == [False, False]


def test_tracking_lift_and_optimizer_relift_agree(tmp_path: Path) -> None:
    geom = _geometry(tmp_path, n=3)
    rng = np.random.default_rng(0)
    xy = np.stack([rng.uniform([0, 0], [W, H], size=(6, 2)) for _ in range(3)]).astype(np.float32)
    payload = {
        "frame_indices": np.arange(3),
        "track_ids": np.arange(6),
        "points_xy": xy,
        "visibility": np.ones((3, 6), bool),
        "original_size_wh": [W, H],
    }
    tracks = TrackArtifact(
        run_id="t",
        stage_name="tracking",
        save_dir=tmp_path / "tracking",
        track_payloads_by_stream={"cam_00": payload},
    )
    TrackLifter3D(point_conf_threshold=1.0).lift(tracks, geom)
    lifted = tracks.track_payloads_by_stream["cam_00"]["lifted_xyz"]
    assert tracks.metadata["lift"]["function"] == "lift_pixels"
    assert tracks.metadata["lift"]["image_transforms"]["cam_00"] == CROP.to_dict()

    solver = JointSpatiotemporalSolver({})
    state = solver.build_state(
        SimpleNamespace(offsets_by_stream={}, stream_ids=lambda: ["cam_00"]), geom, tracks
    )
    solver._relift_tracks_with_refined_cameras(state)
    relifted = state["track_payloads"]["cam_00"]["refined_lifted_xyz"]
    assert np.array_equal(np.isnan(lifted), np.isnan(relifted))
    assert np.allclose(np.nan_to_num(lifted), np.nan_to_num(relifted))
    # the band the crop removed never lifts
    in_band = (xy[..., 1] < 10) | (xy[..., 1] >= 30)
    assert np.isnan(lifted[in_band]).all() and not np.isnan(lifted[~in_band]).any()
    assert state["relift"]["point_conf_threshold"] == 1.0


# -- geometry stage + pose refine ---------------------------------------------------------
def test_pipeline_fills_and_checks_image_transforms(tmp_path: Path) -> None:
    ingest = SimpleNamespace(streams=[SimpleNamespace(stream_id="cam_00", width=W, height=H)])
    geom = _geometry(tmp_path, record=False)
    geom.depthmaps_by_frame = {}
    ReifyScenePipeline._finalize_image_transforms(geom, ingest)  # pose-only: identity at the frame
    assert ImageTransform.from_dict(geom.metadata["image_transforms"]["cam_00"]).is_identity

    bad = _geometry(tmp_path / "b")
    bad.metadata["image_transforms"]["cam_00"] = transform_record(
        ImageTransform((W, H), (Resize((10, 10), "bicubic"),)), recorded_by="test"
    )
    with pytest.raises(ValueError, match="depth maps"):
        ReifyScenePipeline._finalize_image_transforms(bad, ingest)


def test_pose_refine_images_are_the_stream_frames_in_geometry_pixels(tmp_path: Path) -> None:
    from reifyscene.geometry.pose_refine import PoseRefiner

    geom = _geometry(tmp_path)
    images = PoseRefiner({})._geometry_space_images(
        geom, "cam_00", list(geom.selected_frames), ingest_artifact=_ingest(tmp_path))
    assert images.shape == (2, GH, GW, 3) and np.array_equal(images[0], _frame()[10:30])

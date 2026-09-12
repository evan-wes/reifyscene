"""What a loaded artifact must carry at resume, and what is derived when it does not."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reifyscene.api.contracts import effective_settings, path_identity
from reifyscene.api.requirements import REQUIREMENTS, NotDerivable, ResumeContext


def _ctx(stage, artifact, tmp_path, *, cfg=None, artifacts=None, video_paths=()):
    return ResumeContext(
        stage, artifact, artifacts or {}, {}, cfg or {}, tmp_path / stage, list(video_paths)
    )


def _requirement(stage, name):
    return next(r for r in REQUIREMENTS[stage] if r.name == name)


def _geometry(debug_image_paths, backend="vggt-long", **metadata):
    return SimpleNamespace(
        metadata={
            "capability": {"backend_used": backend},
            "debug_image_paths": dict(debug_image_paths),
            **metadata,
        },
        notes=["Used VGGT-Long globally-aligned camera_poses.txt for camera poses."],
    )


def _png(path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))
    return path


# ------------------------------------------------------------------ photos

def test_missing_photos_are_found_in_the_runs_own_copies(tmp_path) -> None:
    copy = _png(tmp_path / "geometry" / "images" / "0__0.png")
    geo = _geometry({"0:0": str(tmp_path / "gone_cache" / "0000_0_000000.png")})
    req = _requirement("geometry", "its photos exist")
    ctx = _ctx("geometry", geo, tmp_path)
    assert "relative or missing" in req.check(ctx)
    done = req.derive(ctx)
    assert geo.metadata["debug_image_paths"]["0:0"] == str(copy.resolve())
    assert done.metadata == {"debug_image_paths": geo.metadata["debug_image_paths"]}
    assert not done.resave  # a metadata patch, not a full re-save
    assert "the run's own copies" in done.note
    assert req.check(ctx) is None


def test_missing_dataset_photos_are_found_under_the_current_dataset_root(tmp_path) -> None:
    root = tmp_path / "data" / "spin"
    photo = _png(root / "rgb" / "2x" / "0_00000.png")
    geo = _geometry({"0:0": "/old/project/data/spin/rgb/2x/0_00000.png"}, backend="dataset")
    cfg = {"geometry": {"backend_name": "dataset", "dataset_root": str(root), "factor": 2}}
    req = _requirement("geometry", "its photos exist")
    req.derive(_ctx("geometry", geo, tmp_path, cfg=cfg))
    assert geo.metadata["debug_image_paths"]["0:0"] == str(photo.resolve())


def test_missing_photos_of_unresized_input_are_the_input_frames(tmp_path) -> None:
    frame = _png(tmp_path / "frames" / "0_00007.png")
    stream = SimpleNamespace(stream_id="0", metadata={"frame_index": {"7": str(frame)}})
    ingest = SimpleNamespace(streams=[stream])
    geo = _geometry({"0:7": str(tmp_path / "deleted_cache" / "x.png")})
    req = _requirement("geometry", "its photos exist")
    req.derive(_ctx("geometry", geo, tmp_path, artifacts={"ingest": ingest}))
    assert geo.metadata["debug_image_paths"]["0:7"] == str(frame.resolve())


def test_photos_that_cannot_be_found_reject(tmp_path) -> None:
    geo = _geometry({"0:0": str(tmp_path / "nowhere.png")})
    cfg = {"geometry": {"resize_max_dim": 252}}  # resized input: the input frame is not the photo
    with pytest.raises(NotDerivable, match="no copy of the photo"):
        req = _requirement("geometry", "its photos exist")
        req.derive(_ctx("geometry", geo, tmp_path, cfg=cfg))


# ------------------------------------------------------------------ rejections

@pytest.mark.parametrize(
    ("stage", "artifact", "problem"),
    [
        ("geometry", _geometry({}, backend="empty"), "empty floor"),
        (
            "geometry",
            SimpleNamespace(metadata={"debug_image_paths": {}}, notes=[]),
            "made before the backend ladder",
        ),
        (
            "geometry",
            _geometry({}, ground_plane_applied=True, world_frame={}),
            "transform is not recorded",
        ),
        (
            "pose_refine",
            _geometry({}, pose_refine={"enabled": True, "refined_streams": []}),
            "refined nothing",
        ),
        (
            "pose_refine",
            _geometry(
                {}, pose_refine={"enabled": True, "refined_streams": ["0"], "stats": {"0": {}}}
            ),
            "re-anchored",
        ),
        ("humans", SimpleNamespace(metadata={}), "fabricated"),
        ("dyn_mask", SimpleNamespace(metadata={"capability": {"backend_used": "all-static"}}),
         None),
    ],
)
def test_unmet_requirements_that_cannot_be_derived(stage, artifact, problem, tmp_path) -> None:
    ctx = _ctx(stage, artifact, tmp_path)
    found = [r.check(ctx) for r in REQUIREMENTS[stage]]
    found = [f for f in found if f]
    if problem is None:
        assert found == []
    else:
        assert any(problem in f for f in found), found


def test_a_detector_run_does_not_pass_for_an_all_static_fallback(tmp_path) -> None:
    art = SimpleNamespace(metadata={"capability": {"backend_used": "all-static"}})
    ctx = _ctx("dyn_mask", art, tmp_path, cfg={"dyn_mask": {"backend_name": "detector"}})
    assert "a fallback ran" in REQUIREMENTS["dyn_mask"][0].check(ctx)


def test_reid_must_embed_exactly_segments_tracks(tmp_path) -> None:
    segment = SimpleNamespace(observations=[SimpleNamespace(track_id=t) for t in ("t0", "t1")])
    reid = SimpleNamespace(embeddings={"t0": np.zeros(2)}, merge_map={})
    ctx = _ctx("reid", reid, tmp_path, artifacts={"segment": segment})
    assert "missing ['t1']" in REQUIREMENTS["reid"][0].check(ctx)
    reid.embeddings["t1"] = np.zeros(2)
    assert REQUIREMENTS["reid"][0].check(ctx) is None


def test_a_scene_that_says_dynamic_needs_its_bundle(tmp_path) -> None:
    scene = SimpleNamespace(kind="dynamic", dynamic_bundle={}, train_views=[])
    assert "has no dynamic bundle" in REQUIREMENTS["scene"][0].check(_ctx("scene", scene, tmp_path))


# ------------------------------------------------------------------ input

def test_a_moved_input_is_rerooted(tmp_path) -> None:
    old = _png(tmp_path / "old" / "1x" / "0_00000.png").parent
    new = tmp_path / "new" / "1x"
    shutil.copytree(old, new)
    from reifyscene.ingest.decoder import VideoDecoder

    stream = VideoDecoder().decode(old, stream_id="cam_00")
    shutil.rmtree(old)
    ingest = SimpleNamespace(streams=[stream])
    req = _requirement("ingest", "its input frames exist")
    ctx = _ctx("ingest", ingest, tmp_path, video_paths=[str(new)])
    assert "do not exist" in req.check(ctx)
    done = req.derive(ctx)
    assert done.resave and Path(stream.video_path) == new.resolve()
    assert Path(stream.metadata["frame_index"]["0"]) == (new / "0_00000.png").resolve()
    assert req.check(ctx) is None


# ------------------------------------------------------------------ path settings

def test_a_dataset_root_is_fingerprinted_by_content_not_location(tmp_path) -> None:
    root = tmp_path / "a" / "spin"
    root.mkdir(parents=True)
    (root / "dataset.json").write_text("{}")
    before = path_identity(root)
    moved = tmp_path / "b" / "spin"
    moved.parent.mkdir()
    shutil.move(str(root), str(moved))
    assert path_identity(moved) == before
    target = moved / "dataset.json"
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    assert path_identity(moved) != before
    cfg = {"geometry": {"backend_name": "dataset", "dataset_root": str(moved)}}
    settings = effective_settings("geometry", cfg)
    assert settings["geometry"]["dataset_root"] == path_identity(moved)


# ------------------------------------------------------------------ crop boxes (B)

def _geometry_with_depth(backend, depth_hw, *, stream="0", **metadata):
    return SimpleNamespace(
        metadata={"capability": {"backend_used": backend}, **metadata},
        selected_frames=[SimpleNamespace(stream_id=stream)],
        depthmaps_by_frame={f"{stream}:0": {"depth": np.zeros(depth_hw, np.float32)}},
        notes=["Used VGGT-Long globally-aligned camera_poses.txt for camera poses."],
    )


def _ingest(w, h, stream="0"):
    return SimpleNamespace(streams=[SimpleNamespace(stream_id=stream, width=w, height=h)])


def _derive_crop(stage, geometry, tmp_path, *, ingest, settings=None, artifacts=None):
    req = _requirement(stage, "its crop box is recorded")
    ctx = ResumeContext(
        stage, geometry, {"ingest": ingest, **(artifacts or {})},
        {stage: {"settings": {"geometry": settings or {}}}}, {}, tmp_path / stage, [],
    )
    assert "made before crop boxes were recorded" in req.check(ctx)
    done = req.derive(ctx)
    assert req.check(ctx) is None
    return done


def test_an_old_vggt_long_geometry_gets_the_crop_its_loader_applied(tmp_path) -> None:
    from reifyscene.geometry.image_transform import ImageTransform

    geo = _geometry_with_depth("vggt-long", (518, 518), vggt_long_resize_max_dim=None)
    done = _derive_crop("geometry", geo, tmp_path, ingest=_ingest(720, 960))
    record = geo.metadata["image_transforms"]["0"]
    assert record["source"] == "derived at resume"
    t = ImageTransform.from_dict(record)
    assert t.output_size == (518, 518) and not t.is_pure_resize  # 518x686, then rows 84..602
    assert [op.to_dict() for op in t.ops][1] == {"op": "crop", "box": [0, 84, 518, 602]}
    assert "upstream loader" in done.note and not done.resave


def test_an_old_dataset_geometry_is_a_plain_resize(tmp_path) -> None:
    from reifyscene.geometry.image_transform import ImageTransform

    geo = _geometry_with_depth("dataset", (480, 360))
    _derive_crop("geometry", geo, tmp_path, ingest=_ingest(720, 960))
    t = ImageTransform.from_dict(geo.metadata["image_transforms"]["0"])
    assert t.is_pure_resize and t.output_size == (360, 480)


def test_a_crop_that_does_not_land_on_the_depth_maps_is_refused(tmp_path) -> None:
    geo = _geometry_with_depth("vggt-long", (392, 392), vggt_long_resize_max_dim=None)
    with pytest.raises(NotDerivable, match="lands on"):
        _derive_crop("geometry", geo, tmp_path, ingest=_ingest(720, 960))


def test_pose_refine_shares_its_geometrys_crop_box(tmp_path) -> None:
    geo = _geometry_with_depth("vggt-long", (518, 518), vggt_long_resize_max_dim=None)
    _derive_crop("geometry", geo, tmp_path, ingest=_ingest(720, 960))
    refined = _geometry_with_depth("vggt-long", (518, 518))
    _derive_crop("pose_refine", refined, tmp_path, ingest=_ingest(720, 960), artifacts={"geometry": geo})
    assert refined.metadata["image_transforms"] == geo.metadata["image_transforms"]


def _loader_ctx(tmp_path, size, **metadata):
    geo = _geometry_with_depth("vggt-long", (252, 252), **metadata)
    return ResumeContext(
        "geometry", geo, {"ingest": _ingest(*size)}, {"geometry": {"settings": {"geometry": {}}}},
        {}, tmp_path / "geometry", [],
    )


def test_old_no_upscale_geometry_of_portrait_frames_is_refused(tmp_path) -> None:
    req = _requirement("geometry", "its frames went in as today's loader takes them")
    old_portrait = _loader_ctx(tmp_path, (720, 960), vggt_long_resize_max_dim=252)
    assert "went in squashed" in req.check(old_portrait)
    assert req.derive is None
    # Landscape frames go in the same way under both loaders: nothing to refuse.
    assert req.check(_loader_ctx(tmp_path, (960, 720), vggt_long_resize_max_dim=252)) is None
    # Today's loader (recorded as version 2), and the full-size upstream loader, are current.
    today = _loader_ctx(tmp_path, (720, 960), vggt_long_resize_max_dim=252,
                        vggt_long_input_loader={"loader": "bounded", "target_size": 252, "version": 2})
    assert req.check(today) is None
    assert req.check(_loader_ctx(tmp_path, (720, 960), vggt_long_resize_max_dim=None)) is None


def test_a_diverged_bundle_adjustment_is_refused() -> None:
    from reifyscene.geometry.pose_refine import bundle_adjustment_diverged

    limits = dict(max_reproj_px=8.0, max_rot_delta_deg=90.0)
    assert bundle_adjustment_diverged(0.883, [3.4, 34.9], **limits) is None  # the converged spin run
    assert "reprojection" in bundle_adjustment_diverged(6.2e149, [4.0, 160.2], **limits)
    assert "reprojection" in bundle_adjustment_diverged(float("nan"), [1.0], **limits)
    assert "turned 160.2" in bundle_adjustment_diverged(0.9, [4.0, 160.2], **limits)


def test_a_kept_diverged_refinement_is_rejected_and_a_refused_one_loads(tmp_path) -> None:
    diverged = _geometry({}, pose_refine={
        "enabled": True, "refined_streams": ["0"],
        "stats": {"0": {"reanchor_scale": 1.0, "mean_reproj_px": 6.2e149,
                        "seed_to_refined_rot_delta_deg_max": 160.2}},
    })
    req = _requirement("pose_refine", "its bundle adjustment converged")
    assert "diverged" in req.check(_ctx("pose_refine", diverged, tmp_path))
    refused = _geometry({}, pose_refine={
        "enabled": True, "refined_streams": [], "stats": {},
        "rejected_streams": {"0": "final mean reprojection error 6.2e+149 px exceeds 8 px"},
    })
    ctx = _ctx("pose_refine", refused, tmp_path)
    assert all(r.check(ctx) is None for r in REQUIREMENTS["pose_refine"] if r.name in (
        "the refinement actually ran", "its bundle adjustment converged"))


def test_a_pass_through_pose_refine_upgrades_but_a_refined_one_cannot(tmp_path) -> None:
    from reifyscene.api.requirements import UPGRADES, NotDerivable

    upgrade = UPGRADES["pose_refine"][1]
    # Nothing was refined, so version 2's bundle adjustment would change nothing.
    disabled = _geometry({}, pose_refine={"enabled": False, "refined_streams": []})
    assert "passed the geometry through" in upgrade(_ctx("pose_refine", disabled, tmp_path)).note
    refused = _geometry({}, pose_refine={
        "enabled": True, "refined_streams": [],
        "rejected_streams": {"0": "final mean reprojection error 6.2e+149 px exceeds 8 px"},
    })
    assert "refined nothing" in upgrade(_ctx("pose_refine", refused, tmp_path)).note
    # A run that did refine has poses from the old solve; only re-running gets the new ones.
    refined = _geometry({}, pose_refine={
        "enabled": True, "refined_streams": ["0"],
        "stats": {"0": {"reanchor_scale": 1.0, "mean_reproj_px": 0.88}},
    })
    with pytest.raises(NotDerivable, match="re-run pose_refine"):
        upgrade(_ctx("pose_refine", refined, tmp_path))


def test_an_old_oracle_geometry_is_retimed_onto_the_ingest_clock(tmp_path) -> None:
    from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState

    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=5.0, cy=5.0)

    def state(t, x):
        pose = np.eye(4)
        pose[0, 3] = x
        return TimedCameraState(stream_id="0", time_sec=t, intrinsics=intr, pose=CameraPose(T_world_cam=pose))

    # The old dataset backend: cameras at time = frame number (0, 1, 2), plus a camera it
    # densified for tracking frame 1 at 1/30 s -- interpolated between dataset frames 0 and 1.
    geo = SimpleNamespace(
        metadata={"capability": {"backend_used": "dataset"}},
        selected_frames=[FrameRef("0", i, float(i)) for i in range(3)],
        camera_states_by_stream={"0": [state(0.0, 0.0), state(1 / 30, 1 / 30), state(1.0, 1.0), state(2.0, 2.0)]},
    )
    ingest = SimpleNamespace(streams=[SimpleNamespace(stream_id="0", fps=30.0, width=10, height=10, num_frames=3)])
    ctx = ResumeContext("geometry", geo, {"ingest": ingest}, {}, {"tracking": {}}, tmp_path / "geometry", [])
    req = _requirement("geometry", "its cameras are on the ingest clock")
    assert "not on the ingest clock" in req.check(ctx)
    done = req.derive(ctx)
    assert done.revision == "camera-clock" and done.resave
    states = geo.camera_states_by_stream["0"]
    assert [round(st.time_sec, 6) for st in states] == [0.0, 0.033333, 0.066667]
    # Each camera is its own dataset frame's pose, not an interpolation between the wrong ones.
    assert [st.pose.T_world_cam[0, 3] for st in states] == [0.0, 1.0, 2.0]
    assert req.check(ctx) is None

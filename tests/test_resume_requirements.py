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

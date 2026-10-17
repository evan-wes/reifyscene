"""What a loaded artifact must carry for a resume to load it (reifyscene/api/requirements.py).

Every requirement is a check: met, or the resume rejects. Nothing is derived or repaired.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reifyscene.api.contracts import effective_settings, path_identity
from reifyscene.api.requirements import REQUIREMENTS, ResumeContext
from reifyscene.common.data_dir import record_name


def _ctx(stage, artifact, *, cfg=None, artifacts=None):
    return ResumeContext(stage, artifact, artifacts or {}, cfg or {})


def _requirement(stage, name):
    return next(r for r in REQUIREMENTS[stage] if r.name == name)


def _geometry(debug_image_paths, backend="vggt-long", **metadata):
    return SimpleNamespace(
        metadata={"capability": {"backend_used": backend},
                  "debug_image_paths": dict(debug_image_paths), **metadata},
        notes=["Used VGGT-Long globally-aligned camera_poses.txt for camera poses."],
        selected_frames=[], depthmaps_by_frame={},
    )


def _png(path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))
    return path


def test_photos_that_do_not_exist_reject(tmp_path) -> None:
    there = _png(tmp_path / "a.png")
    req = _requirement("geometry", "its photos exist")
    assert req.check(_ctx("geometry", _geometry({"0:0": record_name(there)}))) is None
    gone = _geometry({"0:0": record_name(there), "0:1": record_name(tmp_path / "nowhere.png")})
    assert "1 of its 2 photos do not exist" in req.check(_ctx("geometry", gone))


@pytest.mark.parametrize(
    ("stage", "artifact", "problem"),
    [
        ("geometry", _geometry({}, backend="empty"), "empty floor"),
        ("geometry", SimpleNamespace(metadata={"debug_image_paths": {}}, notes=[],
                                     selected_frames=[], depthmaps_by_frame={}),
         "does not record which geometry backend ran"),
        ("geometry", SimpleNamespace(metadata={"capability": {"backend_used": "vggt-long"}},
                                     notes=[], selected_frames=[], depthmaps_by_frame={}),
         "chunk-local poses"),
        ("pose_refine", _geometry({}, pose_refine={"enabled": True, "refined_streams": []}),
         "refined nothing"),
        ("humans", SimpleNamespace(metadata={}), "unknown backend"),
        ("dyn_mask", SimpleNamespace(metadata={"capability": {"backend_used": "all-static"}}),
         None),
    ],
)
def test_unmet_requirements_reject(stage, artifact, problem) -> None:
    found = [f for f in (r.check(_ctx(stage, artifact)) for r in REQUIREMENTS[stage]) if f]
    if problem is None:
        assert found == []
    else:
        assert any(problem in f for f in found), found


def test_a_detector_run_does_not_pass_for_an_all_static_fallback() -> None:
    art = SimpleNamespace(metadata={"capability": {"backend_used": "all-static"}})
    ctx = _ctx("dyn_mask", art, cfg={"dyn_mask": {"backend_name": "detector"}})
    assert "a fallback ran" in REQUIREMENTS["dyn_mask"][0].check(ctx)


def test_reid_must_embed_exactly_segments_tracks() -> None:
    segment = SimpleNamespace(observations=[SimpleNamespace(track_id=t) for t in ("t0", "t1")])
    reid = SimpleNamespace(embeddings={"t0": np.zeros(2)}, merge_map={})
    ctx = _ctx("reid", reid, artifacts={"segment": segment})
    assert "missing ['t1']" in REQUIREMENTS["reid"][0].check(ctx)
    reid.embeddings["t1"] = np.zeros(2)
    assert REQUIREMENTS["reid"][0].check(ctx) is None


def test_a_scene_that_says_dynamic_needs_its_bundle() -> None:
    scene = SimpleNamespace(kind="dynamic", dynamic_bundle={}, train_views=[])
    assert "has no dynamic bundle" in REQUIREMENTS["scene"][0].check(_ctx("scene", scene))


def test_an_input_that_is_not_where_ingest_recorded_it_rejects(tmp_path) -> None:
    from reifyscene.ingest.decoder import VideoDecoder

    folder = _png(tmp_path / "1x" / "0_00000.png").parent
    ingest = SimpleNamespace(streams=[VideoDecoder().decode(folder, stream_id="cam_00")])
    req = _requirement("ingest", "its input frames exist")
    assert req.check(_ctx("ingest", ingest)) is None
    shutil.move(str(folder), str(tmp_path / "moved"))
    assert "do not exist" in req.check(_ctx("ingest", ingest))


def test_a_crop_box_must_be_recorded_for_every_stream_and_land_on_the_depth() -> None:
    from reifyscene.geometry.image_transform import ImageTransform, Resize

    geo = SimpleNamespace(
        metadata={"capability": {"backend_used": "vggt"}},
        selected_frames=[SimpleNamespace(stream_id="0")],
        depthmaps_by_frame={"0:0": {"depth": np.zeros((3, 4), np.float32)}},
    )
    req = _requirement("geometry", "its crop box is recorded and consistent")
    assert "does not record how stream(s) ['0']" in req.check(_ctx("geometry", geo))
    geo.metadata["image_transforms"] = {"0": ImageTransform((8, 6), (Resize((4, 3)),)).to_dict()}
    assert req.check(_ctx("geometry", geo)) is None
    geo.metadata["image_transforms"] = {"0": ImageTransform((8, 6), (Resize((5, 3)),)).to_dict()}
    assert "lands on (5, 3)" in req.check(_ctx("geometry", geo))


def test_a_dataset_root_is_fingerprinted_by_content_not_location(tmp_path, monkeypatch) -> None:
    """A config names a dataset inside the data directory's datasets/; moving it keeps its
    fingerprint, editing a file in it does not."""
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    datasets = tmp_path / "datasets"
    root = datasets / "a" / "spin"
    root.mkdir(parents=True)
    (root / "dataset.json").write_text("{}")
    before = path_identity("a/spin")
    (datasets / "b").mkdir()
    shutil.move(str(root), str(datasets / "b" / "spin"))
    assert path_identity("b/spin") == before
    target = datasets / "b" / "spin" / "dataset.json"
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    assert path_identity("b/spin") != before
    cfg = {"geometry": {"backend_name": "dataset", "dataset_root": "b/spin"}}
    settings = effective_settings("geometry", cfg)
    assert settings["geometry"]["dataset_root"] == path_identity("b/spin")


def test_a_diverged_bundle_adjustment_is_refused() -> None:
    from reifyscene.geometry.pose_refine import bundle_adjustment_diverged

    limits = dict(max_reproj_px=8.0, max_rot_delta_deg=90.0)
    # the converged spin run
    assert bundle_adjustment_diverged(0.883, [3.4, 34.9], **limits) is None
    assert "reprojection" in bundle_adjustment_diverged(6.2e149, [4.0, 160.2], **limits)
    assert "reprojection" in bundle_adjustment_diverged(float("nan"), [1.0], **limits)
    assert "turned 160.2" in bundle_adjustment_diverged(0.9, [4.0, 160.2], **limits)

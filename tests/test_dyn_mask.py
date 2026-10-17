"""Tests for FP6b step 3: dyn_mask stage/ladder, size-gated COLMAP masking, and the
GT-free reconstruction health-check. No torchvision/COLMAP/GPU needed (detector +
reconstruct are mocked)."""

from __future__ import annotations

import cv2
import numpy as np

from reifyscene.common import data_dir
from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.colmap_wrapper import ColmapPoseInitializer
from reifyscene.geometry.compose import GeometryComposer
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.reconstruction_health import assess_pose_health
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.perception.dyn_mask.dyn_mask_artifact import DynMaskArtifact
from reifyscene.perception.dyn_mask.rungs import AllStaticDynMask, DetectorDynMask


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write_video(path, *, width=32, height=24, num_frames=2, value=100):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    assert writer.isOpened()
    for _ in range(num_frames):
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()


def _ingest(tmp_path, num_frames=2):
    video = tmp_path / "in.mp4"
    _write_video(video, num_frames=num_frames)
    stream = VideoDecoder().decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="t", stage_name="ingest", save_dir=tmp_path / "ingest", streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(num_frames)],
    )


def _rot_z(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _states(angles_deg, stream="cam_00"):
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0)
    out = []
    cum = 0.0
    for i, d in enumerate(angles_deg):
        cum += d
        out.append(TimedCameraState(stream_id=stream, frame_idx=i, time_sec=float(i) / 30.0,
                                    intrinsics=intr,
                                    pose=CameraPose(T_world_cam=_rot_z(cum))))
    return {stream: out}


# --------------------------------------------------------------------------- #
# artifact + rungs
# --------------------------------------------------------------------------- #

def test_dyn_mask_artifact_roundtrip(tmp_path):
    m = np.zeros((8, 8), np.uint8)
    m[2:6, 2:6] = 1
    art = DynMaskArtifact(
        run_id="t", stage_name="dyn_mask", save_dir=tmp_path / "dm", backend_name="detector",
        selected_frames=[FrameRef("cam_00", 0, 0.0)],
        masks_by_frame={"cam_00:0": m}, dynamic_fraction_by_frame={"cam_00:0": float(m.mean())},
    )
    art.save()
    loaded = DynMaskArtifact.load(tmp_path / "dm")
    assert loaded.backend_name == "detector"
    assert np.array_equal(loaded.mask_for("cam_00:0"), m)
    assert abs(loaded.aggregate_dynamic_fraction() - float(m.mean())) < 1e-6


def test_all_static_emits_empty_masks(tmp_path):
    ingest = _ingest(tmp_path)
    art = AllStaticDynMask({}).initialize(ingest.keyframes, ingest_artifact=ingest)
    assert art.backend_name == "all-static"
    assert art.masks_by_frame == {} and art.aggregate_dynamic_fraction() == 0.0


def test_detector_mask_frame_unions_only_movable(monkeypatch):
    rung = DetectorDynMask({})

    class _FakeDetector:
        def detect(self, rgb):
            h, w = rgb.shape[:2]
            person = np.full((h, w), -5.0)
            person[1:4, 1:4] = 5.0            # positive logits → masked
            chair = np.full((h, w), 5.0)      # all positive but NOT a movable category
            return [
                {"category": "person", "mask_logits": person},
                {"category": "chair", "mask_logits": chair},
            ]

    monkeypatch.setattr(rung, "_load_detector", lambda: _FakeDetector())
    mask, frac = rung.mask_frame(np.zeros((6, 6, 3), np.uint8))
    assert mask[2, 2] == 1 and mask[0, 0] == 0   # person region only; chair ignored
    assert frac == float(mask.mean()) and 0.0 < frac < 1.0


# --------------------------------------------------------------------------- #
# reconstruction health-check
# --------------------------------------------------------------------------- #

def test_health_ok_on_smooth_trajectory():
    h = assess_pose_health(_states([0, 1, 1, 1, 1, 1]))  # ~1°/frame
    assert h["healthy"] is True
    assert h["per_stream"]["cam_00"]["flagged"] is False


def test_health_flags_outlier_jump():
    h = assess_pose_health(_states([0, 1, 1, 40, 1, 1]))  # a 40° jump
    assert h["healthy"] is False
    assert h["per_stream"]["cam_00"]["flagged"] is True
    assert h["per_stream"]["cam_00"]["max_deg"] > 30.0


def test_health_too_few_frames_not_flagged():
    h = assess_pose_health(_states([0, 90]))  # below min_frames → no verdict
    assert h["healthy"] is True and h["per_stream"]["cam_00"]["flagged"] is False


# --------------------------------------------------------------------------- #
# COLMAP size-gated masking + health stamp
# --------------------------------------------------------------------------- #

def _dyn_mask_for(ingest, fraction, *, h=24, w=32):
    """Build a DynMaskArtifact whose median dynamic fraction ~= `fraction`."""
    masks, fracs = {}, {}
    for f in ingest.keyframes:
        m = np.zeros((h, w), np.uint8)
        n = int(round(fraction * h * w))
        m.flat[:n] = 1
        masks[f"{f.stream_id}:{f.frame_idx}"] = m
        fracs[f"{f.stream_id}:{f.frame_idx}"] = float(m.mean())
    return DynMaskArtifact(run_id="t", stage_name="dyn_mask", save_dir="unsaved",
                           backend_name="detector", masks_by_frame=masks, dynamic_fraction_by_frame=fracs)


def test_colmap_masks_when_dynamic_fraction_large(tmp_path, monkeypatch):
    ingest = _ingest(tmp_path, num_frames=2)  # frames are solid value 100
    dyn = _dyn_mask_for(ingest, 0.5)          # 50% dynamic ≥ gate 0.10 → mask
    init = ColmapPoseInitializer({"colmap_work_dir": f"{tmp_path.name}/cw"})
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=12.0)
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct",
                        lambda self, img_dir: {0: (np.eye(4, dtype=np.float32), intr)})
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest, dyn_mask_artifact=dyn)
    assert art.metadata["dyn_mask_applied"] is True
    work = data_dir.path(data_dir.CACHE, f"{tmp_path.name}/cw")
    saved = cv2.imread(str(work / "images" / "000000.png"))
    assert (saved == 0).any() and (saved > 0).any()   # masked pixels blacked out, rest kept


def test_colmap_skips_mask_when_dynamic_fraction_small(tmp_path, monkeypatch):
    ingest = _ingest(tmp_path, num_frames=2)
    dyn = _dyn_mask_for(ingest, 0.02)         # 2% dynamic < gate 0.10 → unmasked
    init = ColmapPoseInitializer({"colmap_work_dir": f"{tmp_path.name}/cw"})
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=12.0)
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct",
                        lambda self, img_dir: {0: (np.eye(4, dtype=np.float32), intr)})
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest, dyn_mask_artifact=dyn)
    assert art.metadata["dyn_mask_applied"] is False
    work = data_dir.path(data_dir.CACHE, f"{tmp_path.name}/cw")
    saved = cv2.imread(str(work / "images" / "000000.png"))
    assert not (saved == 0).any()             # nothing blacked out


def test_colmap_stamps_pose_health(tmp_path, monkeypatch):
    ingest = _ingest(tmp_path, num_frames=5)
    init = ColmapPoseInitializer({"colmap_work_dir": f"{tmp_path.name}/cw"})
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=12.0)
    # a degenerate draw: frame 2 jumps 40°
    poses = {i: (_rot_z(0 if i < 2 else 40), intr) for i in range(5)}
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: poses)
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest)
    assert art.metadata["pose_health"]["healthy"] is False
    assert any("SUSPECT" in n for n in art.notes)


# --------------------------------------------------------------------------- #
# pipeline stage + composer downgrade
# --------------------------------------------------------------------------- #

def test_pipeline_dyn_mask_stage_all_static_default(tmp_path):
    ingest = _ingest(tmp_path, num_frames=2)
    pipe = ReifyScenePipeline({"run_dir": tmp_path / "run"})  # default → all-static
    art = pipe.compute_dyn_mask(ingest, sync_artifact=None)
    assert art.backend_name == "all-static"
    assert art.capability is not None and art.capability.backend_used == "all-static"
    assert art.aggregate_dynamic_fraction() == 0.0


def test_composer_downgrades_fidelity_on_suspect_pose_health():
    frames = [FrameRef("cam_00", i, float(i) / 30.0) for i in range(2)]
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0)
    states = [TimedCameraState("cam_00", i, intr, CameraPose(T_world_cam=np.eye(4, dtype=np.float32)))
              for i in range(2)]
    pose_art = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="unsaved", backend_name="colmap",
                                selected_frames=frames, intrinsics_by_stream={"cam_00": intr},
                                camera_states_by_stream={"cam_00": states})
    pose_art.metadata["pose_health"] = {"healthy": False, "reason": "cam_00: max pairwise rot 40° > bound 30°"}
    dm, pm = {}, {}
    for f in frames:
        k = f"{f.stream_id}:{f.frame_idx}"
        dm[k] = {"depth": np.full((8, 8), 2.0, np.float32), "confidence": np.ones((8, 8), np.float32)}
        pts = np.zeros((8, 8, 3), np.float32); pts[..., 2] = 3.0
        pm[k] = {"points": pts, "confidence": np.ones((8, 8), np.float32)}
    depth_art = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="unsaved", backend_name="moge-2",
                                 selected_frames=frames, intrinsics_by_stream={"cam_00": intr},
                                 depthmaps_by_frame=dm, pointmaps_by_frame=pm)
    res = GeometryComposer({"scale_min_obs": 1}).compose(
        pose_artifact=pose_art, depth_artifact=depth_art, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="unsaved")
    assert res.fidelity == "fused_metric_suspect"
    assert res.artifact.metadata["pose_health"]["healthy"] is False

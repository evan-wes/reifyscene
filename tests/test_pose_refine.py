"""Tests for the pose_refine stage wiring and the PoseRefiner contract.

The bundle-adjustment path itself (VGGSfM tracker + pycolmap) needs a GPU and the
optional pose_refine extra, so it is exercised by the throwaway spike, not here.
These tests cover the deterministic, dependency-free surface: config parsing, the
identity pass-through when refinement is disabled, graceful fallback when the
refinement runtime is unavailable, and the keyframe-rebuild apply logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import (
    CameraIntrinsics,
    CameraPose,
    FrameRef,
    TimedCameraState,
    VideoStreamRef,
)
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.pose_refine import (
    PoseRefineConfig,
    PoseRefiner,
    PoseRefineUnavailable,
    StreamRefinement,
)
from reifyscene.ingest.ingest_artifact import IngestArtifact

STREAM_ID = "cam_00"


def _intr(fx: float = 300.0) -> CameraIntrinsics:
    return CameraIntrinsics(fx=fx, fy=fx, cx=64.0, cy=64.0, distortion_model="brown", distortion_params=None)


def _pose(i: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [float(i), 0.0, 2.0 + i]
    return T


def _geometry_artifact(tmp_path: Path, *, with_pointmaps: bool = True) -> GeometryArtifact:
    frames = [FrameRef(STREAM_ID, i, float(i)) for i in range(3)]
    states = [
        TimedCameraState(stream_id=STREAM_ID, time_sec=float(i), intrinsics=_intr(), pose=CameraPose(T_world_cam=_pose(i)))
        for i in range(3)
    ]
    pointmaps: dict = {}
    if with_pointmaps:
        for fr in frames:
            key = f"{STREAM_ID}:{fr.frame_idx}"
            pointmaps[key] = {
                "points": np.zeros((8, 8, 3), dtype=np.float32),
                "confidence": np.ones((8, 8), dtype=np.float32),
            }
    return GeometryArtifact(
        run_id="dev",
        stage_name="geometry",
        save_dir=tmp_path / "geometry",
        metadata={},
        backend_name="vggt",
        selected_frames=frames,
        intrinsics_by_stream={STREAM_ID: _intr()},
        camera_states_by_stream={STREAM_ID: states},
        pointmaps_by_frame=pointmaps,
    )


def _ingest_artifact(tmp_path: Path) -> IngestArtifact:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"reifyscene-test-video")
    return IngestArtifact(
        run_id="dev",
        stage_name="ingest",
        save_dir=tmp_path / "ingest",
        streams=[VideoStreamRef(STREAM_ID, video, 30.0, 10, 128, 128, False)],
        keyframes=[FrameRef(STREAM_ID, i, float(i)) for i in range(3)],
        quality_by_stream={},
        sampled_windows={},
        warnings=[],
    )


def test_config_defaults_and_filtering() -> None:
    cfg = PoseRefineConfig.from_dict({"vis_thresh": 0.5, "refine_pp": True, "enabled": True, "bogus": 1})
    assert cfg.vis_thresh == 0.5
    assert cfg.refine_pp is True
    assert cfg.refine_focal is True  # default: refine shared focal, pp fixed
    assert not hasattr(cfg, "bogus")
    assert not hasattr(cfg, "enabled")  # 'enabled' is a pipeline-level flag, not a refiner field


def test_stream_frames_selects_keyframes_with_pointmaps(tmp_path: Path) -> None:
    refiner = PoseRefiner({})
    geom = _geometry_artifact(tmp_path)
    frames = refiner._stream_frames(geom, STREAM_ID)
    assert [f.frame_idx for f in frames] == [0, 1, 2]
    # frames without a pointmap are not eligible for BA seeding
    geom.pointmaps_by_frame.pop(f"{STREAM_ID}:1")
    assert [f.frame_idx for f in refiner._stream_frames(geom, STREAM_ID)] == [0, 2]


def test_refine_poses_passthrough_when_disabled(tmp_path: Path) -> None:
    pipeline = ReifyScenePipeline({"run_dir": tmp_path / "run"})
    geom = _geometry_artifact(tmp_path)
    ingest = _ingest_artifact(tmp_path)

    refined = pipeline.refine_poses(ingest, geom)

    assert refined.stage_name == "pose_refine"
    assert refined.save_dir == tmp_path / "run" / "pose_refine"
    assert (tmp_path / "run" / "pose_refine" / "artifact.json").exists()
    assert refined.metadata["pose_refine"]["enabled"] is False
    assert refined.metadata["pose_refine"]["refined_streams"] == []
    assert any("disabled" in note.lower() for note in refined.notes)

    orig = geom.camera_states_by_stream[STREAM_ID]
    new = refined.camera_states_by_stream[STREAM_ID]
    assert len(new) == len(orig)
    for a, b in zip(orig, new):
        assert np.allclose(a.pose.T_world_cam, b.pose.T_world_cam)
    assert refined.intrinsics_by_stream[STREAM_ID].fx == geom.intrinsics_by_stream[STREAM_ID].fx


def test_refine_poses_falls_back_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    pipeline = ReifyScenePipeline({"run_dir": tmp_path / "run", "pose_refine": {"enabled": True}})
    geom = _geometry_artifact(tmp_path)
    ingest = _ingest_artifact(tmp_path)

    def _raise_unavailable() -> None:
        raise PoseRefineUnavailable("refinement deps absent in test env")

    monkeypatch.setattr(PoseRefiner, "_import_or_unavailable", staticmethod(_raise_unavailable))

    refined = pipeline.refine_poses(ingest, geom)

    assert refined.metadata["pose_refine"]["refined_streams"] == []
    assert any("unavailable" in note.lower() for note in refined.notes)
    new = refined.camera_states_by_stream[STREAM_ID]
    assert np.allclose(new[0].pose.T_world_cam, geom.camera_states_by_stream[STREAM_ID][0].pose.T_world_cam)


def test_reanchor_sim3_recovers_gauge_drift() -> None:
    """The BA-gauge re-anchor recovers a global-similarity drift exactly.

    pycolmap BA optimizes reprojection, which is invariant to a global similarity
    of {poses, points}; with an unpinned datum the overall scale walks freely
    (thousands x). The fix maps the refined poses back onto the seed frame with an
    orientation-based Sim(3). Here we apply a *known* drift (scale + rotation +
    translation) to a seed pose set and assert the re-anchor inverts it, so the
    recovered poses land back on the seed (the relative geometry is preserved).
    """
    from reifyscene.geometry.pose_refine import _apply_sim3_to_c2w, _sim3_src_to_dst

    seed_c2w, seed_w2c = {}, {}
    for i in range(6):
        ang, tilt = 0.4 * i, 0.15 * i
        rz = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]])
        c2w = np.eye(4)
        c2w[:3, :3] = rz @ rx
        c2w[:3, 3] = [np.cos(ang), np.sin(ang), 0.3 * i]
        seed_c2w[str(i)] = c2w
        seed_w2c[str(i)] = np.linalg.inv(c2w)

    # Known gauge drift: 5000x scale + a global rotation + a translation.
    s_true, a = 5000.0, 0.7
    R_true = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    t_true = np.array([12.0, -3.0, 7.0])
    drift_w2c = {}
    for k, c2w in seed_c2w.items():
        d = np.eye(4)
        d[:3, :3] = R_true @ c2w[:3, :3]
        d[:3, 3] = s_true * (R_true @ c2w[:3, 3]) + t_true
        drift_w2c[k] = np.linalg.inv(d)

    keys = list(seed_w2c)
    s, R, t = _sim3_src_to_dst(drift_w2c, seed_w2c, keys)
    assert np.isclose(s, 1.0 / s_true, rtol=1e-6)
    for k in keys:
        back = _apply_sim3_to_c2w(np.linalg.inv(drift_w2c[k]), s, R, t)
        assert np.allclose(back, seed_c2w[k], atol=1e-6)


def test_refine_poses_applies_refined_poses(tmp_path: Path, monkeypatch) -> None:
    pipeline = ReifyScenePipeline({"run_dir": tmp_path / "run", "pose_refine": {"enabled": True}})
    geom = _geometry_artifact(tmp_path)
    ingest = _ingest_artifact(tmp_path)

    refined_poses: dict[str, np.ndarray] = {}
    for fr in geom.selected_frames:
        key = f"{STREAM_ID}:{fr.frame_idx}"
        T = geom._find_state_for_frame(fr).pose.T_world_cam.copy()
        T[:3, 3] *= -1.0  # a visibly different pose
        refined_poses[key] = T
    fake = StreamRefinement(
        stream_id=STREAM_ID,
        refined_T_world_cam_by_frame_key=refined_poses,
        refined_intrinsics=_intr(fx=310.0),
        stats={"num_keyframes": 3},
    )

    work_dirs = []

    def fake_refine(self, geometry, *, decoder, ingest_artifact, work_dir=None):
        work_dirs.append(work_dir)
        return {STREAM_ID: fake}

    monkeypatch.setattr(PoseRefiner, "refine", fake_refine)
    # isolate the keyframe-rebuild logic from the tracker-backed densification
    monkeypatch.setattr(ReifyScenePipeline, "_densify_geometry_cameras_for_tracking", lambda self, art, ing: None)

    refined = pipeline.refine_poses(ingest, geom)

    assert refined.metadata["pose_refine"]["refined_streams"] == [STREAM_ID]
    # BA's scratch goes in pose_refine's own folder, not the geometry stage's.
    assert Path(work_dirs[0]) == pipeline._stage_dir("pose_refine")
    new = refined.camera_states_by_stream[STREAM_ID]
    assert len(new) == 3
    for fr, st in zip(geom.selected_frames, new):
        key = f"{STREAM_ID}:{fr.frame_idx}"
        assert np.allclose(st.pose.T_world_cam, refined_poses[key])
        assert st.intrinsics.fx == 310.0
    assert refined.intrinsics_by_stream[STREAM_ID].fx == 310.0
    # the original geometry artifact must be left untouched (refined is a new artifact)
    assert geom.intrinsics_by_stream[STREAM_ID].fx == 300.0

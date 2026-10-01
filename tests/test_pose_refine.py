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
        TimedCameraState(stream_id=STREAM_ID, frame_idx=i, intrinsics=_intr(), pose=CameraPose(T_world_cam=_pose(i)))
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
    from reifyscene.geometry.similarity import apply_similarity_to_c2w, fit_similarity

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
    s, R, t = fit_similarity(drift_w2c, seed_w2c, keys)
    assert np.isclose(s, 1.0 / s_true, rtol=1e-6)
    for k in keys:
        back = apply_similarity_to_c2w(np.linalg.inv(drift_w2c[k]), s, R, t)
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

    def fake_refine(self, geometry, *, ingest_artifact, work_dir=None):
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


def test_a_refused_refinement_keeps_the_seed_poses_and_says_why(tmp_path: Path, monkeypatch) -> None:
    pipeline = ReifyScenePipeline({"run_dir": tmp_path / "run", "pose_refine": {"enabled": True}})
    geom = _geometry_artifact(tmp_path)
    ingest = _ingest_artifact(tmp_path)

    def fake_refine(self, geometry, *, ingest_artifact, work_dir=None):
        self.rejected[STREAM_ID] = "final mean reprojection error 6.2e+149 px exceeds 8 px"
        return {}

    monkeypatch.setattr(PoseRefiner, "refine", fake_refine)
    monkeypatch.setattr(ReifyScenePipeline, "_densify_geometry_cameras_for_tracking", lambda self, art, ing: None)
    refined = pipeline.refine_poses(ingest, geom)
    info = refined.metadata["pose_refine"]
    assert info["refined_streams"] == [] and "6.2e+149" in info["rejected_streams"][STREAM_ID]
    for before, after in zip(geom.camera_states_by_stream[STREAM_ID], refined.camera_states_by_stream[STREAM_ID]):
        assert np.allclose(before.pose.T_world_cam, after.pose.T_world_cam)


# --------------------------------------------------------------------------- #
# The bundle adjustment: which cameras it refines, and whether it repeats.
# --------------------------------------------------------------------------- #
class _FakePoint2D:
    def __init__(self, has: bool) -> None:
        self._has = has

    def has_point3D(self) -> bool:
        return self._has


class _FakeImage:
    def __init__(self, name: str, num_obs: int) -> None:
        self.name = name
        self.points2D = [_FakePoint2D(True) for _ in range(num_obs)] + [_FakePoint2D(False)]


class _FakeRec:
    """The little of a pycolmap Reconstruction that the camera filter reads."""

    def __init__(self, obs_per_image: dict[str, int]) -> None:
        self._images = {
            i + 1: _FakeImage(name, n) for i, (name, n) in enumerate(obs_per_image.items())
        }

    def reg_image_ids(self):
        return list(self._images)

    def image(self, image_id):
        return self._images[image_id]

    def num_reg_frames(self):
        return len(self._images)


class _FakeConfig:
    def __init__(self) -> None:
        self.added = []

    def add_image(self, image_id):
        self.added.append(image_id)


def _patch_pycolmap(monkeypatch):
    """Record what the adjuster is handed instead of solving anything."""
    import pycolmap

    calls = {"config": None, "solved": 0, "convenience": 0}

    class _Adjuster:
        def solve(self):
            calls["solved"] += 1

    def _create(opts, config, rec):
        calls["config"] = config
        return _Adjuster()

    monkeypatch.setattr(pycolmap, "BundleAdjustmentConfig", _FakeConfig)
    monkeypatch.setattr(pycolmap, "create_default_bundle_adjuster", _create)
    def _convenience(rec, opts):
        calls["convenience"] += 1

    monkeypatch.setattr(pycolmap, "bundle_adjustment", _convenience)
    return calls


def test_only_cameras_with_enough_observations_are_refined(monkeypatch) -> None:
    from reifyscene.geometry.pose_refine import _adjust

    calls = _patch_pycolmap(monkeypatch)
    rec = _FakeRec({"a": 200, "b": 0, "c": 1, "d": 50, "e": 2})
    refined, kept_seed = _adjust(rec, object(), min_obs=3, log=lambda m: None)

    assert refined == 2  # only 'a' and 'd' have 3+ observations
    assert kept_seed == [2, 3, 5]  # image ids of 'b', 'c', 'e'
    assert [rec.image(i).name for i in kept_seed] == ["b", "c", "e"]
    assert calls["config"].added == [1, 4]
    assert calls["solved"] == 1 and calls["convenience"] == 0


def test_min_obs_zero_refines_every_registered_camera(monkeypatch) -> None:
    from reifyscene.geometry.pose_refine import _adjust

    calls = _patch_pycolmap(monkeypatch)
    rec = _FakeRec({"a": 200, "b": 0})
    refined, kept_seed = _adjust(rec, object(), min_obs=0, log=lambda m: None)

    # The behaviour from before under-constrained cameras were found to wreck the solve.
    assert refined == 2 and kept_seed == []
    assert calls["convenience"] == 1 and calls["solved"] == 0


def test_a_solve_with_nothing_left_to_constrain_is_skipped(monkeypatch) -> None:
    from reifyscene.geometry.pose_refine import _adjust

    calls = _patch_pycolmap(monkeypatch)
    notes: list[str] = []
    rec = _FakeRec({"a": 200, "b": 0, "c": 1})
    refined, kept_seed = _adjust(rec, object(), min_obs=3, log=notes.append)

    assert refined == 1 and kept_seed == [2, 3]
    assert calls["solved"] == 0 and calls["convenience"] == 0
    assert any("BA skipped" in n for n in notes)


def test_the_solver_thread_count_is_pinned() -> None:
    from reifyscene.geometry.pose_refine import _ba_options

    assert _ba_options(1.0, refine_focal=True, refine_pp=False).solver_options.num_threads == 1
    eight = _ba_options(1.0, refine_focal=True, refine_pp=False, threads=8)
    assert eight.solver_options.num_threads == 8


def test_either_divergence_check_can_be_skipped() -> None:
    from reifyscene.geometry.pose_refine import bundle_adjustment_diverged

    limits = {"max_reproj_px": 8.0, "max_rot_delta_deg": 90.0}
    # Before the re-anchor only the reprojection error is meaningful; after it, only the
    # rotations are checked, and a huge reprojection error is not re-reported.
    assert bundle_adjustment_diverged(0.9, None, **limits) is None
    assert "reprojection" in bundle_adjustment_diverged(6.2e149, None, **limits)
    assert bundle_adjustment_diverged(None, [4.0, 30.0], **limits) is None
    assert "turned 143.0" in bundle_adjustment_diverged(None, [4.0, 143.0], **limits)
    assert bundle_adjustment_diverged(None, None, **limits) is None


def test_rotation_delta_is_the_angle_between_two_orientations() -> None:
    from reifyscene.geometry.pose_refine import _rotation_delta_deg

    def rz(deg):
        t = np.radians(deg)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        return T

    assert _rotation_delta_deg(rz(0), rz(0)) == 0.0
    assert abs(_rotation_delta_deg(rz(30), rz(0)) - 30.0) < 1e-9
    assert abs(_rotation_delta_deg(rz(-45), rz(45)) - 90.0) < 1e-9


def test_the_reanchor_leaves_an_unrefined_camera_exactly_where_it_was() -> None:
    """A camera the bundle adjustment never moved must come out at its seed pose.

    The Sim(3) undoes the gauge drift of the cameras BA *did* move. A camera it skipped
    (too few observations to determine a pose) never drifted, so putting it through the
    same correction would displace it by exactly that drift -- which is what happened
    before the fit and the correction were restricted to the moved cameras.
    """
    from reifyscene.geometry.pose_refine import _reanchor_to_seed

    seed_c2w, seed_w2c = {}, {}
    for i in range(5):
        ang = 0.3 * i
        c2w = np.eye(4)
        c2w[:3, :3] = [
            [np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]
        ]
        c2w[:3, 3] = [np.cos(ang), np.sin(ang), 0.4 * i]
        seed_c2w[str(i)], seed_w2c[str(i)] = c2w, np.linalg.inv(c2w)

    # BA moved cameras 0-3 by a global similarity and never touched camera 4.
    moved = ["0", "1", "2", "3"]
    s_true, a = 40.0, 0.5
    R_true = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    t_true = np.array([2.0, -1.0, 0.5])
    refined, refined_w2c = {}, {}
    for key, c2w in seed_c2w.items():
        if key in moved:
            drifted = np.eye(4)
            drifted[:3, :3] = R_true @ c2w[:3, :3]
            drifted[:3, 3] = s_true * (R_true @ c2w[:3, 3]) + t_true
        else:
            drifted = c2w.copy()
        refined[key] = drifted
        refined_w2c[key] = np.linalg.inv(drifted)

    untouched_before = refined["4"].copy()
    s, _, _ = _reanchor_to_seed(refined, refined_w2c, seed_w2c, moved)

    assert np.isclose(s, 1.0 / s_true, rtol=1e-6)
    for key in moved:
        assert np.allclose(refined[key], seed_c2w[key], atol=1e-6)
    # The skipped camera is untouched, to the bit.
    assert np.array_equal(refined["4"], untouched_before)
    assert np.array_equal(refined["4"], seed_c2w["4"])


def test_a_landscape_geometry_is_padded_square_without_moving_its_pixels() -> None:
    """The tracker needs a square frame; padding must not move a real pixel."""
    import torch

    from reifyscene.geometry.pose_refine import _pad_to_square

    images = torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7)  # H=4, W=7
    conf = np.arange(2 * 4 * 7, dtype=np.float32).reshape(2, 4, 7)
    points = np.arange(2 * 4 * 7 * 3, dtype=np.float32).reshape(2, 4, 7, 3)

    out_images, out_conf, out_points, pad_h, pad_w = _pad_to_square(images, conf, points)

    assert (pad_h, pad_w) == (3, 0)
    assert tuple(out_images.shape) == (2, 3, 7, 7)
    assert out_conf.shape == (2, 7, 7) and out_points.shape == (2, 7, 7, 3)
    # Every original pixel keeps its coordinates, so a track needs no unpadding.
    assert torch.equal(out_images[:, :, :4, :7], images)
    assert np.array_equal(out_conf[:, :4, :7], conf)
    assert np.array_equal(out_points[:, :4, :7], points)
    # The padding carries no confidence, so the tracker's own filter drops it.
    assert not out_conf[:, 4:, :].any()


def test_a_square_geometry_is_left_alone() -> None:
    import torch

    from reifyscene.geometry.pose_refine import _pad_to_square

    images = torch.zeros(2, 3, 5, 5)
    conf = np.zeros((2, 5, 5), dtype=np.float32)
    points = np.zeros((2, 5, 5, 3), dtype=np.float32)
    out_images, out_conf, out_points, pad_h, pad_w = _pad_to_square(images, conf, points)
    assert (pad_h, pad_w) == (0, 0)
    assert out_images is images and out_conf is conf and out_points is points

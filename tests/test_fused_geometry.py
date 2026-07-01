"""Tests for FP6b step 2: pose/depth ladder split + COLMAP↔MoGe scale reconciliation.

Three layers, none needing hloc/COLMAP/MoGe or a GPU:
- the pure scale solver on synthetic observations (`solve_global_scale`);
- `GeometryComposer` fusion + its poses-only / depth-only / empty degradations
  (COLMAP's sparse read is monkeypatched);
- the pipeline's fused mode wiring (both channel backends mocked) → composite manifest.
"""

from __future__ import annotations

import cv2
import numpy as np

from room4d.api.pipeline import Room4DPipeline
from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from room4d.geometry import compose as compose_mod
from room4d.geometry.colmap_wrapper import ColmapPoseInitializer
from room4d.geometry.compose import GeometryComposer
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.geometry.moge_wrapper import MoGeInitializer
from room4d.geometry.scale_reconcile import solve_global_scale
from room4d.ingest.decoder import VideoDecoder
from room4d.ingest.ingest_artifact import IngestArtifact


# --------------------------------------------------------------------------- #
# pure solver
# --------------------------------------------------------------------------- #

def test_solve_global_scale_recovers_known_scale():
    # depth = k * colmap_depth at every observation → global scale == k, residual ~ 1.
    k = 3.0
    rng = np.random.default_rng(0)
    obs, depth = {}, {}
    for f in range(4):
        key = f"cam_00:{f}"
        d = np.zeros((16, 16), dtype=np.float32)
        samples = []
        for _ in range(20):
            u, v = int(rng.integers(0, 16)), int(rng.integers(0, 16))
            z_colmap = float(rng.uniform(0.5, 5.0))
            d[v, u] = k * z_colmap
            samples.append((float(u), float(v), z_colmap))
        obs[key] = samples
        depth[key] = d
    res = solve_global_scale(obs, depth, min_obs=8)
    assert res.status == "reconciled"
    assert abs(res.scale - k) < 1e-6
    assert abs(res.residual_median - 1.0) < 1e-6
    assert res.n_frames == 4 and res.n_obs == 80


def test_solve_global_scale_identity_fallback_when_too_few():
    res = solve_global_scale({"cam_00:0": [(1.0, 1.0, 2.0)]}, {"cam_00:0": np.full((4, 4), 4.0, np.float32)}, min_obs=32)
    assert res.status == "identity_fallback" and res.scale == 1.0


def test_solve_global_scale_skips_out_of_bounds_and_nonfinite():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    obs = {"cam_00:0": [(0.0, 0.0, 1.0), (99.0, 99.0, 1.0), (2.0, 2.0, 1.0), (3.0, 3.0, 1.0)]}
    res = solve_global_scale(obs, {"cam_00:0": depth}, min_obs=2)
    # only the two in-bounds finite pixels (2,2) and (3,3) survive → ratio 2.0
    assert res.status == "reconciled" and abs(res.scale - 2.0) < 1e-9 and res.n_obs == 2


# --------------------------------------------------------------------------- #
# composer
# --------------------------------------------------------------------------- #

def _frames(n=2, stream="cam_00"):
    return [FrameRef(stream_id=stream, frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(n)]


def _pose_state(stream, t, center):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.asarray(center, dtype=np.float32)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0)
    return TimedCameraState(stream_id=stream, time_sec=float(t), intrinsics=intr, pose=CameraPose(T_world_cam=T))


def _pose_artifact(frames, centers, *, model_dir="dummy"):
    states = [_pose_state(f.stream_id, f.timestamp_sec, c) for f, c in zip(frames, centers)]
    art = GeometryArtifact(
        run_id="t", stage_name="geometry", save_dir="/tmp/x", backend_name="colmap",
        selected_frames=list(frames),
        intrinsics_by_stream={frames[0].stream_id: states[0].intrinsics},
        camera_states_by_stream={frames[0].stream_id: states},
    )
    if model_dir is not None:
        art.metadata["colmap_model_dir"] = model_dir
    return art


def _depth_artifact(frames, *, depth_val=2.0, cam_point=(0.0, 0.0, 3.0), h=8, w=8):
    dm, pm = {}, {}
    for f in frames:
        key = f"{f.stream_id}:{f.frame_idx}"
        dm[key] = {"depth": np.full((h, w), depth_val, np.float32), "confidence": np.ones((h, w), np.float32)}
        pts = np.zeros((h, w, 3), np.float32)
        pts[..., :] = np.asarray(cam_point, np.float32)
        pm[key] = {"points": pts, "confidence": np.ones((h, w), np.float32)}
    return GeometryArtifact(
        run_id="t", stage_name="geometry", save_dir="/tmp/x", backend_name="moge-2",
        selected_frames=list(frames),
        intrinsics_by_stream={frames[0].stream_id: CameraIntrinsics(fx=90.0, fy=90.0, cx=4.0, cy=4.0)},
        depthmaps_by_frame=dm, pointmaps_by_frame=pm,
    )


def test_compose_full_fusion_reconciles_scale(monkeypatch):
    frames = _frames(2)
    # frame1's COLMAP center is 1 unit; depth (2.0) / colmap_depth (1.0) → scale 2.0.
    pose_art = _pose_artifact(frames, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    depth_art = _depth_artifact(frames, depth_val=2.0, cam_point=(0.0, 0.0, 3.0))

    def _fake_extract(model_dir, index_to_key):
        # >= min_obs observations, all with z_colmap=1.0 so depth/z = 2.0
        return {key: [(2.0, 2.0, 1.0)] * 8 for key in index_to_key.values()}

    monkeypatch.setattr(compose_mod, "extract_colmap_observations", _fake_extract)
    res = GeometryComposer({"scale_min_obs": 4}).compose(
        pose_artifact=pose_art, depth_artifact=depth_art, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="/tmp/x")

    assert res.fidelity == "fused_metric"
    assert res.artifact.backend_name == "colmap+moge-2"
    assert res.reconcile is not None and abs(res.reconcile.scale - 2.0) < 1e-9
    # poses scaled: frame1 center 1.0 → 2.0
    states = res.artifact.camera_states_by_stream["cam_00"]
    centers = sorted(float(s.pose.T_world_cam[0, 3]) for s in states)
    assert centers == [0.0, 2.0]
    # world pointmaps: identity rot, cam point (0,0,3) + scaled center.
    pm1 = res.artifact.pointmaps_by_frame["cam_00:1"]["points"]
    assert np.allclose(pm1[0, 0], [2.0, 0.0, 3.0])
    pm0 = res.artifact.pointmaps_by_frame["cam_00:0"]["points"]
    assert np.allclose(pm0[0, 0], [0.0, 0.0, 3.0])
    # camera_states carry the pose-source (COLMAP) intrinsics, MoGe's kept as provenance
    assert res.artifact.intrinsics_by_stream["cam_00"].fx == 100.0
    assert res.artifact.metadata["moge_intrinsics_by_stream"]["cam_00"]["fx"] == 90.0
    assert res.artifact.metadata["scale_reconcile"]["status"] == "reconciled"


def test_compose_identity_fallback_without_model(monkeypatch):
    # no colmap_model_dir → extraction yields nothing → s=1.0, still a complete artifact.
    frames = _frames(2)
    pose_art = _pose_artifact(frames, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], model_dir=None)
    depth_art = _depth_artifact(frames)
    res = GeometryComposer().compose(
        pose_artifact=pose_art, depth_artifact=depth_art, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="/tmp/x")
    assert res.fidelity == "fused_metric"
    assert res.reconcile.status == "identity_fallback" and res.reconcile.scale == 1.0
    # unscaled poses, world points = cam + unscaled center
    assert np.allclose(res.artifact.pointmaps_by_frame["cam_00:1"]["points"][0, 0], [1.0, 0.0, 3.0])


def test_compose_depth_only_when_pose_channel_empty():
    frames = _frames(2)
    empty_pose = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="/tmp/x", backend_name="empty", selected_frames=[])
    depth_art = _depth_artifact(frames)
    res = GeometryComposer().compose(
        pose_artifact=empty_pose, depth_artifact=depth_art, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="/tmp/x")
    assert res.fidelity == "depth_only"
    assert res.artifact.backend_name == "colmap+moge-2"
    assert res.artifact.depthmaps_by_frame and not res.artifact.camera_states_by_stream


def test_compose_poses_only_when_depth_channel_empty():
    frames = _frames(2)
    pose_art = _pose_artifact(frames, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    empty_depth = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="/tmp/x", backend_name="empty", selected_frames=[])
    res = GeometryComposer().compose(
        pose_artifact=pose_art, depth_artifact=empty_depth, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="/tmp/x")
    assert res.fidelity == "poses_only"
    assert res.artifact.camera_states_by_stream and not res.artifact.depthmaps_by_frame


def test_compose_empty_when_both_channels_empty():
    frames = _frames(2)
    empty = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="/tmp/x", backend_name="empty", selected_frames=[])
    res = GeometryComposer().compose(
        pose_artifact=empty, depth_artifact=empty, shared_frames=frames,
        pose_backend="colmap", depth_backend="moge-2", run_id="t", save_dir="/tmp/x")
    assert res.fidelity == "empty" and res.artifact.backend_name == "empty"


# --------------------------------------------------------------------------- #
# pipeline fused-mode wiring
# --------------------------------------------------------------------------- #

def _write_video(path, *, width=32, height=24, num_frames=2, fps=5.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., :] = 20 + idx
        writer.write(frame)
    writer.release()


def _ingest(tmp_path, num_frames=2):
    video = tmp_path / "in.mp4"
    _write_video(video, num_frames=num_frames)
    stream = VideoDecoder(default_fps=30.0).decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="t", stage_name="ingest", save_dir=tmp_path / "ingest", streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) / 30.0) for i in range(num_frames)],
    )


def _fake_colmap_poses(indices):
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=12.0)
    out = {}
    for i in indices:
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = float(i)  # a metre-per-frame slide
        out[i] = (T, intr)
    return out


def _fake_moge_infer(self, rgb):
    h, w = rgb.shape[:2]
    return {
        "depth": np.full((h, w), 2.0, np.float32),
        "points": np.tile(np.asarray([0.0, 0.0, 2.0], np.float32), (h, w, 1)),
        "mask": np.ones((h, w), np.float32),
        "intrinsics": CameraIntrinsics(fx=90.0, fy=90.0, cx=w / 2, cy=h / 2),
    }


def test_pipeline_fused_mode_composes_and_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: _fake_colmap_poses([0, 1]))
    monkeypatch.setattr(MoGeInitializer, "infer_frame", _fake_moge_infer)
    ingest = _ingest(tmp_path, num_frames=2)
    pipe = Room4DPipeline({
        "run_dir": tmp_path / "run",
        "geometry": {"pose_backend": "colmap", "depth_backend": "moge", "compose": {"scale_min_obs": 1}},
    })
    assert pipe.geometry_fused
    art = pipe.initialize_geometry(ingest, sync_artifact=None)

    cap = art.capability
    assert cap is not None
    assert cap.backend_used == "colmap+moge-2" == art.backend_name
    assert cap.fidelity == "fused_metric"
    # both channels' rungs recorded, prefixed
    names = {r.name for r in cap.rungs}
    assert "pose:colmap" in names and "depth:moge-2" in names
    # complete geometry: poses + depth + world pointmaps
    assert art.camera_states_by_stream["cam_00"]
    assert art.depthmaps_by_frame and art.pointmaps_by_frame
    # no colmap model dir was produced by the mock → identity-fallback scale, still complete
    assert art.metadata["scale_reconcile"]["status"] == "identity_fallback"


def test_pipeline_fused_pose_failure_degrades_to_depth_only(tmp_path, monkeypatch):
    def _boom(self, img_dir):
        raise RuntimeError("colmap failed")

    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", _boom)
    monkeypatch.setattr(MoGeInitializer, "infer_frame", _fake_moge_infer)
    ingest = _ingest(tmp_path, num_frames=2)
    pipe = Room4DPipeline({
        "run_dir": tmp_path / "run",
        "geometry": {"pose_backend": "colmap", "depth_backend": "moge"},
    })
    art = pipe.initialize_geometry(ingest, sync_artifact=None)  # must not crash
    cap = art.capability
    assert cap.fidelity == "depth_only"
    assert {r.name: r.status for r in cap.rungs}.get("pose:colmap") == "failed"
    assert art.depthmaps_by_frame and not art.camera_states_by_stream


def test_single_backend_path_not_fused(tmp_path, monkeypatch):
    # back-compat: a lone backend_name still selects the single pose-only path.
    monkeypatch.setattr(ColmapPoseInitializer, "reconstruct", lambda self, img_dir: _fake_colmap_poses([0, 1]))
    ingest = _ingest(tmp_path, num_frames=2)
    pipe = Room4DPipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "colmap"}})
    assert not pipe.geometry_fused
    art = pipe.initialize_geometry(ingest, sync_artifact=None)
    assert art.backend_name == "colmap" and art.capability.backend_used == "colmap"
    assert not art.depthmaps_by_frame  # pose-only, unchanged behavior

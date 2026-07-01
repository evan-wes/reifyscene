"""Tests for the FP6a MoGe-2 depth rung + the geometry backend ladder/manifest.

Uses a fake MoGe model (returns MoGe-style *normalized* intrinsics) so the wrapper
+ ladder are exercised without weights or a GPU.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from room4d.api.pipeline import Room4DPipeline
from room4d.common.types import FrameRef
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.geometry.moge_wrapper import MoGeInitializer
from room4d.ingest.decoder import VideoDecoder
from room4d.ingest.ingest_artifact import IngestArtifact


class _FakeMoGe:
    """Stand-in for ``MoGeModel``: emits normalized intrinsics (fx,fy=0.5, cx,cy=0.5)
    so the wrapper's normalized->pixel conversion is exercised."""

    def infer(self, img):  # img is (3, H, W)
        h, w = int(img.shape[-2]), int(img.shape[-1])
        k = torch.eye(3, dtype=torch.float32)
        k[0, 0] = 0.5
        k[1, 1] = 0.5
        k[0, 2] = 0.5
        k[1, 2] = 0.5
        return {
            "depth": torch.full((h, w), 2.0, dtype=torch.float32),
            "points": torch.zeros((h, w, 3), dtype=torch.float32),
            "mask": torch.ones((h, w), dtype=torch.bool),
            "intrinsics": k,
        }


def _write_test_video(path, *, width=32, height=24, num_frames=2, fps=5.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Failed to create test video."
    for idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., :] = 20 + idx
        writer.write(frame)
    writer.release()


def _ingest_for(tmp_path, name="ingest"):
    video = tmp_path / f"{name}.mp4"
    _write_test_video(video)
    stream = VideoDecoder(default_fps=30.0).decode(video, stream_id="cam_00")
    return IngestArtifact(
        run_id="t", stage_name="ingest", save_dir=tmp_path / name,
        streams=[stream],
        keyframes=[FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)],
    )


def test_moge_infer_frame_shapes_and_pixel_intrinsics():
    init = MoGeInitializer({"device": "cpu"})
    init._model = _FakeMoGe()  # inject fake; _load_model returns it (already set)
    res = init.infer_frame(np.zeros((24, 32, 3), dtype=np.uint8))
    assert res["depth"].shape == (24, 32)
    assert res["points"].shape == (24, 32, 3)
    assert res["mask"].shape == (24, 32)
    k = res["intrinsics"]
    assert (k.fx, k.fy) == (0.5 * 32, 0.5 * 24)  # normalized -> pixel (fx*W, fy*H)
    assert (k.cx, k.cy) == (0.5 * 32, 0.5 * 24)


def test_moge_initialize_empty_in_empty_out():
    init = MoGeInitializer({"device": "cpu"})
    art = init.initialize(frames=[], ingest_artifact=None)
    assert isinstance(art, GeometryArtifact)
    assert art.backend_name == "moge-2"
    assert art.depthmaps_by_frame == {} and art.selected_frames == []
    assert init._model is None  # never loaded on the empty path


def test_moge_initialize_fills_depth_but_no_poses(tmp_path):
    ingest = _ingest_for(tmp_path)
    init = MoGeInitializer({"device": "cpu"})
    init._model = _FakeMoGe()
    art = init.initialize(ingest.keyframes, ingest_artifact=ingest, sync_artifact=None)
    assert art.backend_name == "moge-2"
    assert art.depthmaps_by_frame["cam_00:0"]["depth"].shape == (24, 32)
    assert art.pointmaps_by_frame["cam_00:0"]["points"].shape == (24, 32, 3)
    assert art.intrinsics_by_stream["cam_00"].fx == 0.5 * 32
    assert art.camera_states_by_stream == {}  # depth-only rung: no cross-frame poses


def test_geometry_ladder_selects_moge_and_stamps_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(MoGeInitializer, "_load_model", lambda self: _FakeMoGe())
    ingest = _ingest_for(tmp_path)
    pipe = Room4DPipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "moge", "device": "cpu"}})
    assert isinstance(pipe.geometry_init, MoGeInitializer)

    art = pipe.initialize_geometry(ingest, sync_artifact=None)
    cap = art.capability
    assert cap is not None
    assert cap.backend_used == "moge-2" == art.backend_name  # manifest matches artifact
    assert cap.fidelity == "metric_depth_no_poses"
    assert "cam_00:0" in art.depthmaps_by_frame
    assert art.camera_states_by_stream == {}


def test_geometry_ladder_degrades_to_empty_when_rung_fails(tmp_path, monkeypatch):
    def _boom(self):
        raise RuntimeError("moge weights failed to load")

    monkeypatch.setattr(MoGeInitializer, "_load_model", _boom)
    ingest = _ingest_for(tmp_path)
    pipe = Room4DPipeline({"run_dir": tmp_path / "run", "geometry": {"backend_name": "moge", "device": "cpu"}})

    art = pipe.initialize_geometry(ingest, sync_artifact=None)  # must NOT crash (§4A invariant 3)
    cap = art.capability
    assert cap is not None
    assert cap.backend_used == "empty" and art.backend_name == "empty"
    statuses = {r.name: r.status for r in cap.rungs}
    assert statuses.get("moge-2") == "failed"  # the failed rung is observable in the manifest

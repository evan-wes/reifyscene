"""FP5a: honest human wrappers, the dense ladder rungs, and Sapiens decode math.

Covers the F3 de-fabrication (wrappers report honest backend / no fake data), the
dense-ladder frame loop + rungs (fake estimators, no weights), and the pure
Sapiens heatmap/coordinate decoding. The real Sapiens inference is validated by
scripts/fp5_validate_humans.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from room4d.common.types import FrameRef, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.perception.humans.smpl_fitter import SMPLFitter
from room4d.perception.humans.densepose_wrapper import DensePoseEstimator
from room4d.perception.humans.dense_ladder import BboxOnlyRung, KeypointRcnnRung, _frame_loop
from room4d.perception.humans.sapiens_wrapper import SapiensEstimator
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation


# --- honest wrappers (F3 de-fabrication) ------------------------------------

def test_densepose_honest_unavailable() -> None:
    est = DensePoseEstimator()
    out = est.predict(np.zeros((10, 10, 3), np.uint8), [np.array([0, 0, 4, 4], float)])
    assert out == [None]
    assert est.backend_used == 'unavailable'


def test_smpl_honest_skip() -> None:
    fitter = SMPLFitter()
    assert fitter.fit_frame(None, np.zeros((17, 3), np.float32)) is None
    assert fitter.backend_used == 'skip'
    assert fitter.fit_sequence([object(), object()]) == [None, None]


# --- dense ladder rungs (fake estimators, no weights) -----------------------

def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam0', frame_idx=idx, timestamp_sec=float(idx))


def _ingest(tmp: Path) -> IngestArtifact:
    return IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=tmp,
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=10.0, num_frames=2, width=16, height=16, has_audio=False)],
    )


def _instances(tmp: Path) -> InstanceArtifact:
    obs = [
        InstanceObservation('h0', 't0', _fr(0), kind='human', bbox_xyxy=np.array([0, 0, 4, 4], float)),
        InstanceObservation('h1', 't0', _fr(1), kind='human', bbox_xyxy=np.array([1, 1, 5, 5], float)),
        InstanceObservation('o0', 't1', _fr(0), kind='object', bbox_xyxy=np.array([5, 5, 9, 9], float)),  # excluded
    ]
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp, observations=obs)
    art.rebuild_indices()
    return art


class _Decoder:
    def get_frame(self, stream, idx):
        return np.zeros((16, 16, 3), dtype=np.uint8)


class _FakePose:
    backend_used = 'fake_kp'

    def predict(self, frame, boxes):
        return [np.ones((17, 3), np.float32) for _ in boxes]


def test_keypoint_rcnn_rung_humans_only(tmp_path: Path) -> None:
    out = KeypointRcnnRung(_Decoder(), _FakePose()).run(_ingest(tmp_path), _instances(tmp_path / 'seg'))
    assert set(out) == {'h0', 'h1'}  # the object instance is excluded
    assert out['h0']['backend'] == 'fake_kp'
    assert out['h0']['keypoints_2d'].shape == (17, 3)


def test_bbox_only_rung(tmp_path: Path) -> None:
    out = BboxOnlyRung(_Decoder()).run(_ingest(tmp_path), _instances(tmp_path / 'seg'))
    assert set(out) == {'h0', 'h1'}
    assert all(v == {'backend': 'bbox_only'} for v in out.values())


def test_frame_loop_carries_sapiens_dense_record(tmp_path: Path) -> None:
    class _FakeSapiens:
        def predict(self, frame, boxes):
            return [{
                'keypoints_2d': np.zeros((5, 3), np.float32), 'seg': np.zeros((4, 4), np.uint8),
                'depth': np.zeros((4, 4), np.float32), 'normals': np.zeros((4, 4, 3), np.float32),
                'backend': 'sapiens_fake',
            } for _ in boxes]

    out = _frame_loop(_Decoder(), _FakeSapiens(), _ingest(tmp_path), _instances(tmp_path / 'seg'))
    assert set(out) == {'h0', 'h1'}
    assert out['h0']['backend'] == 'sapiens_fake'
    assert out['h0']['seg'].shape == (4, 4) and out['h0']['normals'].shape == (4, 4, 3)


# --- Sapiens decode math (pure) ---------------------------------------------

def test_sapiens_decode_pose_maps_heatmap_to_image() -> None:
    # peaks at (row, col); expect x = col*(pw/hw) - pad_left + x1, y = row*(ph/hh) - pad_top + y1
    hm = np.zeros((2, 8, 6), np.float32)
    hm[0, 4, 3] = 5.0
    hm[1, 0, 0] = 2.0
    meta = {'pad_top': 0, 'pad_left': 0, 'ph': 16, 'pw': 12, 'ch': 16, 'cw': 12}
    kp = SapiensEstimator._decode_pose(hm, meta, origin=(100, 200))
    assert kp[0, 0] == pytest.approx(106.0)  # 3*(12/6)+100
    assert kp[0, 1] == pytest.approx(208.0)  # 4*(16/8)+200
    assert kp[0, 2] == pytest.approx(5.0)
    assert kp[1, 0] == pytest.approx(100.0) and kp[1, 1] == pytest.approx(200.0)


def test_sapiens_decode_pose_removes_padding() -> None:
    hm = np.zeros((1, 8, 6), np.float32)
    hm[0, 4, 3] = 1.0
    meta = {'pad_top': 4, 'pad_left': 2, 'ph': 16, 'pw': 12, 'ch': 12, 'cw': 10}
    kp = SapiensEstimator._decode_pose(hm, meta, origin=(0, 0))
    assert kp[0, 0] == pytest.approx(3 * (12 / 6) - 2)  # x_pad - pad_left
    assert kp[0, 1] == pytest.approx(4 * (16 / 8) - 4)  # y_pad - pad_top


def test_sapiens_resize_and_unpad_shape() -> None:
    arr = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    meta = {'pad_top': 2, 'pad_left': 1, 'ph': 8, 'pw': 6, 'ch': 6, 'cw': 5}
    out = SapiensEstimator._resize_and_unpad(arr, meta)
    assert out.shape == (6, 5)  # resized to (ph,pw) then sliced to (ch,cw)

"""Tests for the floor `reid` stage: per-tracklet mean embeddings (FP1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.perception.reid.reid_stage import FloorReid


def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam0', frame_idx=idx, timestamp_sec=float(idx))


class _Decoder:
    def get_frame(self, stream, idx):
        return np.zeros((16, 16, 3), dtype=np.uint8)


class _Embedder:
    backend_used = 'fake'

    def __init__(self):
        self.calls = 0

    def embed(self, frame, dets):
        # distinct constant vector per detection by bbox x-origin, so the two
        # tracks get different means
        self.calls += 1
        out = []
        for d in dets:
            x0 = float(np.asarray(d['bbox_xyxy'])[0])
            out.append([1.0, 0.0] if x0 < 5 else [0.0, 1.0])
        return np.asarray(out, dtype=np.float32)


def _instance(tmp) -> InstanceArtifact:
    obs = [
        InstanceObservation('i0', 'track_0000', _fr(0), kind='object', bbox_xyxy=np.array([0, 0, 4, 4], float)),
        InstanceObservation('i1', 'track_0000', _fr(1), kind='object', bbox_xyxy=np.array([0, 0, 4, 4], float)),
        InstanceObservation('i2', 'track_0001', _fr(0), kind='object', bbox_xyxy=np.array([10, 10, 14, 14], float)),
    ]
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp, observations=obs)
    art.rebuild_indices()
    return art


def test_floor_reid_means_per_track_and_normalizes(tmp_path: Path) -> None:
    ingest = IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=tmp_path,
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=30.0, num_frames=3, width=16, height=16, has_audio=False)],
    )
    embedder = _Embedder()
    out = FloorReid(_Decoder(), embedder).run(ingest, _instance(tmp_path / 'seg'))

    assert set(out) == {'track_0000', 'track_0001'}
    np.testing.assert_allclose(out['track_0000'], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out['track_0001'], [0.0, 1.0], atol=1e-6)
    # frames decoded once each (2 unique frames), not once per instance
    assert embedder.calls == 2


def test_floor_reid_empty_instances(tmp_path: Path) -> None:
    ingest = IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=tmp_path,
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=30.0, num_frames=1, width=16, height=16, has_audio=False)],
    )
    empty = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp_path / 'seg')
    out = FloorReid(_Decoder(), _Embedder()).run(ingest, empty)
    assert out == {}

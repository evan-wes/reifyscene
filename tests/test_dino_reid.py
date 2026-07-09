"""Tests for the DINO `reid` rung: cross-occlusion merge builder (FP3).

The pure ``build_cross_occlusion_merge_map`` is tested exhaustively (no model,
no torch). ``DinoReid.run`` is exercised end-to-end via a fake-embed subclass so
the decode/crop/pool/span/merge loop is covered without downloading DINO
weights; the real DINOv2 inference path is validated separately by the
synthetic-occlusion script (scripts/), kept out of the offline unit suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.perception.reid.dino_reid import DinoReid, build_cross_occlusion_merge_map


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- pure merge builder (mutual-NN + ratio test) ----------------------------

def test_merges_unambiguous_mutual() -> None:
    # each is the other's only candidate -> mutual, unambiguous -> merge
    vectors = {'A': _unit(1, 0), 'B': _unit(1, 0)}
    spans = {'A': ('cam0', 0, 5), 'B': ('cam0', 10, 15)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6) == {'B': 'A'}


def test_no_merge_below_threshold() -> None:
    vectors = {'A': _unit(1, 0), 'B': _unit(0, 1)}  # orthogonal -> cosine 0
    spans = {'A': ('cam0', 0, 5), 'B': ('cam0', 10, 15)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6) == {}


def test_no_merge_when_co_visible() -> None:
    # identical appearance but overlapping spans -> co-visible, never merged
    vectors = {'A': _unit(1, 0), 'B': _unit(1, 0)}
    spans = {'A': ('cam0', 0, 10), 'B': ('cam0', 5, 15)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6) == {}


def test_no_merge_across_streams() -> None:
    # cross-stream identity matching is out of FP3 scope
    vectors = {'A': _unit(1, 0), 'B': _unit(1, 0)}
    spans = {'A': ('cam0', 0, 5), 'B': ('cam1', 10, 15)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6) == {}


def test_canonical_is_earliest_appearance() -> None:
    # earliest frame_min wins, regardless of lexicographic id order
    vectors = {'track_z': _unit(1, 0), 'track_a': _unit(1, 0)}
    spans = {'track_z': ('cam0', 0, 5), 'track_a': ('cam0', 10, 15)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6) == {'track_a': 'track_z'}


def test_mutual_nn_rejects_non_mutual() -> None:
    # A's best is B, but B prefers C over A -> A~B not mutual -> A unmerged.
    # B~C is the mutual pair. (ratio_margin=0 to isolate the mutual-NN rule.)
    vectors = {'A': _unit(1, 0, 0), 'B': _unit(0.65, 0.76, 0), 'C': _unit(0.30, 0.954, 0)}
    spans = {'A': ('cam0', 0, 5), 'B': ('cam0', 10, 15), 'C': ('cam0', 20, 25)}
    merge_map = build_cross_occlusion_merge_map(vectors, spans, threshold=0.6, ratio_margin=0.0)
    assert merge_map == {'C': 'B'}
    assert 'A' not in merge_map  # A wanted B, but B's best was C -> no drive-by merge


def test_ratio_test_rejects_ambiguous() -> None:
    # A is near-equally similar to B and C (neither clearly best) -> abstain.
    vectors = {'A': _unit(1, 0, 0), 'B': _unit(0.7, 0.714, 0), 'C': _unit(0.7, 0, 0.714)}
    spans = {'A': ('cam0', 0, 5), 'B': ('cam0', 10, 15), 'C': ('cam0', 20, 25)}
    assert build_cross_occlusion_merge_map(vectors, spans, threshold=0.6, ratio_margin=0.1) == {}


def test_empty_input() -> None:
    assert build_cross_occlusion_merge_map({}, {}, threshold=0.6) == {}


# --- DinoReid.run integration (fake embedder, no weights) --------------------

def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam0', frame_idx=idx, timestamp_sec=float(idx))


class _PaintDecoder:
    """Left half of every frame is identity-red, right half identity-blue."""

    def get_frame(self, stream, idx):
        f = np.zeros((16, 16, 3), dtype=np.uint8)
        f[:, :8] = (255, 0, 0)
        f[:, 8:] = (0, 0, 255)
        return f


class _FakeDino(DinoReid):
    """DinoReid with the DINO model replaced by per-crop mean RGB."""

    def _load_model(self):  # no transformers / no download
        self._device_resolved = 'cpu'

    def _embed_crops(self, crops):
        return np.asarray([[c[..., k].mean() for k in range(3)] for c in crops], dtype=np.float32)


def _obs(iid: str, tid: str, fidx: int, bbox) -> InstanceObservation:
    return InstanceObservation(iid, tid, _fr(fidx), kind='object', bbox_xyxy=np.array(bbox, float))


def test_dino_run_embeds_and_merges(tmp_path: Path) -> None:
    ingest = IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=tmp_path,
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=30.0, num_frames=12, width=16, height=16, has_audio=False)],
    )
    obs = [
        _obs('a0', 'track_a', 0, [0, 0, 8, 16]),   # red, frames 0-1
        _obs('a1', 'track_a', 1, [0, 0, 8, 16]),
        _obs('b0', 'track_b', 10, [0, 0, 8, 16]),  # red, frames 10-11 (disjoint from a)
        _obs('b1', 'track_b', 11, [0, 0, 8, 16]),
        _obs('c0', 'track_c', 0, [8, 0, 16, 16]),  # blue, frames 0-1 (co-visible with a)
        _obs('c1', 'track_c', 1, [8, 0, 16, 16]),
    ]
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp_path / 'seg', observations=obs)
    art.rebuild_indices()

    embeddings, merge_map = _FakeDino(_PaintDecoder(), {'merge_threshold': 0.6}).run(ingest, art)

    assert set(embeddings) == {'track_a', 'track_b', 'track_c'}
    # red track_a + red track_b are disjoint and identical -> merged onto earliest
    assert merge_map == {'track_b': 'track_a'}
    # blue track_c is a different appearance and co-visible with track_a -> separate
    assert 'track_c' not in merge_map

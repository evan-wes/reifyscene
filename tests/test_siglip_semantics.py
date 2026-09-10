"""Tests for the `semantics` rungs (FP4a).

The pure `score_zero_shot` and the `FloorSemantics` rung are tested without any
model; `SiglipSemantics.run` is exercised end-to-end via a fake-embed subclass so
the decode/crop/pool/canonicalize/score loop is covered without downloading
SigLIP2 weights. The real SigLIP2 path is validated by scripts/fp4_validate_*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from reifyscene.perception.semantics.semantics_artifact import UNKNOWN_CATEGORY
from reifyscene.perception.semantics.siglip_semantics import (
    FloorSemantics,
    SiglipSemantics,
    score_zero_shot,
)


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- pure zero-shot scorer ---------------------------------------------------

def test_score_picks_argmax() -> None:
    text = np.eye(2, dtype=np.float32)  # cat=[1,0], dog=[0,1]
    cat, score, topk = score_zero_shot(_unit(1, 0), text, ['cat', 'dog'])
    assert cat == 'cat'
    assert score == 1.0
    assert topk[0] == ('cat', 1.0)


def test_score_topk_ranked() -> None:
    text = np.eye(2, dtype=np.float32)
    cat, score, topk = score_zero_shot(_unit(0.6, 0.8), text, ['cat', 'dog'], topk=2)
    assert cat == 'dog'
    assert [lbl for lbl, _ in topk] == ['dog', 'cat']  # 0.8 > 0.6
    assert abs(score - 0.8) < 1e-6


def test_score_empty_vocab_is_unknown() -> None:
    cat, score, topk = score_zero_shot(_unit(1, 0), np.empty((0, 2), np.float32), [])
    assert cat == UNKNOWN_CATEGORY and score == 0.0 and topk == []


def test_score_min_score_abstains() -> None:
    text = np.stack([_unit(0.3, 0.954)])  # cosine 0.3 with [1,0]
    cat, score, topk = score_zero_shot(_unit(1, 0), text, ['cat'], min_score=0.5)
    assert cat == UNKNOWN_CATEGORY
    assert abs(score - 0.3) < 1e-3  # the (rejected) best score is still reported
    assert topk[0][0] == 'cat'


# --- FloorSemantics ----------------------------------------------------------

def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam0', frame_idx=idx, timestamp_sec=float(idx))


def _obs(iid: str, tid: str, fidx: int, bbox, concept: str = '') -> InstanceObservation:
    return InstanceObservation(iid, tid, _fr(fidx), concept=concept, kind='object', bbox_xyxy=np.array(bbox, float))


class _Reid:
    def __init__(self, merge):
        self.merge = merge

    def canonical(self, track_id: str) -> str:
        return self.merge.get(track_id, track_id)


def test_floor_carries_majority_concept() -> None:
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=Path('/tmp'), observations=[
        _obs('a0', 'track_0', 0, [0, 0, 4, 4], concept='car'),
        _obs('a1', 'track_0', 1, [0, 0, 4, 4], concept='car'),
        _obs('b0', 'track_1', 0, [5, 5, 9, 9], concept=''),  # class-agnostic -> unknown
    ])
    art.rebuild_indices()
    records, embeddings = FloorSemantics().run(None, art)
    assert records['track_0'].category == 'car'
    assert records['track_0'].score == 1.0
    assert records['track_1'].category == UNKNOWN_CATEGORY
    assert embeddings == {}  # floor has no embeddings


def test_floor_groups_by_canonical_reid() -> None:
    # reid merges track_1 -> track_0; both 'car' observations fold into one record
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=Path('/tmp'), observations=[
        _obs('a0', 'track_0', 0, [0, 0, 4, 4], concept='car'),
        _obs('b0', 'track_1', 5, [0, 0, 4, 4], concept='car'),
    ])
    art.rebuild_indices()
    records, _ = FloorSemantics(reid_artifact=_Reid({'track_1': 'track_0'})).run(None, art)
    assert set(records) == {'track_0'}
    assert records['track_0'].category == 'car'


# --- SiglipSemantics.run integration (fake embedder, no weights) -------------

class _PaintDecoder:
    """Left half of every frame is 'cat'-red, right half 'dog'-blue."""

    def get_frame(self, stream, idx):
        f = np.zeros((16, 16, 3), dtype=np.uint8)
        f[:, :8] = (255, 0, 0)
        f[:, 8:] = (0, 0, 255)
        return f


class _FakeSiglip(SiglipSemantics):
    def _load_model(self):  # no transformers / no download
        self._device_resolved = 'cpu'
        self._model = object()
        self._text_embs = np.eye(len(self.vocab), dtype=np.float32)  # cat=[1,0], dog=[0,1]

    def _embed_image_crops(self, crops):
        # red crop -> cat axis, blue crop -> dog axis
        out = []
        for c in crops:
            out.append([1.0, 0.0] if float(c[..., 0].mean()) >= float(c[..., 2].mean()) else [0.0, 1.0])
        return np.asarray(out, dtype=np.float32)


def test_siglip_run_labels_tracks(tmp_path: Path) -> None:
    ingest = IngestArtifact(
        run_id='r', stage_name='ingest', save_dir=tmp_path,
        streams=[VideoStreamRef(stream_id='cam0', video_path=Path('v.mp4'), fps=30.0, num_frames=2, width=16, height=16, has_audio=False)],
    )
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp_path / 'seg', observations=[
        _obs('a0', 'track_a', 0, [0, 0, 8, 16]),   # red -> cat
        _obs('a1', 'track_a', 1, [0, 0, 8, 16]),
        _obs('b0', 'track_b', 0, [8, 0, 16, 16]),  # blue -> dog
        _obs('b1', 'track_b', 1, [8, 0, 16, 16]),
    ])
    art.rebuild_indices()

    records, embeddings = _FakeSiglip(_PaintDecoder(), cfg={'vocab': ['cat', 'dog']}).run(ingest, art)

    assert records['track_a'].category == 'cat'
    assert records['track_b'].category == 'dog'
    assert set(embeddings) == {'track_a', 'track_b'}
    assert records['track_a'].attributes['source'] == 'siglip2'

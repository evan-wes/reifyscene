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
    res = score_zero_shot(_unit(1, 0), text, ['cat', 'dog'])
    assert res.category == 'cat'
    assert res.score == 1.0
    assert res.topk[0] == ('cat', 1.0)
    assert not res.abstained and res.abstain_reason is None


def test_score_topk_ranked() -> None:
    text = np.eye(2, dtype=np.float32)
    res = score_zero_shot(_unit(0.6, 0.8), text, ['cat', 'dog'], topk=2)
    assert res.category == 'dog'
    assert [lbl for lbl, _ in res.topk] == ['dog', 'cat']  # 0.8 > 0.6
    assert abs(res.score - 0.8) < 1e-6


def test_score_empty_vocab_is_unknown() -> None:
    res = score_zero_shot(_unit(1, 0), np.empty((0, 2), np.float32), [])
    assert res.category == UNKNOWN_CATEGORY and res.score == 0.0 and res.topk == []
    assert res.abstained and res.abstain_reason == 'no_labels'
    assert res.margin is None  # nothing to be uncertain between


def test_score_min_score_abstains() -> None:
    text = np.stack([_unit(0.3, 0.954)])  # cosine 0.3 with [1,0]
    res = score_zero_shot(_unit(1, 0), text, ['cat'], min_score=0.5)
    assert res.category == UNKNOWN_CATEGORY
    assert abs(res.score - 0.3) < 1e-3  # the (rejected) best score is still reported
    assert res.topk[0][0] == 'cat'
    assert res.abstain_reason == 'low_score'


# --- separation: recorded always, gated only on request ----------------------

def _spread(top1: float, rest: list[float]) -> tuple[np.ndarray, list[str]]:
    """Text embeddings whose cosines with [1,0] are exactly the given scores."""
    scores = [top1, *rest]
    rows = [_unit(c, float(np.sqrt(max(1 - c ** 2, 0.0)))) for c in scores]
    return np.stack(rows), [f"l{i}" for i in range(len(scores))]


def test_abstention_is_off_by_default() -> None:
    """A near-tie still gets named unless a caller asks for the gate.

    A raw gap shrinks as the candidate list grows, so no constant is a safe default.
    The evidence is always recorded; acting on it is opt-in.
    """
    text, labels = _spread(0.101, [0.100, 0.099, 0.098, 0.097])
    res = score_zero_shot(_unit(1, 0), text, labels)
    assert res.category == 'l0'
    assert not res.abstained
    assert res.margin is not None and res.margin_z is not None


def _vocab(top1: float, bulk_hi: float, bulk_lo: float, n: int = 79):
    """A realistic candidate spread: one winner over a bulk that fans out."""
    return _spread(top1, list(np.linspace(bulk_hi, bulk_lo, n)))


def test_margin_z_separates_a_real_hit_from_a_forced_one() -> None:
    """Shaped like the measured lab-coat distributions: winner clear vs winner crowded."""
    hit = score_zero_shot(_unit(1, 0), *_vocab(0.1151, 0.0821, 0.045))
    forced = score_zero_shot(_unit(1, 0), *_vocab(0.1170, 0.1065, 0.050))
    assert hit.margin_z > forced.margin_z
    # and absolute score points the wrong way, which is the whole point
    assert forced.score > hit.score


def test_margin_z_gate_fires_when_enabled() -> None:
    forced = score_zero_shot(_unit(1, 0), *_vocab(0.1170, 0.1065, 0.050), min_margin_z=3.5)
    assert forced.category == UNKNOWN_CATEGORY
    assert forced.abstained and forced.abstain_reason == 'low_separation'
    assert forced.topk[0][0] == 'l0'  # the rejected winner is still reported


def test_margin_z_gate_keeps_a_clear_win() -> None:
    hit = score_zero_shot(_unit(1, 0), *_vocab(0.1151, 0.0821, 0.045), min_margin_z=3.5)
    assert hit.category == 'l0' and not hit.abstained


def test_margin_z_is_inflated_by_a_degenerate_bulk() -> None:
    """A known limitation, pinned so it is not rediscovered as a surprise.

    The z-score measures the winner against the spread of the candidates. When the
    bulk is nearly identical that spread collapses, and a negligible gap reads as many
    sigma. Real vocabularies fan out, which is why the statistic works on them -- but a
    tiny or highly redundant vocabulary breaks it, and the threshold is not transferable
    between vocabularies of different shape.
    """
    tight = score_zero_shot(_unit(1, 0), *_spread(0.101, [0.100] * 20))
    assert tight.margin < 0.002        # a negligible raw gap
    assert tight.margin_z > 3.0        # nonetheless reads as strongly separated


def test_margin_z_is_none_when_there_is_no_distribution() -> None:
    """With two candidates the z-score is 1.0 whatever the scores -- not evidence."""
    text, labels = _spread(0.9, [0.1])
    res = score_zero_shot(_unit(1, 0), text, labels, min_margin_z=3.0)
    assert res.margin_z is None
    assert res.category == 'l0' and not res.abstained  # the gate cannot fire on nothing
    assert res.margin is not None                      # the raw gap still is recorded


def test_labcoat_separation_measurements() -> None:
    """Regression on the real numbers, as measurements rather than as a threshold.

    Measured top1/top2 cosines from davis_lab-coat_full. The raw gap separates the two
    correctly-named people from every off-vocabulary label by roughly 3x, and the
    highest-scoring track in the run is one of the wrong ones -- which is why absolute
    score is not the instrument. No default acts on this; it is the evidence a
    threshold would be chosen from.
    """
    measured = {  # track -> (top1, top2)
        'person_a': (0.1151, 0.0821),
        'person_b': (0.1080, 0.0631),
        'junk_handbag': (0.1170, 0.1065),  # highest absolute score in the run, and wrong
        'junk_bird': (0.1077, 0.1055),
        'junk_cup': (0.0850, 0.0848),
    }
    margins = {}
    for name, (top1, top2) in measured.items():
        text, labels = _spread(top1, [top2])
        res = score_zero_shot(_unit(1, 0), text, labels)
        assert not res.abstained  # nothing is refused by default
        margins[name] = res.margin

    people = [margins['person_a'], margins['person_b']]
    junk = [v for k, v in margins.items() if k.startswith('junk')]
    assert min(people) > 3 * max(junk)
    assert measured['junk_handbag'][0] > measured['person_a'][0]  # score misleads


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
    assert records['track_1'].abstained
    assert records['track_1'].abstain_reason == 'no_concept'
    assert not records['track_0'].abstained  # a real concept is not a refusal
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

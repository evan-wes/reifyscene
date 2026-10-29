"""The SAM 3 `segment` rungs (sam3_segment.py): vocabulary, kinds, observations and the fallback.

No model is loaded: the tagger, SAM 3's video tracking, the first-frame seeds and SAM 2's
propagation are replaced by fakes at the seams the rungs expose. The real models were measured on
DAVIS and YouTube-VOS before the rungs were built (2026-10-29) and run in full pipeline runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.perception.instance import sam3_segment as s3
from reifyscene.perception.instance.sam2_segment import SAM2InstanceSegmenter
from reifyscene.perception.semantics.vocab import COCO_THINGS


# -- phrases ----------------------------------------------------------------------------------

@pytest.mark.parametrize("phrase, noun", [
    ("man in a red shirt", "man"), ("small brown dog", "dog"), ("horse with a rider", "horse"),
    ("Surfers", "surfers"), ("", ""), ("crowd of people", "crowd"),
])
def test_head_noun(phrase: str, noun: str) -> None:
    assert s3.head_noun(phrase) == noun


@pytest.mark.parametrize("phrase, kind", [
    ("person", "human"), ("woman in black", "human"), ("dancers", "human"), ("bmx rider", "human"),
    ("dog", "object"), ("glass", "object"), ("red car", "object"), ("horse with a rider", "object"),
    ("crowd", "object"), ("bboy", None), ("", None),
])
def test_phrase_kind(phrase: str, kind: str | None) -> None:
    assert s3.phrase_kind(phrase) == kind


def test_parse_tag_lines_strips_numbering_and_dedupes() -> None:
    text = "1. Dog\n- dog\n* red ball.\n\n2) " + "x" * 50 + "\nperson"
    assert s3.parse_tag_lines(text) == ["dog", "red ball", "person"]
    assert s3.parse_tag_lines(text, max_phrases=2) == ["dog", "red ball"]


def test_keyframe_positions_spread_and_include_the_ends() -> None:
    assert s3.keyframe_positions(90, 4) == [0, 30, 59, 89]
    assert s3.keyframe_positions(2, 4) == [0, 1]
    assert s3.keyframe_positions(0, 4) == []


# -- vocabulary -------------------------------------------------------------------------------

def _frames(n: int, stream: str = "cam_00") -> list[FrameRef]:
    return [FrameRef(stream_id=stream, frame_idx=10 + i, timestamp_sec=float(i)) for i in range(n)]


class _Tagger:
    """Lists 'thing<frame value>' for each frame; records what it was shown and whether it was released."""

    instances: list["_Tagger"] = []

    def __init__(self, cfg):
        self.seen: list[int] = []
        self.released = False
        _Tagger.instances.append(self)

    def tag(self, image):
        self.seen.append(int(image[0, 0, 0]))
        return [f"thing{int(image[0, 0, 0])}", "dog"]

    def release(self):
        self.released = True


def _decode(fr: FrameRef) -> np.ndarray:
    return np.full((4, 4, 3), fr.frame_idx, np.uint8)


def test_vocabulary_tags_keyframes_once_and_releases_the_tagger() -> None:
    _Tagger.instances = []
    vocab = s3.Sam3Vocabulary({"sam3_tag_keyframes": 3}, tagger_factory=_Tagger)
    streams = vocab.prepare({"cam_00": _frames(5)}, _decode)
    v = streams["cam_00"]
    assert sorted(v.keyframe_phrases) == [10, 12, 14]
    assert v.phrases[:4] == ["thing10", "dog", "thing12", "thing14"]
    assert v.phrases[4:] == [c for c in COCO_THINGS if c != "dog"]
    assert v.first_frame_phrases()[:2] == ["thing10", "dog"] and "thing14" not in v.first_frame_phrases()
    assert v.to_dict()["source"] == "qwen3-vl+coco80"
    assert _Tagger.instances[0].released
    vocab.prepare({"cam_00": _frames(5)}, _decode)          # already tagged: no second tagger
    assert len(_Tagger.instances) == 1


def test_vocabulary_without_a_tagger_is_coco_and_says_why() -> None:
    def broken(cfg):
        raise OSError("Qwen3-VL is not in the local cache")

    v = s3.Sam3Vocabulary({}, tagger_factory=broken).prepare({"cam_00": _frames(3)}, _decode)["cam_00"]
    assert v.phrases == list(COCO_THINGS)
    assert "not in the local cache" in v.error and v.to_dict()["source"] == "coco80"


# -- the rungs, with fake models ---------------------------------------------------------------

def _ingest(tmp_path: Path) -> IngestArtifact:
    return IngestArtifact(
        run_id="r", stage_name="ingest", save_dir=tmp_path,
        streams=[VideoStreamRef(stream_id="cam_00", video_path=Path("v.mp4"), fps=30.0, num_frames=3,
                                width=8, height=8, has_audio=False)],
    )


class _Decoder:
    def get_frame(self, stream, idx):
        return np.full((8, 8, 3), idx, np.uint8)


class _PersonDetector:
    """A person box over the left half of every frame."""

    backend_used = "fake"

    def detect(self, frame):
        return [{"category": "person", "bbox_xyxy": np.array([0, 0, 4, 8], np.float32), "score": 0.9}]


def _mask(x0: int, x1: int) -> np.ndarray:
    m = np.zeros((8, 8), bool)
    m[:, x0:x1] = True
    return m


def _vocab() -> s3.Sam3Vocabulary:
    return s3.Sam3Vocabulary({}, tagger_factory=_Tagger)


def test_video_rung_names_scores_and_kinds_its_tracks(tmp_path: Path, monkeypatch) -> None:
    seen_phrases = []

    def track(self, images, phrases):
        seen_phrases.append(list(phrases))
        for pos in range(len(images)):
            rows = [(3, "dog", 0.8 + 0.05 * pos, _mask(5, 8)), (7, "bboy", 0.6, _mask(0, 4))]
            if pos == 2:   # arrives on the last frame
                rows.append((9, "", 0.55, _mask(4, 5)))
            yield pos, rows

    monkeypatch.setattr(s3.SAM3VideoSegmenter, "_track", track)
    monkeypatch.setattr(s3.SAM3VideoSegmenter, "release", lambda self: None)
    rung = s3.SAM3VideoSegmenter(_Decoder(), _PersonDetector(), _vocab(), {})
    obs, masks = rung.run(_ingest(tmp_path), _frames(3))

    assert "thing10" in seen_phrases[0] and "person" in seen_phrases[0]
    by_track = {}
    for o in obs:
        by_track.setdefault(o.track_id, []).append(o)
    assert set(by_track) == {"sam3:cam_00:3", "sam3:cam_00:7", "sam3:cam_00:9"}
    dog = by_track["sam3:cam_00:3"]
    assert [o.concept for o in dog] == ["dog"] * 3 and [o.kind for o in dog] == ["object"] * 3
    assert [round(o.confidence, 2) for o in dog] == [0.8, 0.85, 0.9]
    assert dog[0].attributes == {"segmenter_backend": "sam3_video", "obj_id": 3, "concept_source": "sam3"}
    # "bboy" names neither a person nor a known thing: the person detector's box decides
    assert {o.kind for o in by_track["sam3:cam_00:7"]} == {"human"}
    late = by_track["sam3:cam_00:9"]
    assert [o.frame_ref.frame_idx for o in late] == [12] and late[0].kind == "object"
    assert all(masks[o.instance_id].any() for o in obs)
    assert np.array_equal(dog[0].bbox_xyxy, np.array([5, 0, 8, 8], np.float32))


def test_video_rung_out_of_memory_releases_then_raises(tmp_path: Path, monkeypatch) -> None:
    import torch

    released = []

    def track(self, images, phrases):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.51 GiB")
        yield  # pragma: no cover

    monkeypatch.setattr(s3.SAM3VideoSegmenter, "_track", track)
    monkeypatch.setattr(s3.SAM3VideoSegmenter, "release", lambda self: released.append(True))
    vocab = _vocab()
    rung = s3.SAM3VideoSegmenter(_Decoder(), _PersonDetector(), vocab, {})
    with pytest.raises(RuntimeError, match="ran out of GPU memory .*1.51 GiB"):
        rung.run(_ingest(tmp_path), _frames(3))
    assert released == [True]
    assert "cam_00" in vocab.streams          # the fallback reuses these tags


def test_seed_rung_tracks_first_frame_instances_with_sam2(tmp_path: Path, monkeypatch) -> None:
    asked = {}

    def seed(self, first_frames):
        asked.update({sid: self.vocabulary.streams[sid].first_frame_phrases() for sid in first_frames})
        return {sid: [SimpleNamespace(phrase="person", score=0.9, mask=_mask(0, 4)),
                      SimpleNamespace(phrase="kite", score=0.7, mask=_mask(5, 7))] for sid in first_frames}

    def track_seeds(self, images, seed_masks):
        assert len(seed_masks) == 2
        return {pos: {0: seed_masks[0], 1: seed_masks[1] if pos < 2 else np.zeros((8, 8), bool)}
                for pos in range(len(images))}

    monkeypatch.setattr(s3.SAM3SeedsSAM2Segmenter, "_seed", seed)
    monkeypatch.setattr(SAM2InstanceSegmenter, "track_seeds", track_seeds)
    rung = s3.SAM3SeedsSAM2Segmenter(_Decoder(), _PersonDetector(), _vocab(), {})
    obs, masks = rung.run(_ingest(tmp_path), _frames(3))

    assert "thing10" in asked["cam_00"] and "thing12" not in asked["cam_00"]   # first-frame tags only
    person = [o for o in obs if o.track_id == "sam3seed:cam_00:0"]
    kite = [o for o in obs if o.track_id == "sam3seed:cam_00:1"]
    assert len(person) == 3 and {(o.concept, o.kind, o.confidence) for o in person} == {("person", "human", 0.9)}
    assert len(kite) == 2 and kite[0].attributes["segmenter_backend"] == "sam3_seeds_sam2"
    assert set(masks) == {o.instance_id for o in obs}


# -- the probe --------------------------------------------------------------------------------

def test_probe_rejects_unpatched_transformers(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    source = tmp_path / "modeling_sam3_video.py"
    source.write_text("def nms_masks(): ...\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(source)))
    assert s3.sam3_video_patched().startswith("not-patched")
    source.write_text("def greedy_nms_from_ious(): ...\ncross_prompt_nms_thresh = 0.0\n")
    assert s3.sam3_video_patched() is None
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert s3.sam3_video_patched().startswith("not-installed")


# -- the ladder ---------------------------------------------------------------------------------

def test_segment_falls_back_to_first_frame_seeds_when_video_mode_runs_out_of_memory(
    tmp_path: Path, monkeypatch
) -> None:
    """The stage records why video mode did not run, which rung did, and the vocabulary searched."""
    import reifyscene.api.pipeline as pipeline_module
    from reifyscene.api.pipeline import ReifyScenePipeline
    from reifyscene.perception.instance.instance_artifact import InstanceObservation
    from tests.conftest import write_test_video

    def out_of_memory(self, ingest_artifact, frames):
        self._prepare_vocabulary(ingest_artifact, s3._frames_by_stream(frames))
        raise RuntimeError("SAM 3 video mode ran out of GPU memory (CUDA out of memory)")

    def seeds(self, ingest_artifact, frames):
        fr = frames[0]
        return [InstanceObservation("inst:a", f"sam3seed:{fr.stream_id}:0", fr, concept="kite", kind="object",
                                    bbox_xyxy=np.zeros(4, np.float32), confidence=0.7)], {"inst:a": _mask(0, 2)}

    monkeypatch.setattr(pipeline_module, "sam3_video_patched", lambda: None)
    monkeypatch.setattr(pipeline_module, "importable", lambda *mods: (lambda: None))
    monkeypatch.setattr(pipeline_module, "Sam3Vocabulary", lambda cfg: s3.Sam3Vocabulary(cfg, tagger_factory=_Tagger))
    monkeypatch.setattr(s3.SAM3VideoSegmenter, "run", out_of_memory)
    monkeypatch.setattr(s3.SAM3SeedsSAM2Segmenter, "run", seeds)

    video = tmp_path / "clip.mp4"
    write_test_video(video, num_frames=6)
    pipeline = ReifyScenePipeline({"run_dir": str(tmp_path / "run"), "geometry": {"backend_mode": "heuristic"}})
    artifact = pipeline.segment(pipeline.preprocess([str(video)]))

    cap = artifact.capability
    assert cap.backend_used == "sam3_seeds_sam2" and cap.fidelity == "open_vocab_first_frame"
    failed = cap.rungs[0]
    assert (failed.name, failed.status) == ("sam3_video", "failed") and "out of GPU memory" in failed.reason
    assert [o.concept for o in artifact.observations] == ["kite"]
    vocab = artifact.metadata["sam3_vocabulary"]
    (stream_vocab,) = vocab.values()
    assert stream_vocab["source"] == "qwen3-vl+coco80" and stream_vocab["keyframes"]

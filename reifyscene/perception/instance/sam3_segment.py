"""SAM 3 instance segmentation: every object a vision-language model lists, found and tracked by SAM 3.

The top rungs of the `segment` ladder (design §4A), above SAM 2's prompt-free discovery. They were
chosen by measurement (2026-10-29, DAVIS 2017 and the YouTube-VOS 2019 validation split):

- **Vocabulary.** Qwen3-VL-8B lists the kinds of object visible on keyframes spread through each
  stream (4 by default), and COCO-80's classes are added. On the first frames of DAVIS's 90 clips,
  SAM 3 with Qwen's phrases for that frame plus COCO-80 found 94% of the annotated objects, against
  42% for the 20 largest of SAM 2's automatic masks. Tagging the first frame alone misses what
  arrives later: on YouTube-VOS, SAM 3 found 67% of late arrivals on the frame they appear with
  keyframe tags, and 49% with first-frame tags.
- **`sam3_video`**: SAM 3's video mode detects every phrase on every frame and keeps identities, so
  an object that arrives late becomes a track when it appears. It found 95.5% of the annotated
  object-frames on eight DAVIS clips. Its memory grows with the number of tracks, and on the crowd
  clips india and lindy-hop it ran out on a 32 GB card.
- **`sam3_seeds_sam2`**: the fallback when it does. SAM 3 runs on the first frame and SAM 2's video
  predictor tracks each instance it found: 91.5% on the same eight clips, and it completes the
  crowds. Nothing that arrives after the first frame is found.

Both rungs name their tracks. The phrase SAM 3 matched becomes the observation's ``concept``, which
the semantics stage uses as the track's label, and SAM 3's score becomes its ``confidence``. A
phrase also decides ``kind`` when it is a person word or a known thing; for any other phrase the
person detector votes, as on the SAM 2 rung.

SAM 3's video mode needs reifyscene's transformers patches (``third_party_patches/transformers/``,
installed by ``setup_reifyscene.sh --with-sam3``). Unpatched, it keeps every duplicate detection
and runs out of GPU memory, so the rung's probe skips it when they are missing.
"""

from __future__ import annotations

import gc
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceObservation
from reifyscene.perception.instance.sam2_segment import (
    SAM2InstanceSegmenter,
    _mask_bbox_xyxy,
    vote_human_kind,
)
from reifyscene.perception.semantics.vocab import BACKGROUND_VOCAB, COCO_THINGS, THING_VOCAB

DEFAULT_SAM3_MODEL_ID = "facebook/sam3"
DEFAULT_TAGGER_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TAG_KEYFRAMES = 4
# Every frame the tagger was measured on was at most 1280 pixels wide; larger frames are shrunk to that.
DEFAULT_TAG_MAX_SIDE = 1280
DEFAULT_MAX_PHRASES_PER_KEYFRAME = 40
DEFAULT_SEED_SCORE_THRESHOLD = 0.5
# Two detections from different phrases overlapping by more than this are one object. 0.5 keeps a
# part inside a whole: a shirt on a person overlaps the person far less.
DEFAULT_CROSS_PROMPT_NMS_IOU = 0.5

TAG_PROMPT = (
    "List every distinct kind of object, animal or person visible in this image, as short "
    "lowercase English noun phrases, one per line, most prominent first. Do not list surfaces "
    "or regions such as floor, wall, sky, ground, water or grass. No numbering, no other text."
)

# Head nouns that make a phrase a person. Plurals are matched by dropping a trailing "s".
PERSON_NOUNS = frozenset({
    "person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
    "baby", "toddler", "infant", "adult", "teenager", "teen", "lady", "gentleman", "guy",
    "player", "dancer", "rider", "cyclist", "biker", "bicyclist", "motorcyclist", "skier",
    "snowboarder", "surfer", "skateboarder", "skater", "runner", "jogger", "swimmer", "diver",
    "athlete", "pedestrian", "spectator", "worker", "student", "performer", "musician", "singer",
    "climber", "hiker", "gymnast", "boxer", "wrestler", "fighter", "judoka", "referee", "driver",
    "passenger", "officer", "soldier", "chef", "doctor", "nurse", "farmer", "fisherman", "parent",
    "mother", "father", "mom", "dad", "grandmother", "grandfather", "bride", "groom", "actor",
    "actress", "tourist",
})
# Parts of a body, which are never an entity of their own. Qwen lists them as objects ("arm",
# "hand", "knee"), and SAM 3 then finds one on every person: on DAVIS lindy-hop, before they were
# dropped, 99 of the 122 object tracks SAM 3 seeded lay mostly inside a person.
BODY_PART_NOUNS = frozenset({
    "arm", "leg", "hand", "foot", "feet", "head", "face", "hair", "eye", "ear", "nose", "mouth",
    "lip", "tooth", "teeth", "tongue", "neck", "shoulder", "elbow", "wrist", "finger", "fingertip",
    "fingernail", "thumb", "knee", "ankle", "toe", "heel", "back", "waist", "hip", "chest", "torso",
    "belly", "stomach", "skin", "beard", "mustache", "moustache", "eyebrow", "forehead", "cheek",
    "chin", "jaw", "thigh", "fist", "body", "limb",
    "paw", "tail", "wing", "fin", "beak", "hoof", "mane", "snout", "whisker",
})
# Everything the semantics vocabularies call a thing or a backdrop, with LVIS's parenthetical
# glosses dropped ("glass (drink container)" -> "glass"); COCO's "person" is a person, not a thing.
_KNOWN_THINGS = frozenset(
    re.sub(r"\s*\(.*?\)", "", label).strip().lower() for label in THING_VOCAB + BACKGROUND_VOCAB
) - PERSON_NOUNS
# What follows the noun a phrase is about: "man in a red shirt", "horse with a rider".
_QUALIFIER = re.compile(r"\s+(?:in|with|on|wearing|holding|of|at|near|riding|carrying)\s.*$")


def head_noun(phrase: str) -> str:
    """The noun a short phrase is about: ``'man in a red shirt' -> 'man'``, ``'small brown dog' -> 'dog'``."""
    core = _QUALIFIER.sub("", phrase.strip().lower())
    words = re.findall(r"[a-z][a-z'-]*", core)
    return words[-1] if words else ""


def phrase_kind(phrase: str) -> str | None:
    """``'human'`` for a person word, ``'object'`` for a known thing, ``None`` when the phrase says neither."""
    text = phrase.strip().lower()
    noun = head_noun(text)
    singular = noun[:-1] if noun.endswith("s") else noun
    if noun in PERSON_NOUNS or singular in PERSON_NOUNS:
        return "human"
    if text in _KNOWN_THINGS or noun in _KNOWN_THINGS or singular in _KNOWN_THINGS:
        return "object"
    return None


# Words that can stand before a body part without making it a thing: "left hand", "blond hair".
_BODY_MODIFIERS = frozenset({"left", "right", "upper", "lower", "raised", "bare", "human", "blond", "blonde"})


def is_body_part(phrase: str) -> bool:
    """Whether a phrase names a part of a body (``'knees'``, ``"person's leg"``, ``'left hand'``).

    Only the part on its own, or after a possessive or a side word. A compound with another noun is a
    thing: among the 49 distinct phrases about a part that Qwen listed for the 130 clips it was
    measured on were "railway crossing gate arm", "shower head" and "paragliding wing".
    """
    words = re.findall(r"[a-z][a-z'-]*", _QUALIFIER.sub("", phrase.strip().lower()))
    if not words:
        return False
    noun = words[-1]
    if noun not in BODY_PART_NOUNS and not (noun.endswith("s") and noun[:-1] in BODY_PART_NOUNS):
        return False
    return all(w in _BODY_MODIFIERS or w.endswith("'s") for w in words[:-1])


def parse_tag_lines(text: str, max_phrases: int = DEFAULT_MAX_PHRASES_PER_KEYFRAME) -> list[str]:
    """The tagger's reply as a phrase list: one per line, bullets and numbering stripped, lowercased,
    duplicates and implausibly long lines dropped."""
    phrases = []
    for line in text.splitlines():
        phrase = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip(".").lower()
        if phrase and len(phrase) < 40:
            phrases.append(phrase)
    return list(dict.fromkeys(phrases))[: max(0, int(max_phrases))]


def keyframe_positions(num_frames: int, count: int) -> list[int]:
    """``count`` positions spread evenly over ``num_frames`` frames, first and last included."""
    if num_frames <= 0 or count <= 0:
        return []
    return sorted(set(np.round(np.linspace(0, num_frames - 1, count)).astype(int).tolist()))


def sam3_video_patched() -> str | None:
    """Probe: ``None`` when transformers carries reifyscene's SAM 3 video patches, else why not.

    Reads the module's source rather than importing it, so the probe stays cheap.
    """
    try:
        spec = importlib.util.find_spec("transformers.models.sam3_video.modeling_sam3_video")
    except (ImportError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        return "not-installed: transformers with SAM 3 video"
    source = Path(spec.origin).read_text()
    if "def greedy_nms_from_ious" not in source or "cross_prompt_nms_thresh" not in source:
        return "not-patched: transformers lacks third_party_patches/transformers (setup_reifyscene.sh --with-sam3)"
    return None


def _hf_cache_dir(cfg: dict[str, Any]) -> str:
    """The Hugging Face hub cache to load SAM 3 and the tagger from: the configured one, else the default."""
    configured = cfg.get("sam3_hf_cache_dir")
    if configured:
        return str(Path(str(configured)).expanduser())
    from huggingface_hub import constants

    return str(constants.HF_HUB_CACHE)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# -- vocabulary ---------------------------------------------------------------------------------

class PhraseTagger:
    """Qwen3-VL lists the kinds of object on a frame, as short noun phrases.

    Loaded from the local cache only: its weights are about 17 GB, which ``setup_reifyscene.sh
    --with-sam3`` downloads, and a run should never start that download by itself. Without them the
    tagger fails and the vocabulary is COCO-80 alone (see :class:`Sam3Vocabulary`).
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.model_id = str(cfg.get("sam3_tagger_model_id", DEFAULT_TAGGER_MODEL_ID))
        self.max_side = int(cfg.get("sam3_tag_max_side", DEFAULT_TAG_MAX_SIDE))
        self.max_phrases = int(cfg.get("sam3_tag_max_phrases", DEFAULT_MAX_PHRASES_PER_KEYFRAME))
        self.cache_dir = _hf_cache_dir(cfg)
        self.device = cfg.get("sam3_device")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        common = {"cache_dir": self.cache_dir, "local_files_only": True}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **common)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, **common
        ).to(self.device).eval()

    def tag(self, image: np.ndarray) -> list[str]:
        import torch

        self._load()
        picture = Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8))
        if max(picture.size) > self.max_side:
            picture.thumbnail((self.max_side, self.max_side), Image.Resampling.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": picture},
                                                 {"type": "text", "text": TAG_PROMPT}]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        reply = self._processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return parse_tag_lines(reply, self.max_phrases)

    def release(self) -> None:
        self._model = None
        self._processor = None
        _release_gpu()


@dataclass
class StreamVocabulary:
    """The phrases one stream is searched for: what the tagger listed per keyframe, plus COCO-80."""

    keyframe_phrases: dict[int, list[str]] = field(default_factory=dict)  # frame idx -> phrases
    base: list[str] = field(default_factory=lambda: list(COCO_THINGS))
    first_frame_idx: int | None = None
    error: str | None = None
    dropped: list[str] = field(default_factory=list)  # tagged, but parts of a body

    @property
    def phrases(self) -> list[str]:
        listed = [p for idx in sorted(self.keyframe_phrases) for p in self.keyframe_phrases[idx]]
        return list(dict.fromkeys(listed + self.base))

    def first_frame_phrases(self) -> list[str]:
        """What a run that only looked at the first frame would search for: its tags plus COCO-80."""
        first = self.keyframe_phrases.get(self.first_frame_idx, []) if self.first_frame_idx is not None else []
        return list(dict.fromkeys(list(first) + self.base))

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframes": {str(idx): list(p) for idx, p in sorted(self.keyframe_phrases.items())},
            "num_phrases": len(self.phrases),
            "source": "coco80" if self.error is not None or not self.keyframe_phrases else "qwen3-vl+coco80",
            "error": self.error,
            "dropped_body_parts": list(self.dropped),
        }


class Sam3Vocabulary:
    """Per-stream phrase lists, tagged once and shared by both SAM 3 rungs. Parts of a body the
    tagger lists are dropped (:func:`is_body_part`) and recorded as dropped.

    When video mode runs out of memory, the fallback rung reuses these tags rather than loading the
    tagger again. If the tagger cannot run at all, the vocabulary is COCO-80 alone and the error is
    recorded: SAM 3 with COCO-80 still found 82% of DAVIS's first-frame objects.
    """

    def __init__(self, cfg: dict[str, Any] | None = None,
                 tagger_factory: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.num_keyframes = int(self.cfg.get("sam3_tag_keyframes", DEFAULT_TAG_KEYFRAMES))
        self._tagger_factory = tagger_factory or PhraseTagger
        self.streams: dict[str, StreamVocabulary] = {}

    def prepare(self, frames_by_stream: dict[str, list[FrameRef]],
                decode: Callable[[FrameRef], np.ndarray]) -> dict[str, StreamVocabulary]:
        """Tag every stream not tagged yet, then release the tagger so SAM 3 has the GPU to itself."""
        todo = {sid: frames for sid, frames in frames_by_stream.items() if sid not in self.streams}
        if not todo:
            return self.streams
        tagger = None
        error = None
        try:
            tagger = self._tagger_factory(self.cfg)
            for stream_id, frames in todo.items():
                vocab = StreamVocabulary(first_frame_idx=int(frames[0].frame_idx) if frames else None)
                for pos in keyframe_positions(len(frames), self.num_keyframes):
                    tags = tagger.tag(decode(frames[pos]))
                    vocab.keyframe_phrases[int(frames[pos].frame_idx)] = [p for p in tags if not is_body_part(p)]
                    vocab.dropped.extend(p for p in tags if is_body_part(p) and p not in vocab.dropped)
                self.streams[stream_id] = vocab
        except Exception as exc:  # no tagger: search for COCO-80 alone, and say so
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if tagger is not None and hasattr(tagger, "release"):
                tagger.release()
        for stream_id, frames in todo.items():
            if error is not None and stream_id not in self.streams:
                self.streams[stream_id] = StreamVocabulary(
                    first_frame_idx=int(frames[0].frame_idx) if frames else None, error=error
                )
        return self.streams

    def to_dict(self) -> dict[str, Any]:
        return {stream_id: vocab.to_dict() for stream_id, vocab in sorted(self.streams.items())}


# -- shared ---------------------------------------------------------------------------------------

@dataclass
class _Track:
    phrase: str
    per_frame: dict[int, tuple[float, np.ndarray]] = field(default_factory=dict)  # pos -> (score, mask)


def _frames_by_stream(selected_frames: list[FrameRef]) -> dict[str, list[FrameRef]]:
    by_stream: dict[str, list[FrameRef]] = {}
    for fr in selected_frames:
        by_stream.setdefault(fr.stream_id, []).append(fr)
    return {sid: sorted(frames, key=lambda fr: int(fr.frame_idx)) for sid, frames in by_stream.items()}


class _Sam3Rung:
    """What both SAM 3 rungs share: decoding, the vocabulary, kinds, and building observations."""

    backend_name = ""
    track_prefix = ""

    def __init__(self, decoder, detector, vocabulary: Sam3Vocabulary, cfg: dict[str, Any] | None = None) -> None:
        self.decoder = decoder
        self.detector = detector
        self.vocabulary = vocabulary
        self.cfg = dict(cfg or {})
        self.model_id = str(self.cfg.get("sam3_model_id", DEFAULT_SAM3_MODEL_ID))
        self.cache_dir = _hf_cache_dir(self.cfg)
        self.local_files_only = bool(self.cfg.get("sam3_local_files_only", False))
        self.device = self.cfg.get("sam3_device")
        self.person_iou = float(self.cfg.get("sam2_person_iou", 0.5))
        self._streams: dict[str, Any] = {}

    def _decode(self, frame_ref: FrameRef) -> np.ndarray:
        stream = self._streams[frame_ref.stream_id]
        return np.asarray(self.decoder.get_frame(stream, frame_ref.frame_idx))[..., :3].astype(np.uint8)

    def _prepare_vocabulary(self, ingest_artifact, by_stream: dict[str, list[FrameRef]]) -> None:
        self._streams = {s.stream_id: s for s in ingest_artifact.streams}
        self.vocabulary.prepare(by_stream, self._decode)

    def _kinds(self, images: list[np.ndarray], tracks: dict[int, _Track]) -> dict[int, str]:
        """A track's kind from its phrase, or from the person detector when the phrase says neither."""
        kinds = {obj: phrase_kind(t.phrase) for obj, t in tracks.items()}
        undecided = [obj for obj, kind in kinds.items() if kind is None]
        if undecided:
            per_frame: dict[int, dict[int, np.ndarray]] = {}
            for obj in undecided:
                for pos, (_score, mask) in tracks[obj].per_frame.items():
                    per_frame.setdefault(pos, {})[obj] = mask
            people = SAM2InstanceSegmenter(self.decoder, self.detector, self.cfg).cached_person_boxes(images)
            kinds.update(vote_human_kind(per_frame, len(images), people, self.person_iou, obj_ids=undecided))
        return {obj: kind or "object" for obj, kind in kinds.items()}

    def _observations(self, stream_id: str, frames: list[FrameRef], images: list[np.ndarray],
                      tracks: dict[int, _Track]) -> tuple[list[InstanceObservation], dict[str, np.ndarray]]:
        kinds = self._kinds(images, tracks)
        observations: list[InstanceObservation] = []
        mask_arrays: dict[str, np.ndarray] = {}
        for obj, track in sorted(tracks.items()):
            for pos, (score, mask) in sorted(track.per_frame.items()):
                if not mask.any():
                    continue
                frame_ref = frames[pos]
                instance_id = f"inst:{stream_id}:{int(frame_ref.frame_idx)}:{obj}"
                mask_arrays[instance_id] = mask
                observations.append(InstanceObservation(
                    instance_id=instance_id,
                    track_id=f"{self.track_prefix}:{stream_id}:{obj}",
                    frame_ref=frame_ref,
                    concept=track.phrase,
                    kind=kinds[obj],
                    bbox_xyxy=_mask_bbox_xyxy(mask),
                    confidence=float(score),
                    attributes={"segmenter_backend": self.backend_name, "obj_id": int(obj),
                                "concept_source": "sam3"},
                ))
        observations.sort(key=lambda o: (int(o.frame_ref.frame_idx), o.track_id))
        return observations, mask_arrays

    def _resolved_device(self) -> str:
        import torch

        return str(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))


# -- rung 1: SAM 3 video mode ---------------------------------------------------------------------

class SAM3VideoSegmenter(_Sam3Rung):
    """SAM 3's video mode: every phrase detected on every frame, with persistent ids."""

    backend_name = "sam3_video"
    track_prefix = "sam3"

    def __init__(self, decoder, detector, vocabulary: Sam3Vocabulary, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(decoder, detector, vocabulary, cfg)
        self.cross_prompt_nms_iou = float(self.cfg.get("sam3_cross_prompt_nms_iou", DEFAULT_CROSS_PROMPT_NMS_IOU))
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        common = {"cache_dir": self.cache_dir, "local_files_only": self.local_files_only}
        self._model = Sam3VideoModel.from_pretrained(
            self.model_id, cross_prompt_nms_thresh=self.cross_prompt_nms_iou, **common
        ).to(self._resolved_device(), dtype=torch.bfloat16).eval()
        self._processor = Sam3VideoProcessor.from_pretrained(self.model_id, **common)

    def _track(self, images: list[np.ndarray], phrases: list[str]
               ) -> Iterator[tuple[int, list[tuple[int, str, float, np.ndarray]]]]:
        """``(frame position, [(object id, phrase, score, mask)])`` for each frame, masks on the CPU."""
        import torch

        self._load()
        session = self._processor.init_video_session(
            video=images, inference_device=self._resolved_device(), processing_device="cpu",
            video_storage_device="cpu", dtype=torch.bfloat16,
        )
        session = self._processor.add_text_prompt(session, phrases)
        with torch.inference_mode():
            for output in self._model.propagate_in_video_iterator(session):
                post = self._processor.postprocess_outputs(session, output)
                phrase_of = {int(i): str(p) for p, ids in post["prompt_to_obj_ids"].items() for i in ids}
                masks = post["masks"].cpu().numpy().astype(bool)
                rows = [
                    (int(obj), phrase_of.get(int(obj), ""), float(score), masks[j])
                    for j, (obj, score) in enumerate(zip(post["object_ids"].tolist(), post["scores"].tolist()))
                ]
                yield int(output.frame_idx), rows

    def release(self) -> None:
        self._model = None
        self._processor = None
        _release_gpu()

    def _segment(self, ingest_artifact, selected_frames: list[FrameRef]):
        by_stream = _frames_by_stream(selected_frames)
        self._prepare_vocabulary(ingest_artifact, by_stream)
        observations: list[InstanceObservation] = []
        mask_arrays: dict[str, np.ndarray] = {}
        for stream_id, frames in by_stream.items():
            images = [self._decode(fr) for fr in frames]
            if not images:
                continue
            tracks: dict[int, _Track] = {}
            for pos, rows in self._track(images, self.vocabulary.streams[stream_id].phrases):
                for obj, phrase, score, mask in rows:
                    tracks.setdefault(obj, _Track(phrase=phrase)).per_frame[pos] = (score, mask)
            obs, masks = self._observations(stream_id, frames, images, tracks)
            observations.extend(obs)
            mask_arrays.update(masks)
        return observations, mask_arrays

    def run(self, ingest_artifact, selected_frames: list[FrameRef]):
        """Segment and track; out of GPU memory raises after the model is released, so the next rung
        starts with the memory free."""
        import torch

        out_of_memory = None
        try:
            return self._segment(ingest_artifact, selected_frames)
        except torch.cuda.OutOfMemoryError as exc:
            out_of_memory = str(exc).splitlines()[0] if str(exc) else "CUDA out of memory"
        finally:
            self.release()
        raise RuntimeError(f"SAM 3 video mode ran out of GPU memory ({out_of_memory})")


# -- rung 2: SAM 3 on the first frame, SAM 2 tracking ----------------------------------------------

class SAM3SeedsSAM2Segmenter(_Sam3Rung):
    """SAM 3 finds the objects on each stream's first frame; SAM 2's video predictor tracks them."""

    backend_name = "sam3_seeds_sam2"
    track_prefix = "sam3seed"

    def __init__(self, decoder, detector, vocabulary: Sam3Vocabulary, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(decoder, detector, vocabulary, cfg)
        self.score_threshold = float(self.cfg.get("sam3_seed_score_threshold", DEFAULT_SEED_SCORE_THRESHOLD))

    def _seed(self, first_frames: dict[str, np.ndarray]) -> dict[str, list[Any]]:
        """SAM 3 on each stream's first frame: ``{stream: [measured_space Instance]}``, highest score first.

        The image model never downloads (measured-space loads from the cache only), so the weights are
        fetched first unless the run is offline.
        """
        from measured_space.perception import Sam3Segmenter

        if not self.local_files_only:
            from huggingface_hub import snapshot_download

            snapshot_download(self.model_id, cache_dir=self.cache_dir)
        segmenter = Sam3Segmenter(self.model_id, cache_dir=self.cache_dir, device=self._resolved_device())
        try:
            return {
                stream_id: segmenter.segment(
                    image, self.vocabulary.streams[stream_id].first_frame_phrases(),
                    score_threshold=self.score_threshold,
                )
                for stream_id, image in first_frames.items()
            }
        finally:
            del segmenter
            _release_gpu()

    def run(self, ingest_artifact, selected_frames: list[FrameRef]):
        by_stream = _frames_by_stream(selected_frames)
        self._prepare_vocabulary(ingest_artifact, by_stream)
        seeds = self._seed({sid: self._decode(frames[0]) for sid, frames in by_stream.items() if frames})
        tracker = SAM2InstanceSegmenter(self.decoder, self.detector, self.cfg)
        observations: list[InstanceObservation] = []
        mask_arrays: dict[str, np.ndarray] = {}
        for stream_id, frames in by_stream.items():
            found = seeds.get(stream_id) or []
            if not found:
                continue
            images = [self._decode(fr) for fr in frames]
            per_frame = tracker.track_seeds(images, [inst.mask for inst in found])
            tracks = {i: _Track(phrase=inst.phrase) for i, inst in enumerate(found)}
            for pos, masks in per_frame.items():
                for obj, mask in masks.items():
                    tracks[obj].per_frame[pos] = (found[obj].score, mask)
            obs, masks = self._observations(stream_id, frames, images, tracks)
            observations.extend(obs)
            mask_arrays.update(masks)
        return observations, mask_arrays

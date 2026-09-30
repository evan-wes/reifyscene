"""A run's camera streams: the one place the pipeline learns anything about its input data.

:class:`FrameStream` is a ``torch.utils.data.Dataset`` over one stream's frames -- ``len()`` is
the number of frames that exist and ``stream[pos]`` the frame at that position, so a stage can
hand it to a standard ``DataLoader`` (:func:`collate_frames` batches the records) -- extended
with what the pipeline needs to know and must never guess:

- ``id``: the stream id is data (``cam_00`` for a video, ``0`` for DyCheck's ``0_00060.png``);
- ``frames``: the frame numbers that actually exist -- a folder of frames keeps the numbers
  in its file names, so they can start above 0 or skip -- never ``range(num_frames)``;
- ``fps``: a recorded fact (the video container, or the dataset's own record) or ``None``.
  Never a default: a wrong rate scales every velocity by its error and every acceleration by
  its square. ``seconds()`` refuses when it is unknown; motion is then measured per frame;
- ``source``: what the input is (:mod:`reifyscene.ingest.sources`) -- DAVIS, DyCheck, a video,
  a bare folder -- recognised from the recorded input on disk, with its ground truth;
- ``transform`` / ``render_raster``: the recorded image transform into the geometry's pixels
  (and so the render's), and the raster it lands in;
- ``key()`` / :func:`parse_key`: the ``"<stream>:<frame>"`` format views and cameras are keyed by.

:class:`RunStreams` lists a run's streams from its ingest artifact and their transforms from its
geometry; :meth:`RunStreams.resolve` takes an explicit id or the run's only stream, and refuses
to choose between several. Decoding goes through :mod:`reifyscene.ingest.frame_decoders`, whose
heavy backends are optional.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class StreamError(ValueError):
    """The run has no such stream, or which stream was meant is ambiguous, or it cannot say."""


class UnknownFrameRate(StreamError):
    """A time in seconds was asked of a stream whose frame rate is not recorded anywhere."""


def frame_key(stream_id: str, frame: int) -> str:
    return f"{stream_id}:{int(frame)}"


def parse_key(key: str) -> tuple[str, int]:
    """``"cam_00:12"`` -> ``("cam_00", 12)``. The stream id may itself contain colons."""
    stream, sep, frame = str(key).rpartition(":")
    if not sep or not stream:
        raise ValueError(f"not a frame key: {key!r} (expected '<stream>:<frame>')")
    return stream, int(frame)


class FrameStream(Dataset):
    """One camera stream of a run, as a torch ``Dataset`` of its frames."""

    def __init__(self, ref, transform_record: dict[str, Any] | None = None, *,
                 frame_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 source=None) -> None:
        self.ref = ref                                  # VideoStreamRef as ingest recorded it
        self.id = str(ref.stream_id)
        self.transform_record = transform_record
        self.frame_transform = frame_transform
        self._source = source
        meta = dict(ref.metadata or {})
        self._index: dict[int, str] | None = (
            {int(k): v for k, v in meta["frame_index"].items()} if meta.get("frame_index")
            else None)
        # a video's frames are its decode order; a folder's are the numbers in its file names
        self.frames: list[int] = (sorted(self._index) if self._index is not None
                                  else list(range(int(ref.num_frames))))

    # -- torch Dataset -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, pos: int) -> dict[str, Any]:
        f = self.frames[pos]
        img = self.frame_tensor(f)
        if self.frame_transform is not None:
            img = self.frame_transform(img)
        return {"image": img, "frame": f, "key": self.key(f),
                "seconds": self.seconds(f) if self.fps else None, "stream": self.id}

    def with_transform(self, fn: Callable[[torch.Tensor], torch.Tensor] | None) -> FrameStream:
        """The same stream, its ``__getitem__`` images passed through ``fn``."""
        return FrameStream(self.ref, self.transform_record, frame_transform=fn,
                           source=self._source)

    def in_render(self) -> FrameStream:
        """The same stream with its frames put into the render raster."""
        t = self.transform
        return self.with_transform(
            lambda img: torch.from_numpy(
                np.ascontiguousarray(t.apply_to_image(img.permute(1, 2, 0).numpy(),
                                                      kind="rgb"))).permute(2, 0, 1))

    # -- identity and time ---------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def key(self, frame: int) -> str:
        return frame_key(self.id, frame)

    def has(self, frame: int) -> bool:
        return int(frame) in set(self.frames)

    @property
    def source(self):
        """The :class:`~reifyscene.ingest.sources.DataSource` the frames came from (detected
        from the recorded input on first use)."""
        if self._source is None:
            from reifyscene.ingest.sources import detect_source

            path = Path(self.ref.video_path)
            if not path.exists():
                raise StreamError(f"stream {self.id!r}: its input is not at {path} any more")
            self._source = detect_source(path)
        return self._source

    @property
    def fps(self) -> float | None:
        """The recorded frame rate, or None. Ingest's own number is not used: it invented 30
        for every folder of frames."""
        return self.source.fps

    def seconds(self, frame: int) -> float:
        if not self.fps:
            raise UnknownFrameRate(
                f"stream {self.id!r} ({self.source.kind}) records no frame rate: "
                "give this in frames, not seconds")
        return (int(frame) - self.frames[0]) / float(self.fps)

    # -- pixels ----------------------------------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of its frames as ingested."""
        return int(self.ref.width), int(self.ref.height)

    @property
    def transform(self):
        """The recorded :class:`~reifyscene.geometry.image_transform.ImageTransform` into the
        geometry's (and so the render's) pixels. Raises when the geometry recorded none."""
        from reifyscene.geometry.image_transform import ImageTransform

        if self.transform_record is None:
            raise StreamError(f"stream {self.id!r}: the geometry recorded no image transform "
                              "for it; its pixels cannot be put into the render's without "
                              "guessing")
        return ImageTransform.from_dict(self.transform_record)

    @property
    def render_raster(self) -> tuple[int, int]:
        """(width, height) of the geometry's pixels -- the raster the scene was trained at."""
        w, h = self.transform.output_size
        return int(w), int(h)

    def frame_file(self, frame: int) -> Path:
        """The image file of one frame of a folder stream (where it is now, if it moved)."""
        if self._index is None:
            raise StreamError(f"stream {self.id!r} is a video: its frames are not files")
        if int(frame) not in self._index:
            raise StreamError(f"stream {self.id!r} has no frame {frame}")
        recorded = Path(self._index[int(frame)])
        if recorded.is_file():
            return recorded
        moved = Path(self.ref.video_path) / recorded.name
        if moved.is_file():
            return moved
        raise StreamError(f"stream {self.id!r}: frame {frame} is neither at {recorded} "
                          f"nor in {self.ref.video_path}")

    def frame_tensor(self, frame: int) -> torch.Tensor:
        """Frame ``frame`` as ingested: RGB ``uint8`` ``(3, H, W)``."""
        from reifyscene.ingest.frame_decoders import read_image, read_video_frame

        if self._index is not None:
            return read_image(self.frame_file(frame))
        if int(frame) not in range(int(self.ref.num_frames)):
            raise StreamError(f"stream {self.id!r} has no frame {frame}")
        return read_video_frame(Path(self.ref.video_path), int(frame))

    def frame(self, frame: int) -> np.ndarray:
        """Frame ``frame`` as ingested: RGB ``uint8`` ``(H, W, 3)``."""
        return self.frame_tensor(frame).permute(1, 2, 0).numpy()

    def frame_in_render(self, frame: int) -> np.ndarray:
        """Frame ``frame`` put into the geometry's pixels (RGB ``uint8`` ``(H, W, 3)``)."""
        return self.transform.apply_to_image(self.frame(frame), kind="rgb")

    def mask_in_render(self, mask: np.ndarray) -> np.ndarray:
        """A mask in this stream's source pixels, put into the geometry's -- placed with the
        recorded transform, never resized to fit."""
        return self.transform.apply_to_image(np.asarray(mask).astype(np.uint8),
                                             kind="mask") > 0

    # -- ground truth ------------------------------------------------------------------------------

    def annotation(self, frame: int) -> np.ndarray | None:
        """The source's instance-id annotation for ``frame`` in source pixels, or None when it
        annotates none. Raises when the source has no annotations at all."""
        src = self.source
        if not hasattr(src, "annotation_for"):
            raise StreamError(f"stream {self.id!r}: a {src.kind} source has no per-frame "
                              "annotations")
        return src.annotation_for(self.frame_file(frame))

    def capability(self) -> dict[str, Any]:
        """What this stream can and cannot say, for a stage's manifest."""
        from reifyscene.ingest.frame_decoders import image_backend, video_backend

        src = self.source
        return {"stream": self.id, "frames": len(self.frames),
                "first": self.frames[0] if self.frames else None,
                "last": self.frames[-1] if self.frames else None,
                "contiguous": bool(self.frames)
                and self.frames[-1] - self.frames[0] + 1 == len(self.frames),
                "decoder": image_backend() if self._index is not None else video_backend(),
                "physical_time": "available" if src.fps else "unavailable",
                "source": src.describe()}


def collate_frames(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A ``DataLoader`` collate for :class:`FrameStream` records: images stacked (they share a
    raster within a stream), the rest as lists -- ``seconds`` may be None."""
    return {"image": torch.stack([b["image"] for b in batch]),
            "frame": torch.as_tensor([b["frame"] for b in batch]),
            "key": [b["key"] for b in batch], "seconds": [b["seconds"] for b in batch],
            "stream": [b["stream"] for b in batch]}


# kept for the call sites written against the first version of this module
Stream = FrameStream


def _ref_from_json(raw: dict[str, Any]):
    """A ``VideoStreamRef`` from its serialized form (``{"value": {...}}`` or the bare fields).

    Missing optional fields stay empty rather than defaulted: no fps is filled in (it is read
    from the source), no size is invented (0 means ingest did not record one).
    """
    from reifyscene.common.types import VideoStreamRef

    v = raw.get("value", raw) if isinstance(raw, dict) else {}
    path = v.get("video_path", "")
    path = path.get("value", "") if isinstance(path, dict) else path
    return VideoStreamRef(
        stream_id=str(v["stream_id"]), video_path=Path(str(path)),
        fps=float(v.get("fps") or 0.0), num_frames=int(v.get("num_frames") or 0),
        width=int(v.get("width") or 0), height=int(v.get("height") or 0),
        has_audio=bool(v.get("has_audio", False)), metadata=dict(v.get("metadata") or {}))


class RunStreams:
    """The streams of one run, in ingest order."""

    def __init__(self, streams: list[FrameStream]) -> None:
        self._streams = {s.id: s for s in streams}

    @classmethod
    def from_run(cls, run_dir: Path) -> RunStreams:
        """Streams from ``<run>/ingest``, transforms from ``<run>/geometry`` when it exists.

        Reads the ingest JSON directly (only the stream records are needed), so a run whose
        ingest predates fields added since still lists its streams.
        """
        run_dir = Path(run_dir)
        raw = json.loads((run_dir / "ingest" / "artifact.json").read_text())
        refs = [_ref_from_json(s) for s in raw.get("streams") or []]
        records: dict[str, Any] = {}
        geometry = run_dir / "geometry" / "artifact.json"
        if geometry.is_file():
            meta = json.loads(geometry.read_text()).get("metadata") or {}
            records = meta.get("image_transforms") or {}
        return cls([FrameStream(r, records.get(str(r.stream_id))) for r in refs])

    @property
    def ids(self) -> list[str]:
        return list(self._streams)

    def get(self, stream_id: str) -> FrameStream:
        if stream_id not in self._streams:
            raise StreamError(f"no stream {stream_id!r}; this run has {self.ids}")
        return self._streams[stream_id]

    def only(self) -> FrameStream:
        """The run's stream when it has exactly one."""
        if len(self._streams) != 1:
            raise StreamError(f"this run has {len(self._streams)} streams {self.ids}; "
                              "say which one")
        return next(iter(self._streams.values()))

    def resolve(self, stream_id: str | None) -> FrameStream:
        """``stream_id`` when given, else the only stream."""
        return self.only() if stream_id is None else self.get(stream_id)

    def __iter__(self):
        return iter(self._streams.values())

    def __len__(self) -> int:
        return len(self._streams)

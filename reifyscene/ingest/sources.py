"""Where a stream's frames came from, and what that source says about them.

A :class:`DataSource` is recognised from the input a run was actually given -- the folder or
file recorded at ingest -- by the structure on disk, never by a run's folder name:

- :class:`DyCheckSource`: ``<root>/rgb/<k>x`` with ``<root>/extra.json`` and ``<root>/camera``.
  DyCheck records its frame rate (``extra.json`` ``fps``), so the stream's fps is a fact.
- :class:`DavisSource`: ``<root>/JPEGImages/<resolution>/<sequence>`` with the matching
  ``<root>/Annotations/<resolution>/<sequence>``. DAVIS publishes frames, not a frame rate:
  fps is unknown. Ground truth for a frame is the annotation with the same file stem as the
  frame's own image -- found from the stream's frame file, not by formatting a name.
- :class:`FramesSource` / :class:`VideoSource`: anything else. A video's fps comes from its
  container; a bare folder of frames has none.

A source never guesses a frame rate: an unknown one is ``None``, and motion is then measured per
frame (a wrong rate would scale every velocity by its error, every acceleration by its square).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DataSource:
    """What the pipeline knows about where a stream's frames came from."""

    path: Path
    kind: str = "frames"
    fps: float | None = None
    fps_source: str = "unknown"          # "container", "dataset record", or "unknown"
    sequence: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_ground_truth(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self.path), "sequence": self.sequence,
                "fps": self.fps, "fps_source": self.fps_source, "notes": list(self.notes)}


@dataclass
class FramesSource(DataSource):
    """A folder of frames with no dataset structure around it: no frame rate, no ground truth."""


@dataclass
class VideoSource(DataSource):
    """A video file: its frame rate is whatever the container records."""

    kind: str = "video"


@dataclass
class DyCheckSource(DataSource):
    """A DyCheck (iPhone / Shape-of-Motion layout) sequence."""

    kind: str = "dycheck"
    root: Path | None = None
    factor: int | None = None

    @property
    def has_ground_truth(self) -> bool:
        return True

    @property
    def dycheck(self):
        from reifyscene.ingest.datasets import DyCheckSequence

        return DyCheckSequence(self.root)


@dataclass
class DavisSource(DataSource):
    """A DAVIS sequence: per-frame instance annotations, no frame rate."""

    kind: str = "davis"
    root: Path | None = None
    resolution: str | None = None

    @property
    def has_ground_truth(self) -> bool:
        return True

    @property
    def davis(self):
        from reifyscene.ingest.datasets import DavisSequence

        return DavisSequence(self.root, str(self.resolution), str(self.sequence))

    def annotation_for(self, frame_file: Path) -> np.ndarray | None:
        """The instance-id annotation of one frame (source pixels); None when DAVIS
        annotates no such frame."""
        return self.davis.annotation_for(frame_file)


def detect_source(path: Path) -> DataSource:
    """Recognise what ``path`` (a stream's recorded input) is, from what is on disk."""
    from reifyscene.ingest.datasets import DavisSequence, DyCheckSequence
    from reifyscene.ingest.frame_decoders import video_fps

    path = Path(path)
    if path.is_file():
        fps = video_fps(path)
        return VideoSource(path=path, fps=fps, fps_source="container" if fps else "unknown",
                           notes=[] if fps else ["the container records no frame rate"])
    found = DyCheckSequence.of_frames_dir(path)
    if found is not None:
        seq, factor = found
        fps = seq.fps
        return DyCheckSource(path=path, root=seq.root, factor=factor, sequence=seq.name,
                             fps=fps, fps_source="dataset record" if fps else "unknown")
    davis = DavisSequence.of_frames_dir(path)
    if davis is not None:
        return DavisSource(path=path, root=davis.root, resolution=davis.resolution,
                           sequence=davis.name,
                           notes=["DAVIS publishes frames without a frame rate"])
    return FramesSource(path=path, notes=["a folder of frames records no frame rate"])

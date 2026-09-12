"""All-pairs 2D point tracks (Shape-of-Motion ``2d_tracks`` semantics).

This is the first-class tracker output reifyscene's native point tracker produces and
:func:`reifyscene.lifting.native_track_lift.lift_tracks_3d` consumes. It is distinct
from the single-query-frame trajectories the scaffold pipeline carries in
``TrackArtifact.track_payloads_by_stream`` (CoTracker-style ``points_xy``): an
all-pairs set seeds a *different* grid of foreground query points in **every**
frame and tracks each to **every** frame, with TAPIR's two extra channels
(occlusion + expected-distance logits).

In memory it is a length-``T`` list of ragged per-query arrays
(``tracks[i]`` has shape ``(N_i, T, 4)`` = ``[x, y, occ_logit, dist_logit]``);
``pair(i, j) == tracks[i][:, j]`` is the SoM ``load_pair(i, j)``.

**Persistence (reifyscene's own):** one packed, memmappable array — ``tracks.npy``
of shape ``(ΣN_i, T, 4)`` plus an ``index.json`` (frame_names, query_offsets,
factor, image_hw) — written by :meth:`write` / read by :meth:`load`. This replaces
the O(N²) one-file-per-pair layout. The SoM ``<name_i>_<name_j>.npy`` per-pair
layout survives only as :meth:`write_som_pairs` / :meth:`load_som_pairs` for
interop with the **external** borrowed DyCheck tracks (which ship that way and
``lift_tracks_3d`` reads directly from disk).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AllPairsTracks:
    """Ragged all-pairs 2D tracks for one monocular stream.

    Attributes
    ----------
    frame_names:
        Ordered frame stems (length ``T``); ``tracks[i]`` is the query set seeded
        in ``frame_names[i]`` and the ``j`` axis indexes ``frame_names[j]``.
    tracks:
        Length-``T`` list; ``tracks[i]`` has shape ``(N_i, T, 4)`` with channels
        ``[x, y, occ_logit, dist_logit]``. ``N_i`` varies per query frame (ragged).
    factor:
        Resolution factor the pixel coords live at (e.g. ``1`` for full res).
    image_hw:
        ``(H, W)`` at ``factor`` resolution — metadata for sanity checks.
    """

    frame_names: list[str]
    tracks: list[np.ndarray]
    factor: int
    image_hw: tuple[int, int]

    def __post_init__(self) -> None:
        t = len(self.frame_names)
        if len(self.tracks) != t:
            raise ValueError(f"tracks has {len(self.tracks)} query frames but frame_names has {t}")
        for i, arr in enumerate(self.tracks):
            if arr.ndim != 3 or arr.shape[1] != t or arr.shape[2] != 4:
                raise ValueError(f"tracks[{i}] must be (N_i, {t}, 4); got {tuple(arr.shape)}")

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def query_counts(self) -> list[int]:
        """Number of query points seeded per frame (``N_i`` for each ``i``)."""
        return [int(arr.shape[0]) for arr in self.tracks]

    def pair(self, i: int, j: int) -> np.ndarray:
        """SoM ``load_pair(i, j)``: query-frame ``i``'s points seen in frame ``j``, ``(N_i, 4)``."""
        return self.tracks[i][:, j]

    # -- packed persistence (reifyscene's own, replaces the O(N²) per-pair files) --

    def _pack(self) -> tuple[np.ndarray, np.ndarray]:
        counts = self.query_counts()
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)  # (T+1,)
        if offsets[-1] == 0:
            packed = np.zeros((0, self.num_frames, 4), dtype=np.float32)
        else:
            packed = np.concatenate([np.asarray(a, dtype=np.float32) for a in self.tracks], axis=0)
        return packed, offsets

    def write(self, out_dir: str | Path) -> Path:
        """Write the single packed structure (``tracks.npy`` + ``index.json``) into ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        packed, offsets = self._pack()
        np.save(out_dir / "tracks.npy", packed)
        (out_dir / "index.json").write_text(json.dumps({
            "frame_names": list(self.frame_names),
            "query_offsets": offsets.tolist(),
            "factor": int(self.factor),
            "image_hw": [int(self.image_hw[0]), int(self.image_hw[1])],
        }))
        return out_dir

    @classmethod
    def load(cls, in_dir: str | Path, *, mmap: bool = True) -> "AllPairsTracks":
        """Read the packed structure written by :meth:`write` (memmapped by default)."""
        in_dir = Path(in_dir)
        index = json.loads((in_dir / "index.json").read_text())
        offsets = np.asarray(index["query_offsets"], dtype=np.int64)
        packed = np.load(in_dir / "tracks.npy", mmap_mode="r" if mmap else None)
        tracks = [np.asarray(packed[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]
        return cls(
            frame_names=list(index["frame_names"]),
            tracks=tracks,
            factor=int(index["factor"]),
            image_hw=(int(index["image_hw"][0]), int(index["image_hw"][1])),
        )

    # -- SoM per-pair interop (external borrowed DyCheck tracks only) ----------

    def write_som_pairs(self, out_dir: str | Path) -> Path:
        """Write the legacy SoM ``<name_i>_<name_j>.npy`` per-pair files (interop only)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        names = self.frame_names
        for i, arr in enumerate(self.tracks):
            for j in range(self.num_frames):
                np.save(out_dir / f"{names[i]}_{names[j]}.npy", arr[:, j].astype(np.float32))
        return out_dir

    @classmethod
    def load_som_pairs(
        cls,
        tracks_dir: str | Path,
        frame_names: list[str],
        factor: int,
        image_hw: tuple[int, int],
    ) -> "AllPairsTracks":
        """Reconstruct from a directory of SoM ``<name_i>_<name_j>.npy`` per-pair files."""
        tracks_dir = Path(tracks_dir)
        t = len(frame_names)
        tracks: list[np.ndarray] = []
        for i in range(t):
            cols = [
                np.load(tracks_dir / f"{frame_names[i]}_{frame_names[j]}.npy").astype(np.float32)
                for j in range(t)
            ]
            tracks.append(np.stack(cols, axis=1))  # (N_i, T, 4)
        return cls(frame_names=list(frame_names), tracks=tracks, factor=factor, image_hw=image_hw)

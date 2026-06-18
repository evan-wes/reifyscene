"""All-pairs 2D point tracks (Shape-of-Motion ``2d_tracks`` format).

This is the first-class tracker output room4d's native point tracker produces and
:func:`room4d.lifting.native_track_lift.lift_tracks_3d` consumes. It is distinct
from the single-query-frame trajectories the scaffold pipeline carries in
``TrackArtifact.track_payloads_by_stream`` (CoTracker-style ``points_xy``): an
all-pairs set seeds a *different* grid of foreground query points in **every**
frame and tracks each to **every** frame, with TAPIR's two extra channels
(occlusion + expected-distance logits).

On-disk layout (identical to SoM ``preproc/compute_tracks_torch.py``):

    <dir>/<name_i>_<name_j>.npy   # array (N_i, 4) = [x, y, occ_logit, dist_logit]

one file per ordered (query frame ``i``, target frame ``j``) pair, where ``N_i``
is the number of foreground query points seeded in frame ``i``. The diagonal file
``<name_i>_<name_i>.npy`` carries the exact integer query pixel coords in
``[:, :2]``. ``lift_tracks_3d`` reads ``<name_i>_<name_j>.npy`` as ``load_pair(i, j)``.
"""

from __future__ import annotations

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
        ``[x, y, occ_logit, dist_logit]``. ``N_i`` varies per query frame (ragged),
        so this is a list rather than one stacked array.
    factor:
        Resolution factor the pixel coords live at (e.g. ``1`` for full res). The
        tracks ship at ``<dir>/<factor>x/`` in the dataset layout.
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
            raise ValueError(
                f"tracks has {len(self.tracks)} query frames but frame_names has {t}"
            )
        for i, arr in enumerate(self.tracks):
            if arr.ndim != 3 or arr.shape[1] != t or arr.shape[2] != 4:
                raise ValueError(
                    f"tracks[{i}] must be (N_i, {t}, 4); got {tuple(arr.shape)}"
                )

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def query_counts(self) -> list[int]:
        """Number of query points seeded per frame (``N_i`` for each ``i``)."""
        return [int(arr.shape[0]) for arr in self.tracks]

    def write(self, out_dir: str | Path) -> Path:
        """Write the SoM ``<name_i>_<name_j>.npy`` all-pairs files into ``out_dir``.

        ``out_dir`` is the ``<factor>x`` directory itself (the caller decides where
        that sits — e.g. ``<root>/flow3d_preprocessed/2d_tracks/<factor>x``). Each
        file is ``tracks[i][:, j]`` of shape ``(N_i, 4)``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        names = self.frame_names
        for i, arr in enumerate(self.tracks):
            for j in range(self.num_frames):
                np.save(out_dir / f"{names[i]}_{names[j]}.npy", arr[:, j].astype(np.float32))
        return out_dir

    @classmethod
    def load(
        cls,
        tracks_dir: str | Path,
        frame_names: list[str],
        factor: int,
        image_hw: tuple[int, int],
    ) -> "AllPairsTracks":
        """Reconstruct from a directory of SoM ``<name_i>_<name_j>.npy`` files."""
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

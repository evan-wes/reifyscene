"""Point tracks that span an occlusion gap: geometric evidence two tracklets are one.

Re-identification across a gap is an appearance question by default -- does the thing
that reappeared *look like* the thing that left. That is exactly the question that is
hardest on the footage this pipeline cares about: ``davis_lab-coat_full`` was chosen
because three similar-looking people are an identity stress test, and appearance
features are least discriminative precisely there.

A point track answers a different question. If the same tracked point sat inside
tracklet A's masks before the gap and inside tracklet B's masks after it, that is not a
statement about how they look -- it is the observation that a piece of the world went
from one to the other without the tracker losing it. Two tracklets bridged by several
such points are the same entity, however alike or unlike they appear.

This module counts those bridges. It does not merge anything: the merge stays precision
-first (mutual nearest neighbour plus a ratio test), and a bridge only **relaxes the
appearance threshold** for a pair that has one, so geometry can admit a candidate that
appearance alone would have been too cautious to consider. Nothing here can create a
merge on its own, because a point track drifting off one entity onto another is a real
failure mode and it must not be able to fuse two people by itself.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def tracklet_point_sets(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_indices: np.ndarray,
    points_xy: np.ndarray,
    visibility: np.ndarray | None,
    min_frames: int = 2,
) -> dict[str, set[int]]:
    """Which point tracks were seen inside each tracklet, and often enough to count.

    ``min_frames`` guards against a point that clipped an entity's mask for a single
    frame in passing: one frame inside is a coincidence, several is evidence.
    """
    frames = np.asarray(frame_indices, dtype=int).reshape(-1)
    xy = np.asarray(points_xy, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[0] != frames.shape[0]:
        return {}
    num = xy.shape[1]
    vis = (
        np.ones((frames.shape[0], num), dtype=bool)
        if visibility is None or np.asarray(visibility).size == 0
        else np.asarray(visibility).astype(bool).reshape(frames.shape[0], num)
    )

    counts: dict[str, np.ndarray] = {}
    for t, frame_idx in enumerate(frames):
        masks = masks_by_frame.get(int(frame_idx))
        if not masks:
            continue
        x = np.round(xy[t, :, 0]).astype(int)
        y = np.round(xy[t, :, 1]).astype(int)
        for track_id, mask in masks.items():
            m = np.asarray(mask, dtype=bool)
            h, w = m.shape[:2]
            on = vis[t] & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            hit = np.zeros(num, dtype=bool)
            hit[on] = m[y[on], x[on]]
            key = str(track_id)
            if key not in counts:
                counts[key] = np.zeros(num, dtype=int)
            counts[key] += hit

    return {k: set(np.nonzero(v >= int(min_frames))[0].tolist()) for k, v in counts.items()}


def bridge_counts(point_sets: Mapping[str, set[int]]) -> dict[tuple[str, str], int]:
    """How many point tracks each pair of tracklets shares, keyed by sorted pair."""
    ids = sorted(point_sets)
    out: dict[tuple[str, str], int] = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = len(point_sets[ids[i]] & point_sets[ids[j]])
            if shared:
                out[(ids[i], ids[j])] = int(shared)
    return out

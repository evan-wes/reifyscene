"""Two tracklets in the same place at the same time are one thing.

Re-identification asks whether the thing that reappeared is the thing that left, and it
answers with appearance. That question only makes sense for tracklets whose lifetimes do
not overlap, so :func:`~reifyscene.perception.reid.dino_reid.build_cross_occlusion_merge_map`
deliberately refuses any pair that shares a frame -- two tracklets visible at once are
"over-segmented parts, a different problem than occlusion re-acquisition".

That different problem was never solved anywhere. On ``davis_lab-coat_full`` the
segmenter emitted two tracklets whose masks overlap at **IoU 0.996**, whose 2D centroids
sit **0.1 px** apart, and which are co-visible on all 34 frames they exist for. They are
one object. The graph reported them as two entity nodes, both labelled ``cup``, and split
the point-track evidence 49 / 1 between them.

Appearance cannot settle it -- a duplicate mask and a genuinely co-visible part look
equally alike. Geometry can, and decisively: on that clip the duplicate pair's 3D
centroids are **0.2% of their own size** apart, while two entities that really are
different things sit **11 times their own size** apart. That is not a threshold anyone
has to tune carefully; it is a gap of three orders of magnitude.

Separation is measured **relative to the pair's own extent**, never in world units: a
monocular reconstruction is fixed only up to a similarity, so an absolute distance here
would be a constant tuned to one clip -- the defect this pipeline keeps finding in
itself.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _root(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def build_covisible_merge_map(
    positions: Mapping[str, Mapping[int, np.ndarray]],
    extents: Mapping[str, np.ndarray],
    *,
    max_separation: float = 0.25,
    min_shared_frames: int = 3,
) -> dict[str, str]:
    """Fold tracklets that occupy the same 3D place at the same time onto one id.

    Args:
        positions: ``track_id -> {frame_idx: xyz}``, in the graph's world frame.
        extents: ``track_id -> per-axis extent`` for the same tracks.
        max_separation: how far apart two tracklets may sit and still be one thing,
            as a fraction of their own mean size. The default is two orders of
            magnitude looser than the duplicates it is meant to catch and an order
            tighter than the closest genuinely-distinct pair measured.
        min_shared_frames: how many frames the two must both be placed on. One
            coincidence is not evidence of identity.

    Returns ``{track_id: canonical_id}`` for every non-canonical member of a merged
    group, the same shape the appearance merge returns, so the two compose through
    :meth:`~reifyscene.perception.reid.reid_artifact.ReidArtifact.canonical`. The
    canonical is the earliest-appearing member, then the smallest id.

    Merging is transitive -- three duplicate masks of one object fold onto one id, not
    into a chain of pairs.
    """
    tracks = sorted(t for t in positions if t in extents and positions[t])
    parent = {t: t for t in tracks}

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = tracks[i], tracks[j]
            shared = sorted(set(positions[a]) & set(positions[b]))
            if len(shared) < int(min_shared_frames):
                continue
            size = 0.5 * (
                float(np.linalg.norm(np.asarray(extents[a], dtype=float)))
                + float(np.linalg.norm(np.asarray(extents[b], dtype=float)))
            )
            if size <= 0:
                continue
            gaps = [
                float(np.linalg.norm(np.asarray(positions[a][f]) - np.asarray(positions[b][f])))
                for f in shared
            ]
            # A median, so one badly-placed frame cannot join two things or keep one apart.
            if float(np.median(gaps)) / size < float(max_separation):
                ra, rb = _root(parent, a), _root(parent, b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    groups: dict[str, list[str]] = {}
    for t in tracks:
        groups.setdefault(_root(parent, t), []).append(t)

    merge_map: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda t: (min(positions[t]), t))
        for t in members:
            if t != canonical:
                merge_map[t] = canonical
    return merge_map

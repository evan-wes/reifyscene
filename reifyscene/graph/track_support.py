"""Which point tracks are evidence for which entity.

The graph holds real things -- people, objects, the room -- and point tracks are the
evidence that they are there and where they went. This module decides, from the
segment stage's masks, which tracks support which entity.

A track supports an entity when it lands inside that entity's mask often enough: over
the frames where the entity has a mask AND the track is visible, it is inside at least
``min_ratio`` of the time, on at least ``min_frames`` of them. A track supports at most
one entity -- the one it is inside most often -- because a point on the world is one
point, not a member of two things. Tracks inside nothing are the room's evidence;
tracks inside something but never enough are reported separately rather than quietly
handed to the background.

Positions are in ingest-frame pixels, the space the tracking stage records
``points_xy`` in and the space the masks are drawn in. A point is tested at its nearest
pixel (:func:`numpy.round`, the convention
:func:`~reifyscene.lifting.lift.lift_pixels` samples depth with); one that falls
outside the frame counts as not visible rather than being clamped onto the border.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrackSupport:
    """The evidence relation between point tracks and entities.

    ``inside_frames[entity_id][track_id]`` is how many frames that track was inside
    that entity's mask, counted only for the entities a track ended up supporting.
    ``shared_frames`` is how many frames the two could have been compared on -- the
    denominator of the ratio -- so a node can say how strong its evidence is.
    """

    entity_of_track: dict[int, str] = field(default_factory=dict)
    inside_frames: dict[str, dict[int, int]] = field(default_factory=dict)
    shared_frames: dict[str, dict[int, int]] = field(default_factory=dict)
    ambiguous_track_ids: list[int] = field(default_factory=list)
    free_track_ids: list[int] = field(default_factory=list)

    def track_ids_for(self, entity_id: str) -> list[int]:
        """The ids of the tracks supporting one entity, ascending."""
        return sorted(self.inside_frames.get(entity_id, {}))


def assign_track_support(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_indices: np.ndarray,
    points_xy: np.ndarray,
    visibility: np.ndarray | None,
    track_ids: np.ndarray,
    min_ratio: float = 0.5,
    min_frames: int = 2,
) -> TrackSupport:
    """Decide which tracks are evidence for which entity.

    ``masks_by_frame`` maps a frame index to that frame's entity masks, keyed by the
    entity's graph node id: boolean arrays (H, W) in ingest-frame pixels. The tracking
    arrays are as the tracking stage records them -- ``frame_indices`` (T,),
    ``points_xy`` (T, N, 2), ``visibility`` (T, N) or None when every point is visible,
    ``track_ids`` (N,).
    """
    frames = np.asarray(frame_indices, dtype=int).reshape(-1)
    xy = np.asarray(points_xy, dtype=np.float64)
    ids = np.asarray(track_ids, dtype=int).reshape(-1)
    num_tracks = ids.shape[0]
    if xy.ndim != 3 or xy.shape[0] != frames.shape[0] or xy.shape[1] != num_tracks:
        raise ValueError(
            f"points_xy must be ({frames.shape[0]}, {num_tracks}, 2) to match the frames "
            f"and track ids; got {xy.shape}"
        )
    vis = (
        np.ones((frames.shape[0], num_tracks), dtype=bool)
        if visibility is None or np.asarray(visibility).size == 0
        else np.asarray(visibility).astype(bool).reshape(frames.shape[0], num_tracks)
    )

    inside = _Counter(num_tracks)
    shared = _Counter(num_tracks)
    for t, frame_idx in enumerate(frames):
        masks = masks_by_frame.get(int(frame_idx))
        if not masks:
            continue
        x = np.round(xy[t, :, 0]).astype(int)
        y = np.round(xy[t, :, 1]).astype(int)
        for entity_id, mask in masks.items():
            m = np.asarray(mask, dtype=bool)
            height, width = m.shape[:2]
            # Outside the frame is not evidence either way: it is not visible here.
            on_frame = vis[t] & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            hit = np.zeros(num_tracks, dtype=bool)
            hit[on_frame] = m[y[on_frame], x[on_frame]]
            inside.add(entity_id, hit)
            shared.add(entity_id, on_frame)

    entity_of_track: dict[int, str] = {}
    inside_frames: dict[str, dict[int, int]] = {}
    shared_frames: dict[str, dict[int, int]] = {}
    ambiguous: list[int] = []
    free: list[int] = []
    for i, track_id in enumerate(ids):
        best: tuple[int, float, str] | None = None
        seen_inside = False
        for entity_id in inside.entities():
            inside_count = int(inside.counts[entity_id][i])
            if inside_count <= 0:
                continue
            seen_inside = True
            shared_count = int(shared.counts[entity_id][i])
            ratio = inside_count / shared_count if shared_count else 0.0
            if inside_count < min_frames or ratio < min_ratio:
                continue
            # Most frames inside wins; then the cleaner ratio; then the id, so the
            # answer does not depend on dictionary order.
            candidate = (inside_count, ratio, entity_id)
            if best is None or candidate[:2] > best[:2] or (
                candidate[:2] == best[:2] and entity_id < best[2]
            ):
                best = candidate
        if best is None:
            (ambiguous if seen_inside else free).append(int(track_id))
            continue
        entity_id = best[2]
        entity_of_track[int(track_id)] = entity_id
        inside_frames.setdefault(entity_id, {})[int(track_id)] = int(inside.counts[entity_id][i])
        shared_frames.setdefault(entity_id, {})[int(track_id)] = int(shared.counts[entity_id][i])

    return TrackSupport(
        entity_of_track=entity_of_track,
        inside_frames=inside_frames,
        shared_frames=shared_frames,
        ambiguous_track_ids=sorted(ambiguous),
        free_track_ids=sorted(free),
    )


def positions_by_frame(
    columns: np.ndarray,
    *,
    lifted_frame_indices: np.ndarray,
    lifted_xyz: np.ndarray,
    lifted_valid: np.ndarray,
) -> list[dict[str, object]]:
    """Where an entity was, per frame, from the 3D points of the tracks supporting it.

    ``columns`` are the entity's columns into the lifted arrays. The position is the
    median of its supporting tracks' lifted points on that frame -- a median, not a
    mean, so one badly lifted point does not drag the entity across the room. Frames
    where no supporting track lifted are left out; ``num_points`` says how many points
    the position rests on.
    """
    cols = np.asarray(columns, dtype=int).reshape(-1)
    xyz = np.asarray(lifted_xyz, dtype=np.float64)
    valid = np.asarray(lifted_valid, dtype=bool)
    frames = np.asarray(lifted_frame_indices, dtype=int).reshape(-1)
    if cols.size == 0 or xyz.ndim != 3 or xyz.shape[0] == 0:
        return []

    out: list[dict[str, object]] = []
    for k, frame_idx in enumerate(frames):
        if k >= xyz.shape[0]:
            break
        ok = valid[k, cols] if valid.size else np.ones(cols.shape, dtype=bool)
        points = xyz[k, cols][ok]
        points = points[np.isfinite(points).all(axis=1)] if points.size else points
        if points.shape[0] == 0:
            continue
        centre = np.median(points, axis=0)
        out.append(
            {
                "frame_idx": int(frame_idx),
                "xyz": [float(v) for v in centre],
                "num_points": int(points.shape[0]),
            }
        )
    return out


class _Counter:
    """Per-entity frame counts over the same track axis."""

    def __init__(self, num_tracks: int) -> None:
        self.num_tracks = int(num_tracks)
        self.counts: dict[str, np.ndarray] = {}

    def add(self, entity_id: str, flags: np.ndarray) -> None:
        """Add one frame's boolean flags to an entity's running count."""
        row = self.counts.get(entity_id)
        if row is None:
            row = np.zeros((self.num_tracks,), dtype=np.int32)
            self.counts[entity_id] = row
        row += flags.astype(np.int32)

    def entities(self) -> list[str]:
        """Entity ids in a stable order."""
        return sorted(self.counts)

"""Query points for the tracker, placed on the entities rather than over the frame.

CoTracker seeds a uniform grid on the clip's first frame and follows it. That is the
wrong question to ask of a scene with entities in it, for two reasons a denser grid
cannot fix:

- **Whatever is not covered at the seed frame is never covered.** On
  ``davis_lab-coat_full`` the two people occupy 3.6% and 3.2% of frame 0 -- they walk
  toward the camera and end at 16% -- so a 12x12 grid puts about five points on each
  before any filtering, and 130 of its 144 points never touch an entity at all. An
  entity that appears *after* the seed frame gets nothing, ever.
- **A grid cannot be asked about a thing.** The jobs only tracks can do -- bridging an
  occlusion, telling one instance from another, judging whether a mask is one object --
  all need points *on the entity*, in numbers that survive it turning away or being
  briefly hidden.

So the masks that ``segment`` already produces choose the query points: each entity is
sampled inside its own mask, at the frame it first appears and again after any gap long
enough that its original points are unlikely to have survived. The uniform grid stays
alongside them -- the background still has to be tracked, and it is what the
static/dynamic split is measured against.

Points are sampled deterministically from a seeded generator, so the same run twice
gives the same queries; a tracker that is reproducible only up to a random sample is not
reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def seed_count_for(area_px: int, *, min_points: int, per_kilopixel: float, max_points: int) -> int:
    """How many query points one entity earns on one frame.

    Proportional to masked area, so a person gets more than a handbag, with a floor so
    a small object still gets enough points to survive losing some, and a ceiling so one
    near-frame-filling entity cannot crowd out the rest of the budget.
    """
    scaled = int(round(per_kilopixel * area_px / 1000.0))
    return int(np.clip(scaled, min_points, max_points))


def seed_frames_for(frames: Sequence[int], *, reseed_gap: int) -> list[int]:
    """The frames an entity should be seeded on: its first, and its first after any gap.

    A gap means the entity was unobserved -- occluded, out of frame, or missed. Points
    seeded before it may or may not have survived, and an entity that came back with no
    points on it is invisible to everything downstream. Re-seeding makes a second point
    set for one entity, which is a problem for identity to resolve rather than a reason
    to lose the entity.
    """
    ordered = sorted({int(f) for f in frames})
    if not ordered:
        return []
    seeds = [ordered[0]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur - prev > int(reseed_gap):
            seeds.append(cur)
    return seeds


def entity_seed_queries(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_indices: Sequence[int],
    model_wh: tuple[int, int],
    source_wh: tuple[int, int],
    min_points: int = 16,
    per_kilopixel: float = 2.0,
    max_points: int = 256,
    reseed_gap: int = 4,
    erode_px: int = 1,
    rng_seed: int = 0,
) -> np.ndarray:
    """Query points on every entity, as ``(N, 3)`` rows of ``(t, x, y)``.

    ``masks_by_frame`` maps a frame index to that frame's masks keyed by track id, in
    **ingest pixels**; ``frame_indices`` is the frame list the tracker will actually run
    on, and ``t`` indexes into it (not the video's own numbering). ``x`` and ``y`` are in
    the tracker's **model pixels**, scaled from ``source_wh`` to ``model_wh``.

    Frames an entity is masked on that the tracker is not running on are skipped: a query
    the tracker cannot place is worse than no query.
    """
    frames = [int(f) for f in frame_indices]
    t_of_frame = {f: i for i, f in enumerate(frames)}
    if not frames or not masks_by_frame:
        return np.zeros((0, 3), dtype=np.float32)

    by_entity: dict[str, list[int]] = {}
    for frame_idx, masks in masks_by_frame.items():
        for track_id in masks or {}:
            by_entity.setdefault(str(track_id), []).append(int(frame_idx))

    sx = float(model_wh[0]) / float(source_wh[0]) if source_wh[0] else 1.0
    sy = float(model_wh[1]) / float(source_wh[1]) if source_wh[1] else 1.0
    rng = np.random.default_rng(int(rng_seed))

    rows: list[np.ndarray] = []
    for track_id in sorted(by_entity):
        observed = [f for f in by_entity[track_id] if f in t_of_frame]
        for frame_idx in seed_frames_for(observed, reseed_gap=reseed_gap):
            mask = masks_by_frame.get(frame_idx, {}).get(track_id)
            if mask is None:
                continue
            m = np.asarray(mask, dtype=bool)
            if erode_px > 0 and m.any():
                import cv2

                k = 2 * int(erode_px) + 1
                eroded = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
                m = eroded if eroded.any() else m
            ys, xs = np.nonzero(m)
            if xs.size == 0:
                continue
            count = min(
                seed_count_for(
                    int(xs.size), min_points=min_points, per_kilopixel=per_kilopixel,
                    max_points=max_points,
                ),
                int(xs.size),
            )
            pick = rng.choice(xs.size, size=count, replace=False)
            t = float(t_of_frame[frame_idx])
            rows.append(
                np.stack(
                    [
                        np.full(count, t, dtype=np.float32),
                        (xs[pick].astype(np.float32) + 0.5) * sx,
                        (ys[pick].astype(np.float32) + 0.5) * sy,
                    ],
                    axis=-1,
                )
            )

    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def queries_for_window(
    queries: np.ndarray, *, start: int, end: int, grid_xy: np.ndarray
) -> np.ndarray:
    """The window's own queries: its slice of the entity points, plus the grid.

    Entity ``t`` values are rebased onto the window, and the grid is seeded at the
    window's first frame so the background is tracked in every window -- including one
    that happens to contain no entity at all.
    """
    grid = np.asarray(grid_xy, dtype=np.float32).reshape(-1, 2)
    rows = [np.concatenate([np.zeros((grid.shape[0], 1), np.float32), grid], axis=-1)]
    q = np.asarray(queries, dtype=np.float32).reshape(-1, 3)
    if q.size:
        inside = (q[:, 0] >= start) & (q[:, 0] < end)
        if inside.any():
            local = q[inside].copy()
            local[:, 0] -= float(start)
            rows.append(local)
    return np.concatenate(rows, axis=0).astype(np.float32)

"""Where an entity is, from the geometry's own dense points.

The graph asks point tracks where an entity is. On ``davis_lab-coat_full`` the
tracker seeded five points on one person and none on the other, so the graph
answered with five positions and zero -- across forty-seven frames in which both
were masked the whole time. The evidence was never missing. The geometry stage
already writes a world-space point for **every pixel** of every frame
(``pose_refine/pointmaps/``), in the canonical ground-plane frame, and ``segment``
already says which pixels belong to which entity. Intersecting the two gives a
position **and an extent** for every entity on every frame it is masked, with no
dependence on whether a sparse track happened to land on it.

Point tracks keep the jobs only they can do -- motion, identity through occlusion,
objectness. Position was never one of them.

Two things have to line up, and both are why this lives here rather than in a caller:

- **The geometry's pixels, not the ingest frame's.** VGGT resizes each frame and
  crops it square, so its pointmaps describe a different raster than the masks do.
  Callers pass the stream's recorded
  :class:`~reifyscene.geometry.image_transform.ImageTransform` and the mask is mapped
  through it -- never guessed from sizes, because the loaders round differently.
- **A silhouette is not a surface.** A mask's boundary ring straddles the depth
  discontinuity between the entity and whatever is behind it, so the pixels most
  likely to be *included* by a mask are the ones least likely to *belong* to the
  object. Left alone, a person's centroid drifts into the trees behind them. The mask
  is eroded first, and what survives is trimmed by robust distance from its own
  median.

Magnitudes here are only ever comparable to each other: a monocular reconstruction is
fixed only up to a similarity, so an extent of ``0.04`` means nothing on its own and
means "four times that one" next to an extent of ``0.01``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

_MAD_TO_SIGMA = 1.4826  # scale that makes a MAD comparable to a standard deviation


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink a boolean mask by ``px`` pixels, keeping it non-empty.

    Erosion is what removes the boundary ring where a mask and the background share
    pixels. A small or thin entity can erode to nothing, and an empty mask is worse
    evidence than a slightly dirty one, so the original is returned in that case and
    the caller is told by ``eroded=False`` in the record.
    """
    m = np.asarray(mask, dtype=bool)
    if px <= 0 or not m.any():
        return m
    import cv2

    k = 2 * int(px) + 1
    kernel = np.ones((k, k), np.uint8)
    out = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out if out.any() else m


def robust_centre_and_extent(
    points: np.ndarray, *, trim_sigmas: float = 3.0
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Median centre and robust per-axis extent, after trimming distant outliers.

    The centre is a median, not a mean, so a handful of points that lifted onto the
    background cannot drag it. Points are then trimmed on their distance from that
    median -- background bleed sits far beyond an entity's own spread, while the
    spread along a person's body is legitimate and survives. The extent is a per-axis
    MAD scaled to be read like a standard deviation, which is what a relation needs to
    ask whether one thing is above or inside another.

    Returns ``(centre, extent, num_kept, num_trimmed)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts.reshape(0, 3)
    if pts.shape[0] == 0:
        return np.full(3, np.nan), np.full(3, np.nan), 0, 0

    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre, axis=1)
    med_d = float(np.median(dist))
    mad_d = float(np.median(np.abs(dist - med_d))) * _MAD_TO_SIGMA
    if mad_d > 0:
        keep = dist <= med_d + trim_sigmas * mad_d
        # Never trim everything: a degenerate spread means every point is the same
        # point, which is a fine answer, not an error.
        kept = pts[keep] if keep.any() else pts
    else:
        kept = pts

    centre = np.median(kept, axis=0)
    extent = np.median(np.abs(kept - centre), axis=0) * _MAD_TO_SIGMA
    return centre, extent, int(kept.shape[0]), int(pts.shape[0] - kept.shape[0])


def entity_positions_from_pointmaps(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    *,
    pointmap_for_frame: Callable[[int], tuple[np.ndarray, np.ndarray | None] | None],
    transform: Any = None,
    min_confidence: float = 0.0,
    erode_px: int = 2,
    trim_sigmas: float = 3.0,
    min_points: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """Per-entity, per-frame position and extent from the geometry's dense points.

    ``masks_by_frame`` maps a frame index to that frame's entity masks keyed by graph
    node id, in **ingest-frame pixels** -- the same structure
    :func:`~reifyscene.graph.track_support.assign_track_support` consumes.
    ``pointmap_for_frame`` returns that frame's ``(points, confidence)`` in the
    geometry's pixel space, or ``None`` when the frame has no pointmap; it is a
    callable so a caller can read seventy megabytes of frames one at a time rather
    than holding them all. ``transform`` is the stream's recorded
    :class:`ImageTransform`; ``None`` means the masks are already in the geometry's
    raster.

    Frames where fewer than ``min_points`` pixels survive are left out entirely rather
    than reported with a position nobody should trust. Each record carries the counts
    that earned it, so a consumer can weigh it instead of taking it on faith.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for frame_idx in sorted(masks_by_frame):
        loaded = pointmap_for_frame(int(frame_idx))
        if loaded is None:
            continue
        points, confidence = loaded
        points = np.asarray(points)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"pointmap for frame {frame_idx} must be (H, W, 3); got {points.shape}"
            )
        height, width = points.shape[:2]

        conf_ok = np.ones((height, width), dtype=bool)
        if confidence is not None and np.asarray(confidence).size:
            conf_ok = np.asarray(confidence, dtype=np.float64) >= float(min_confidence)
        finite = np.isfinite(points).all(axis=-1)

        for node_id, mask in (masks_by_frame[frame_idx] or {}).items():
            m = np.asarray(mask, dtype=bool)
            if transform is not None:
                m = np.asarray(transform.apply_to_image(m, kind="mask"), dtype=bool)
            if m.shape[:2] != (height, width):
                raise ValueError(
                    f"mask for {node_id!r} on frame {frame_idx} is {m.shape[:2]} but the "
                    f"pointmap is {(height, width)}; pass the stream's image transform"
                )
            eroded = erode_mask(m, erode_px)
            sel = eroded & finite & conf_ok
            if int(sel.sum()) < min_points:
                continue
            centre, extent, kept, trimmed = robust_centre_and_extent(
                points[sel], trim_sigmas=trim_sigmas
            )
            if kept < min_points or not np.isfinite(centre).all():
                continue
            out.setdefault(str(node_id), []).append(
                {
                    "frame_idx": int(frame_idx),
                    "xyz": [float(v) for v in centre],
                    "extent": [float(v) for v in extent],
                    "num_points": int(kept),
                    "num_trimmed": int(trimmed),
                    "eroded": bool(eroded is not m),
                    "source": "pointmap",
                }
            )
    return out

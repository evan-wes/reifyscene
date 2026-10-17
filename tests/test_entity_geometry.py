"""Tests for placing entities from the geometry's dense pointmaps.

The pure helpers are exercised directly; ``entity_positions_from_pointmaps`` is driven
with hand-built pointmaps so the mask-to-raster mapping, the edge-bleed trimming and
the refusal to answer on too little evidence are all covered without a run on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.lifting.entity_geometry import (
    entity_positions_from_pointmaps,
    erode_mask,
    robust_centre_and_extent,
)

# --- erode_mask --------------------------------------------------------------

def test_erode_removes_the_boundary_ring() -> None:
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    out = erode_mask(m, 2)
    assert out.sum() < m.sum()
    assert out[10, 10]           # interior survives
    assert not out[5, 5]         # the corner of the ring does not


def test_erode_keeps_a_thin_mask_rather_than_emptying_it() -> None:
    """An empty mask is worse evidence than a slightly dirty one."""
    m = np.zeros((20, 20), bool)
    m[10, 8:12] = True           # one pixel tall -- erosion would wipe it
    out = erode_mask(m, 2)
    assert out.any()
    assert np.array_equal(out, m)  # fell back to the original


def test_erode_is_a_noop_at_zero() -> None:
    m = np.zeros((6, 6), bool)
    m[2:4, 2:4] = True
    assert np.array_equal(erode_mask(m, 0), m)


# --- robust_centre_and_extent ------------------------------------------------

def test_centre_is_robust_to_background_bleed() -> None:
    """The whole point: a few pixels on the trees must not move the person.

    Twenty points on an object at z=0, then five that leaked onto a background ten
    units behind it. A mean lands nowhere; the median stays on the object.
    """
    obj = np.random.default_rng(0).normal(0, 0.05, size=(20, 3))
    bleed = np.array([[0.0, 0.0, 10.0]] * 5)
    pts = np.vstack([obj, bleed])

    centre, extent, kept, trimmed = robust_centre_and_extent(pts)

    assert abs(float(np.asarray(pts).mean(axis=0)[2])) > 1.0   # the mean is dragged
    assert abs(float(centre[2])) < 0.2                         # the median is not
    assert trimmed == 5 and kept == 20                         # and the bleed is named
    assert float(extent[2]) < 1.0


def test_extent_reports_the_shape() -> None:
    """A tall thin thing must read as tall and thin -- above/below depends on it."""
    rng = np.random.default_rng(1)
    pts = np.stack([rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200), rng.normal(0, 0.5, 200)], 1)
    _centre, extent, _kept, _trimmed = robust_centre_and_extent(pts)
    assert extent[2] > 10 * extent[0]
    assert extent[2] > 10 * extent[1]


def test_degenerate_spread_is_an_answer_not_an_error() -> None:
    pts = np.tile([1.0, 2.0, 3.0], (12, 1))
    centre, extent, kept, trimmed = robust_centre_and_extent(pts)
    assert np.allclose(centre, [1.0, 2.0, 3.0])
    assert np.allclose(extent, 0.0)
    assert kept == 12 and trimmed == 0


def test_no_points_is_nan_not_a_crash() -> None:
    centre, extent, kept, trimmed = robust_centre_and_extent(np.zeros((0, 3)))
    assert np.isnan(centre).all() and np.isnan(extent).all()
    assert kept == 0 and trimmed == 0


# --- entity_positions_from_pointmaps -----------------------------------------

def _pointmap(h: int, w: int, xyz) -> tuple[np.ndarray, np.ndarray]:
    pts = np.tile(np.asarray(xyz, dtype=np.float32), (h, w, 1))
    return pts, np.ones((h, w), np.float32)


def _square(h: int, w: int, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_places_every_entity_on_every_masked_frame() -> None:
    masks = {f: {"a": _square(40, 40, (5, 35, 5, 35))} for f in range(3)}
    out = entity_positions_from_pointmaps(
        masks, pointmap_for_frame=lambda f: _pointmap(40, 40, (1.0, 2.0, 3.0))
    )
    assert [r["frame_idx"] for r in out["a"]] == [0, 1, 2]
    assert np.allclose(out["a"][0]["xyz"], [1.0, 2.0, 3.0])
    assert out["a"][0]["source"] == "pointmap"
    assert out["a"][0]["eroded"] is True


def test_frames_without_a_pointmap_are_skipped_not_faked() -> None:
    masks = {f: {"a": _square(40, 40, (5, 35, 5, 35))} for f in range(3)}
    out = entity_positions_from_pointmaps(
        masks,
        pointmap_for_frame=lambda f: None if f == 1 else _pointmap(40, 40, (0.0, 0.0, 1.0)),
    )
    assert [r["frame_idx"] for r in out["a"]] == [0, 2]


def test_too_little_evidence_declines_to_answer() -> None:
    masks = {0: {"a": _square(40, 40, (20, 22, 20, 22))}}  # 4 px, erodes to 4, < min_points
    out = entity_positions_from_pointmaps(
        masks, pointmap_for_frame=lambda f: _pointmap(40, 40, (0.0, 0.0, 1.0)), min_points=16
    )
    assert out == {}


def test_low_confidence_pixels_are_excluded() -> None:
    def pm(_f):
        pts, conf = _pointmap(40, 40, (0.0, 0.0, 1.0))
        conf[:] = 0.1
        return pts, conf
    masks = {0: {"a": _square(40, 40, (5, 35, 5, 35))}}
    assert entity_positions_from_pointmaps(masks, pointmap_for_frame=pm, min_confidence=0.5) == {}
    assert entity_positions_from_pointmaps(masks, pointmap_for_frame=pm, min_confidence=0.0)


def test_a_mask_in_the_wrong_raster_raises_rather_than_guessing() -> None:
    """Sizes must never be reconciled by squashing -- pass the transform instead."""
    masks = {0: {"a": _square(80, 80, (10, 70, 10, 70))}}
    with pytest.raises(ValueError, match="image transform"):
        entity_positions_from_pointmaps(
            masks, pointmap_for_frame=lambda f: _pointmap(40, 40, (0.0, 0.0, 1.0))
        )


def test_transform_maps_the_mask_into_the_geometry_raster() -> None:
    from reifyscene.geometry.image_transform import ImageTransform, Resize

    masks = {0: {"a": _square(80, 80, (10, 70, 10, 70))}}
    tf = ImageTransform(source_size=(80, 80), ops=(Resize(size=(40, 40), resample="nearest"),))
    out = entity_positions_from_pointmaps(
        masks, pointmap_for_frame=lambda f: _pointmap(40, 40, (4.0, 5.0, 6.0)), transform=tf
    )
    assert np.allclose(out["a"][0]["xyz"], [4.0, 5.0, 6.0])


def test_two_entities_are_placed_independently() -> None:
    def pm(_f):
        pts = np.zeros((40, 40, 3), np.float32)
        pts[:, :20] = (1.0, 0.0, 0.0)
        pts[:, 20:] = (9.0, 0.0, 0.0)
        return pts, np.ones((40, 40), np.float32)
    masks = {
        0: {"left": _square(40, 40, (5, 35, 2, 18)),
            "right": _square(40, 40, (5, 35, 22, 38))}
    }
    out = entity_positions_from_pointmaps(masks, pointmap_for_frame=pm)
    assert np.allclose(out["left"][0]["xyz"], [1.0, 0.0, 0.0])
    assert np.allclose(out["right"][0]["xyz"], [9.0, 0.0, 0.0])

"""The removal tests T1-T4 (reifyscene/eval/benchmarks/removal), on synthetic views.

What these pin: background another frame saw behind an entity is found, with that frame's
colour, and background no frame saw is not; a point in front of what a frame shows is not
background; the floor's band keeps away from every mask's edge; a slab holds what lies between
the entity's surface and its width behind it, only over its own pixels; a trace needs a view
the detector could judge and a detection over the entity's own region; and the pass marks are
the owner's.
"""

from __future__ import annotations

import numpy as np

from reifyscene.eval.benchmarks.edit_damage.silhouettes import Silhouette
from reifyscene.eval.benchmarks.removal import PASS, verdict
from reifyscene.eval.benchmarks.removal.seen import BackgroundProjector, SquaredError, band
from reifyscene.eval.benchmarks.removal.trace import entity_region, has_trace
from reifyscene.eval.benchmarks.removal.views import RunViews
from reifyscene.eval.benchmarks.removal.volume import EntityVolume, entity_volume, entity_width

W, H, FOCAL, DEPTH = 48, 32, 40.0, 5.0
NODE = "object:cam_00:sam2:cam_00:1"


def _views(boxes: dict[int, tuple[int, int]], *, depth: float = DEPTH) -> RunViews:
    """A static camera over a textured wall at ``depth``; the entity is a 10x10 box at column
    ``boxes[frame]`` (left, top), standing 1 unit in front of the wall."""
    K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]])
    yy, xx = np.mgrid[0:H, 0:W]
    wall = np.stack([xx * 5, yy * 7, (xx + yy) * 3], -1).astype(np.uint8)
    v = RunViews(frames=sorted(boxes), photos={}, depth={}, K={}, w2c={}, id_maps={},
                 class_of={NODE: 1}, labels={NODE: "box"})
    for f, (x0, y0) in boxes.items():
        ids = np.zeros((H, W), np.uint16)
        ids[y0:y0 + 10, x0:x0 + 10] = 1
        d = np.full((H, W), depth, np.float32)
        d[ids == 1] = depth - 1.0
        photo = wall.copy()
        photo[ids == 1] = (250, 0, 0)
        v.photos[f], v.depth[f], v.K[f], v.w2c[f], v.id_maps[f] = photo, d, K, np.eye(4), ids
    return v


def test_background_behind_a_moving_entity_is_seen_with_the_frame_that_saw_it() -> None:
    views = _views({0: (5, 10), 1: (15, 10), 2: (25, 10)})
    pr = BackgroundProjector(views, device="cpu", erode_px=0).project(1)
    m = views.mask(NODE, 1)
    assert pr.seen[m].all()                       # frames 0 and 2 saw all of it
    expected = views.photos[0].astype(np.float32) / 255.0
    ok = m & (pr.source == 0)
    assert ok.any() and np.allclose(pr.colour[ok], expected[ok], atol=1e-6)


def test_background_no_frame_saw_is_unseen() -> None:
    views = _views({0: (15, 10), 1: (15, 10), 2: (16, 10)})   # it barely moves
    pr = BackgroundProjector(views, device="cpu", erode_px=0).project(1)
    m = views.mask(NODE, 1)
    assert not pr.seen[m].all() and pr.seen[m].any()
    assert not pr.seen[11:19, 17:24].any()          # covered on every frame


def test_a_point_in_front_of_what_the_frame_shows_is_not_background() -> None:
    views = _views({0: (5, 10), 1: (15, 10)})
    views.depth[0][views.id_maps[0] == 0] = 2.0     # frame 0 claims the wall is much nearer
    pr = BackgroundProjector(views, device="cpu", erode_px=0).project(1)
    outside = views.id_maps[1] == 0
    assert not pr.seen[outside].any()               # in front of frame 1's own wall


def test_the_floor_band_keeps_away_from_every_mask() -> None:
    views = _views({0: (15, 10)})
    b = band(views, NODE, 0, px=3, gap=2)
    near = np.zeros((H, W), bool)                   # within 2 px of the box, along an axis
    near[8:22, 15:25] = True
    near[10:20, 13:27] = True
    assert b.any() and not (b & near).any()
    assert not b[0:4].any()                         # farther than gap + px


def test_squared_error_pools_pixels() -> None:
    img = np.full((2, 2, 3), 255, np.uint8)
    e = SquaredError()
    e.add(img, np.ones((2, 2, 3)), np.ones((2, 2), bool))
    assert e.psnr() == 60.0
    e.add(img, np.zeros((2, 2, 3)), np.ones((2, 2), bool))
    assert e.psnr() == round(10 * np.log10(2.0), 3)


def test_a_slab_holds_what_lies_from_the_surface_to_the_width_behind_it() -> None:
    views = _views({0: (15, 10)})
    width = entity_width(views, NODE, 0)
    assert np.isclose(width, 9 * (DEPTH - 1.0) / FOCAL * 0.9, rtol=0.25)
    vol = entity_volume(views, NODE)
    assert vol.kind(0) == "slab"
    centre = np.array([(20 - W / 2) / FOCAL, (15 - H / 2) / FOCAL, 1.0])
    pts = np.stack([centre * z for z in (DEPTH - 1.0, DEPTH - 1.0 + 0.5 * width,
                                         DEPTH - 1.0 + 2 * width, DEPTH - 1.0 - width)])
    assert vol.inside(pts, 0).tolist() == [True, True, False, False]
    beside = np.array([(2 - W / 2) / FOCAL, (15 - H / 2) / FOCAL, 1.0]) * (DEPTH - 1.0)
    assert not vol.inside(beside[None], 0)[0]       # not over its pixels
    assert EntityVolume(views, NODE).kind(0) is None


def _sil(x0: int, x1: int, score: float = 0.9) -> Silhouette:
    m = np.zeros((H, W), bool)
    m[5:25, x0:x1] = True
    return Silhouette(m, np.zeros(4), score, "test")


def test_a_trace_needs_a_judged_view_and_a_detection_over_the_entity() -> None:
    hint = _sil(10, 20).mask
    assert entity_region([], hint) is None                       # nothing to judge against
    assert entity_region([_sil(10, 20, score=0.4)], hint) is None  # too unsure before
    region = entity_region([_sil(30, 40), _sil(11, 21)], hint)
    assert region is not None and region[:, 11:21].any() and not region[:, 30:40].any()
    assert has_trace([_sil(12, 20, score=0.35)], region)          # a faint ghost is a trace
    assert not has_trace([_sil(30, 40)], region)                  # a neighbour is not
    assert not has_trace([_sil(12, 20, score=0.2)], region)


def test_the_pass_marks_are_the_owners() -> None:
    assert PASS == {"t1_clean": 0.95, "t2_left": 0.05, "t3_gap_db": 1.0}
    row = {"t1_train": {"clean": 0.96}, "t1_orbit": {"clean": None},
           "t2_left": 0.05, "t3_gap_db": 1.2}
    assert verdict(row) == {"t1": True, "t2": True, "t3": False, "all": False}
    row["t3_gap_db"] = None
    assert verdict(row)["all"] is True

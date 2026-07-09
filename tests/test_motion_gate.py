"""FP2c motion gate: ever-moved (p95) foreground vs static-rerouted-to-background."""
from __future__ import annotations

import torch

from room4d.lifting.native_track_lift import gate_tracks_by_motion
from room4d.lifting.tracks_3d import StaticPoints3D, Tracks3D

T = 40


def _tracks() -> Tracks3D:
    xyz = torch.zeros(4, T, 3)
    xyz[0] = 0.0                                              # 0: fully static -> bg
    xyz[1, :, 0] = torch.linspace(0.0, 2.0, T)               # 1: moves every frame -> fg
    xyz[1, :, 1:] = 1.0
    xyz[2] = torch.tensor([1.0, 1.0, 1.0])                   # 2: static then moves (last 8 frames)
    xyz[2, 32:, 2] = 1.0 + torch.linspace(0.1, 0.8, 8)
    xyz[3] = torch.tensor([2.0, 2.0, 2.0])                   # 3: one-frame glitch -> still static
    xyz[3, 20, 2] = 4.0
    vis = torch.ones(4, T, dtype=torch.bool)
    return Tracks3D(xyz=xyz, visibles=vis, invisibles=torch.zeros(4, T, dtype=torch.bool),
                    confidences=torch.ones(4, T), colors=torch.rand(4, 3))


def _bg(n: int = 5) -> StaticPoints3D:
    return StaticPoints3D(xyz=torch.randn(n, 3), normals=torch.zeros(n, 3), colors=torch.rand(n, 3))


def test_gate_keeps_ever_moved_reroutes_static() -> None:
    tracks, bg = _tracks(), _bg(5)
    fg, bg2 = gate_tracks_by_motion(tracks, bg, percentile=95.0, threshold=0.05)
    # dynamic (1) + static-then-moving (2) stay foreground; fully-static (0) +
    # one-frame-glitch (3) reroute to background.
    assert fg.num_tracks == 2
    assert bg2.num_points == 5 + 2
    assert bg2.normals.shape == bg2.xyz.shape  # rerouted static points carry (zero) normals
    # the foreground survivors are tracks 1 and 2 (the moving ones), in order
    assert torch.allclose(fg.xyz[0], tracks.xyz[1]) and torch.allclose(fg.xyz[1], tracks.xyz[2])


def test_p95_catches_brief_motion_that_median_misses() -> None:
    """The static-then-moving track: median spread says static, p95 says dynamic."""
    z = _tracks().xyz[2]
    spread = (z - z.median(dim=0).values).norm(dim=1)
    assert float(spread.median()) <= 0.05       # a median reducer would mislabel it static
    assert float(torch.quantile(spread, 0.95)) > 0.05  # p95 correctly flags the brief motion


def test_gate_noop_when_all_dynamic() -> None:
    tracks, bg = _tracks(), _bg(3)
    # keep only the two moving tracks -> every track is dynamic -> gate is a no-op
    dyn = tracks.filter_valid(torch.tensor([False, True, True, False]))
    fg, bg2 = gate_tracks_by_motion(dyn, bg, percentile=95.0, threshold=0.05)
    assert fg.num_tracks == 2 and bg2.num_points == 3  # nothing rerouted

"""Tests for the SoM-shaped densification schedule derivation.

Guards the invariant that makes the terminal-opacity-reset collapse impossible:
the opacity reset always lands with recovery + polish after it
(``reset < stop < iters``), for any iteration budget.
"""

from __future__ import annotations

import pytest

from room4d.render.gaussian4d.densify import DensifyConfig, densify_schedule


def test_schedule_reproduces_som_at_5000() -> None:
    # At SoM's max_steps the derived schedule must equal SoM's literal numbers.
    sched = densify_schedule(5000, num_frames=277)
    assert sched == {
        "start_iter": 200,
        "refine_every": 100,
        "reset_opacity_every": 3000,
        "stop_iter": 4000,
        "pause_refine_after_reset": 277,
    }


@pytest.mark.parametrize("max_iters", [500, 1000, 3000, 5000, 7000, 12000])
def test_schedule_invariants(max_iters: int) -> None:
    s = densify_schedule(max_iters, num_frames=100)
    # Reset has recovery; control stops before training does; warmup before reset.
    assert 0 < s["start_iter"] < s["reset_opacity_every"] < s["stop_iter"] < max_iters
    # Windows align to the control cadence.
    for key in ("start_iter", "reset_opacity_every", "stop_iter"):
        assert s[key] % s["refine_every"] == 0


def test_from_cfg_derives_absent_fields() -> None:
    cfg = DensifyConfig.from_cfg({"enabled": True}, max_iters=5000, num_frames=277)
    assert cfg.start_iter == 200
    assert cfg.reset_opacity_every == 3000
    assert cfg.stop_iter == 4000
    assert cfg.pause_refine_after_reset == 277


def test_from_cfg_explicit_overrides_derived() -> None:
    cfg = DensifyConfig.from_cfg(
        {"enabled": True, "stop_iter": 1234, "reset_opacity_every": 800},
        max_iters=5000, num_frames=277,
    )
    assert cfg.stop_iter == 1234            # explicit wins
    assert cfg.reset_opacity_every == 800   # explicit wins
    assert cfg.start_iter == 200            # still derived


def test_from_cfg_rejects_stop_iter_past_end() -> None:
    # The original footgun: a reset/densify window running to the final iteration.
    with pytest.raises(ValueError, match="must be < iters"):
        DensifyConfig.from_cfg(
            {"enabled": True, "stop_iter": 4000, "reset_opacity_every": 3000},
            max_iters=3000, num_frames=277,
        )


def test_from_cfg_without_iters_is_backward_compatible() -> None:
    # No max_iters -> concrete dataclass defaults, no derivation (existing callers).
    cfg = DensifyConfig.from_cfg({"grad2d_threshold": 0.0})
    assert cfg.start_iter == 500 and cfg.stop_iter == 15000
    assert cfg.grad2d_threshold == 0.0

from __future__ import annotations

import pytest

from reifyscene.api.resume import resolve_stage_window
from reifyscene.api.stages import STAGE_ORDER


def _window(start: str, stop: str) -> list[str]:
    start_idx, stop_idx = resolve_stage_window(start, stop)
    return list(STAGE_ORDER)[start_idx:stop_idx + 1]


def test_default_stop_runs_full_pipeline() -> None:
    # Default stop is the final stage, so the window must cover everything.
    assert _window('ingest', STAGE_ORDER[-1]) == list(STAGE_ORDER)


def test_geometry_window_runs_only_through_geometry() -> None:
    assert _window('ingest', 'geometry') == ['ingest', 'sync', 'dyn_mask', 'geometry']


def test_single_stage_window() -> None:
    idx = STAGE_ORDER.index('geometry')
    assert resolve_stage_window('geometry', 'geometry') == (idx, idx)


def test_mid_window_excludes_endpoints_outside_range() -> None:
    assert _window('sync', 'tracking') == ['sync', 'dyn_mask', 'geometry', 'pose_refine', 'segment', 'tracking']


def test_pose_refine_sits_between_geometry_and_tracking() -> None:
    order = list(STAGE_ORDER)
    assert order.index('geometry') < order.index('pose_refine') < order.index('tracking')
    assert order[order.index('geometry') + 1] == 'pose_refine'


def test_stop_before_start_raises() -> None:
    with pytest.raises(ValueError, match="precedes"):
        resolve_stage_window('geometry', 'sync')


def test_unknown_start_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_stage_window('bogus', 'render')


def test_unknown_stop_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_stage_window('ingest', 'bogus')

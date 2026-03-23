"""Basic tests for tracking scaffolding."""

from room4d.tracking.cotracker_wrapper import CoTrackerWrapper


def test_cotracker_wrapper_constructs() -> None:
    tracker = CoTrackerWrapper()
    assert tracker.cfg == {}

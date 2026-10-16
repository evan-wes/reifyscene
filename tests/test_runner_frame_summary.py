from __future__ import annotations

from types import SimpleNamespace

from reifyscene.common.types import FrameRef
from scripts.run_single_video_reifyscene import _build_frame_summary_rows, _sample_rate_hz


def test_frame_summary_rows_tracking_vs_scene():
    geometry = SimpleNamespace(selected_frames=[FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 2, 0.2)])
    tracks = SimpleNamespace(selected_frames=[FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 1, 0.1), FrameRef("cam_00", 2, 0.2)])
    humans = SimpleNamespace(humans=[SimpleNamespace(frame_ref=FrameRef("cam_00", 0, 0.0)), SimpleNamespace(frame_ref=FrameRef("cam_00", 1, 0.1))], metadata={"frame_source": "tracking"})
    objects = SimpleNamespace(objects=[SimpleNamespace(frame_ref=FrameRef("cam_00", 0, 0.0)), SimpleNamespace(frame_ref=FrameRef("cam_00", 1, 0.1))], metadata={"frame_source": "tracking"})
    scene = SimpleNamespace(frame_refs=[FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 1, 0.1), FrameRef("cam_00", 2, 0.2)], timeline={"source": "tracking"})
    render = SimpleNamespace(path_index={"orbit": ["a", "b", "c"], "replay": ["d", "e", "f"]})
    outputs = {"geometry": geometry, "tracks": tracks, "humans": humans, "objects": objects, "scene": scene, "render": render}
    cfg = {"humans": {"frame_source": "tracking"}, "objects": {"frame_source": "tracking"}, "scene": {"timeline_source": "geometry"}}

    rows = _build_frame_summary_rows(outputs, cfg)
    by_stage = {row["stage"]: row for row in rows}
    assert by_stage["Tracking"]["count"] == "3"
    assert by_stage["Scene"]["source"] == "tracking"
    assert by_stage["Render"]["count"] == "3"


def test_sample_rate_is_a_frame_step_when_the_source_records_no_frame_rate():
    """DAVIS frames have no time: the summary says how far apart they are in frames."""
    assert _sample_rate_hz([FrameRef("cam_00", i) for i in (0, 2, 4)]) == "every 2 frames"
    assert _sample_rate_hz([FrameRef("cam_00", i, i / 30) for i in (0, 1, 2)]) == "30.00 Hz"

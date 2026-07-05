"""Tests for the dynamic-scene enablement fixes (surfaced by the G1-single stress test).

Two latent bugs kept every runner-driven render `kind:static`: (1) the scene stage's
registry `run_fn` never passed the segment `instance_artifact`, so the dynamic path got
zero foreground; (2) `_fg_mask_by_frame` read `frame_index`, which is only built at
save-time, so an in-memory (pre-save) segment artifact yielded no foreground.
"""

from __future__ import annotations

import numpy as np

from room4d.api.stages import build_stage_registry
from room4d.common.types import FrameRef
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.scene.assembler import SingleVideoSceneAssembler


def test_scene_stage_wires_segment_instance_artifact():
    # The scene stage must depend on `segment` so the runner materializes it and the
    # run_fn can pass it (else the dynamic path silently gets no foreground).
    scene_spec = build_stage_registry()["scene"]
    assert "segment" in scene_spec.dependencies


def test_fg_mask_rebuilds_from_observations_when_frame_index_empty():
    m0 = np.zeros((4, 4), dtype=bool); m0[1:3, 1:3] = True   # 4 fg px
    m1 = np.zeros((4, 4), dtype=bool); m1[0, :] = True        # 4 fg px, frame 5
    ia = InstanceArtifact(
        run_id="t", stage_name="segment", save_dir="/tmp/x",
        observations=[
            InstanceObservation(instance_id="i0", track_id="p", frame_ref=FrameRef("0", 0, 0.0)),
            InstanceObservation(instance_id="i1", track_id="p", frame_ref=FrameRef("0", 5, 0.16)),
        ],
        mask_arrays={"i0": m0, "i1": m1},
        frame_index={},   # in-memory (pre-save) state: index not yet built
    )
    fg = SingleVideoSceneAssembler()._fg_mask_by_frame(ia)
    assert set(fg) == {"0:0", "0:5"}
    assert int(fg["0:0"].sum()) == 4 and int(fg["0:5"].sum()) == 4


def test_fg_mask_unions_multiple_instances_per_frame():
    a = np.zeros((4, 4), dtype=bool); a[0, 0] = True
    b = np.zeros((4, 4), dtype=bool); b[3, 3] = True
    ia = InstanceArtifact(
        run_id="t", stage_name="segment", save_dir="/tmp/x",
        observations=[
            InstanceObservation(instance_id="i0", track_id="p", frame_ref=FrameRef("0", 0, 0.0)),
            InstanceObservation(instance_id="i1", track_id="q", frame_ref=FrameRef("0", 0, 0.0)),
        ],
        mask_arrays={"i0": a, "i1": b},
        frame_index={},
    )
    fg = SingleVideoSceneAssembler()._fg_mask_by_frame(ia)
    assert int(fg["0:0"].sum()) == 2   # union of both instances

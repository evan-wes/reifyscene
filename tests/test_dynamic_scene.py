"""Tests for the dynamic-scene enablement fixes (surfaced by the G1-single stress test).

Two latent bugs kept every runner-driven render `kind:static`: (1) the scene stage's
registry `run_fn` never passed the segment `instance_artifact`, so the dynamic path got
zero foreground; (2) `_fg_mask_by_frame` read `frame_index`, which is only built at
save-time, so an in-memory (pre-save) segment artifact yielded no foreground.
"""

from __future__ import annotations

import numpy as np

from reifyscene.api.stages import build_stage_registry
from reifyscene.common.types import FrameRef
from reifyscene.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from reifyscene.scene.assembler import SingleVideoSceneAssembler


def test_scene_stage_wires_segment_instance_artifact():
    # The scene stage must depend on `segment` so the runner materializes it and the
    # run_fn can pass it (else the dynamic path silently gets no foreground).
    scene_spec = build_stage_registry()["scene"]
    assert "segment" in scene_spec.dependencies


def test_fg_mask_rebuilds_from_observations_when_frame_index_empty():
    m0 = np.zeros((4, 4), dtype=bool); m0[1:3, 1:3] = True   # 4 fg px
    m1 = np.zeros((4, 4), dtype=bool); m1[0, :] = True        # 4 fg px, frame 5
    ia = InstanceArtifact(
        run_id="t", stage_name="segment", save_dir="unsaved",
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


def test_fold_ground_plane_into_norm():
    import torch
    from reifyscene.geometry.geometry_artifact import GeometryArtifact
    # a non-trivial ground-plane transform (45° about x + a translation)
    c, s = float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4))
    t_gp = np.array([[1, 0, 0, 0], [0, c, -s, 0.5], [0, s, c, 0], [0, 0, 0, 1]], np.float32)
    ga = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="unsaved",
                          metadata={"world_frame": {"transform_world_old_to_new": t_gp.tolist()}})
    # identity scene_norm ∘ T_gp == T_gp (so a raw val camera gets the Z-up rotation)
    out = SingleVideoSceneAssembler._fold_ground_plane_into_norm(torch.eye(4), ga)
    assert np.allclose(out.numpy(), t_gp)
    # no world_frame (no ground plane ran) → no-op
    ga2 = GeometryArtifact(run_id="t", stage_name="geometry", save_dir="unsaved", metadata={})
    assert torch.allclose(SingleVideoSceneAssembler._fold_ground_plane_into_norm(torch.eye(4), ga2), torch.eye(4))


def test_fg_mask_unions_multiple_instances_per_frame():
    a = np.zeros((4, 4), dtype=bool); a[0, 0] = True
    b = np.zeros((4, 4), dtype=bool); b[3, 3] = True
    ia = InstanceArtifact(
        run_id="t", stage_name="segment", save_dir="unsaved",
        observations=[
            InstanceObservation(instance_id="i0", track_id="p", frame_ref=FrameRef("0", 0, 0.0)),
            InstanceObservation(instance_id="i1", track_id="q", frame_ref=FrameRef("0", 0, 0.0)),
        ],
        mask_arrays={"i0": a, "i1": b},
        frame_index={},
    )
    fg = SingleVideoSceneAssembler()._fg_mask_by_frame(ia)
    assert int(fg["0:0"].sum()) == 2   # union of both instances

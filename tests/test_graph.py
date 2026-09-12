"""The graph holds real entities; point tracks are the evidence for them.

Synthetic masks, tracks and instances: a person on the left, a mug on the right, and
tracks placed so each outcome -- supporting a person, supporting an object, supporting
the room -- is checked on the nodes the builder emits.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from reifyscene.common.types import FrameRef, HumanState, ObjectState
from reifyscene.graph.builders import DynamicSceneGraphBuilder
from reifyscene.graph.nodes import NodeFactory

HEIGHT, WIDTH = 40, 60
FRAMES = [0, 1, 2, 3]


def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


PERSON_BOX = box(5, 5, 20, 30)
MUG_BOX = box(35, 5, 55, 30)


def geometry(stream_id: str = "cam0"):
    """The little of the geometry artifact the builder reads."""
    return SimpleNamespace(
        run_id="run",
        stream_ids=lambda: [stream_id],
        camera_states_by_stream={stream_id: [object(), object()]},
        intrinsics_by_stream={stream_id: object()},
    )


def humans(stream_id: str = "cam0"):
    """One person, masked on every frame, with the segment masks the stage copied."""
    records = [
        HumanState(
            human_id="p1",
            frame_ref=FrameRef(stream_id=stream_id, frame_idx=f, timestamp_sec=float(f)),
            bbox_xyxy=np.asarray([5, 5, 20, 30], dtype=np.float32),
            mask_rle=None,
            keypoints_2d=None,
            densepose=None,
            smpl_params=None,
            confidence=0.9,
            attributes={"detection_id": f"inst:{stream_id}:{f}:0"},
        )
        for f in FRAMES
    ]
    masks = {f"inst:{stream_id}:{f}:0": PERSON_BOX for f in FRAMES}
    return SimpleNamespace(humans=records, mask_arrays=masks)


def objects(stream_id: str = "cam0"):
    """One mug, masked on every frame."""
    records = [
        ObjectState(
            object_id="o1",
            frame_ref=FrameRef(stream_id=stream_id, frame_idx=f, timestamp_sec=float(f)),
            category="mug",
            bbox_xyxy=np.asarray([35, 5, 55, 30], dtype=np.float32),
            mask_rle=None,
            embedding=None,
            confidence=0.8,
            attributes={"detection_id": f"inst:{stream_id}:{f}:1"},
        )
        for f in FRAMES
    ]
    masks = {f"inst:{stream_id}:{f}:1": MUG_BOX for f in FRAMES}
    return SimpleNamespace(objects=records, mask_arrays=masks)


def tracking(paths, *, static_labels=None, lifted=None, stream_id: str = "cam0"):
    """A track payload shaped like the tracking stage's: ingest pixels, one row per frame."""
    xy = np.asarray(paths, dtype=np.float32).transpose(1, 0, 2)
    num = xy.shape[1]
    payload = {
        "frame_indices": np.asarray(FRAMES, dtype=np.int32),
        "points_xy": xy,
        "visibility": np.ones((len(FRAMES), num), dtype=bool),
        "track_ids": np.arange(num, dtype=np.int32),
        "static_labels": np.asarray(
            static_labels if static_labels is not None else [-1] * num, dtype=np.int8
        ),
    }
    if lifted is not None:
        payload["lifted_frame_indices"] = np.asarray(lifted["frames"], dtype=np.int32)
        payload["lifted_xyz"] = np.asarray(lifted["xyz"], dtype=np.float32)
        payload["lifted_valid"] = np.asarray(lifted["valid"], dtype=bool)
    return SimpleNamespace(
        track_payloads_by_stream={stream_id: payload},
        stream_ids=lambda: [stream_id],
    )


def build(track_artifact, human_artifact=None, object_artifact=None, cfg=None):
    builder = DynamicSceneGraphBuilder(cfg)
    return builder.build(geometry(), track_artifact, human_artifact, object_artifact)


def nodes_by_id(artifact):
    return {node.node_id: node for node in artifact.nodes}


def test_node_factory_camera() -> None:
    node = NodeFactory().camera_node("cam0", {})
    assert node.node_id == "cam0"


def test_the_graph_holds_entities_not_tracks() -> None:
    inside_person = [(10.0, 10.0)] * 4
    in_the_room = [(28.0, 35.0)] * 4
    artifact = build(tracking([inside_person, in_the_room]), humans(), objects())

    ids = set(nodes_by_id(artifact))
    assert ids == {"cam0", "human:cam0:p1", "object:cam0:o1", "background:cam0"}
    assert not [n for n in artifact.nodes if n.node_id.startswith("track:")]
    assert artifact.source_summary["num_entities"] == 2


def test_a_person_carries_the_tracks_inside_its_mask() -> None:
    inside_person = [(10.0, 10.0)] * 4
    inside_mug = [(40.0, 10.0)] * 4
    artifact = build(tracking([inside_person, inside_mug]), humans(), objects())

    person = nodes_by_id(artifact)["human:cam0:p1"]
    mug = nodes_by_id(artifact)["object:cam0:o1"]
    assert person.attrs["support_track_ids"] == [0]
    assert person.attrs["support_frames"] == {"0": 4}
    assert person.attrs["num_support_tracks"] == 1
    assert person.attrs["mean_support_ratio"] == 1.0
    assert mug.attrs["support_track_ids"] == [1]
    assert mug.label == "mug"


def test_tracks_inside_nothing_are_the_background_s_evidence() -> None:
    inside_person = [(10.0, 10.0)] * 4
    in_the_room = [(28.0, 35.0)] * 4
    elsewhere = [(2.0, 38.0)] * 4
    artifact = build(tracking([inside_person, in_the_room, elsewhere]), humans(), objects())

    background = nodes_by_id(artifact)["background:cam0"]
    assert background.node_type == "region"
    assert background.label == "background"
    assert background.attrs["support_track_ids"] == [1, 2]
    assert background.attrs["num_support_tracks"] == 2
    assert background.attrs["ambiguous_track_ids"] == []


def test_a_track_inside_a_person_too_rarely_is_named_ambiguous() -> None:
    # Inside the person on one frame of four: not its evidence, and not the room's either.
    wandering = [(10.0, 10.0), (28.0, 35.0), (28.0, 35.0), (28.0, 35.0)]
    artifact = build(tracking([wandering]), humans(), objects())

    background = nodes_by_id(artifact)["background:cam0"]
    person = nodes_by_id(artifact)["human:cam0:p1"]
    assert person.attrs["support_track_ids"] == []
    assert background.attrs["support_track_ids"] == []
    assert background.attrs["ambiguous_track_ids"] == [0]
    assert background.attrs["num_ambiguous_tracks"] == 1


def test_an_entity_knows_where_it_was_from_its_own_tracks() -> None:
    inside_person = [(10.0, 10.0)] * 4
    in_the_room = [(28.0, 35.0)] * 4
    lifted = {
        "frames": [0, 2],
        # The person's track (column 0) moves; the room's (column 1) does not.
        "xyz": [[[1.0, 0.0, 5.0], [9.0, 9.0, 9.0]], [[3.0, 0.0, 5.0], [9.0, 9.0, 9.0]]],
        "valid": [[True, True], [True, True]],
    }
    artifact = build(tracking([inside_person, in_the_room], lifted=lifted), humans(), objects())

    positions = nodes_by_id(artifact)["human:cam0:p1"].attrs["positions"]
    assert [p["frame_idx"] for p in positions] == [0, 2]
    assert positions[0]["xyz"] == [1.0, 0.0, 5.0]
    assert positions[1]["xyz"] == [3.0, 0.0, 5.0]
    assert positions[0]["num_points"] == 1


def test_an_entity_reports_whether_its_evidence_moved() -> None:
    inside_person = [(10.0, 10.0)] * 4
    also_inside_person = [(12.0, 12.0)] * 4
    in_the_room = [(28.0, 35.0)] * 4
    artifact = build(
        tracking([inside_person, also_inside_person, in_the_room], static_labels=[0, 1, 1]),
        humans(),
        objects(),
    )

    motion = nodes_by_id(artifact)["human:cam0:p1"].attrs["support_motion"]
    assert motion == {"static": 1, "moving": 1, "uncertain": 0, "static_fraction": 0.5}
    assert nodes_by_id(artifact)["background:cam0"].attrs["support_motion"]["static"] == 1


def test_the_thresholds_come_from_the_config() -> None:
    # Inside the person on two of four frames: accepted at 0.5, refused at 0.75.
    half_inside = [(10.0, 10.0), (10.0, 10.0), (28.0, 35.0), (28.0, 35.0)]
    loose = build(tracking([half_inside]), humans(), objects())
    strict = build(
        tracking([half_inside]),
        humans(),
        objects(),
        cfg={"track_membership_min_ratio": 0.75, "track_membership_min_frames": 2},
    )
    assert nodes_by_id(loose)["human:cam0:p1"].attrs["support_track_ids"] == [0]
    assert nodes_by_id(strict)["human:cam0:p1"].attrs["support_track_ids"] == []


def test_entities_without_masks_carry_no_evidence_and_the_notes_say_so() -> None:
    maskless = SimpleNamespace(humans=humans().humans, mask_arrays={})
    artifact = build(tracking([[(10.0, 10.0)] * 4]), maskless, None)

    person = nodes_by_id(artifact)["human:cam0:p1"]
    assert person.attrs["support_track_ids"] == []
    assert person.attrs["num_support_tracks"] == 0
    assert any("no entity masks" in note for note in artifact.notes)


def test_tracked_frames_that_share_nothing_with_the_masks_are_reported() -> None:
    artifact = build(tracking([[(10.0, 10.0)] * 4]), humans(), objects())
    # The mask frames and the tracking frames are the same here, so the note is the
    # count, not the warning.
    assert any("shared frames" in note for note in artifact.notes)
    assert not any("do not overlap" in note for note in artifact.notes)


def test_every_entity_is_observed_by_its_camera() -> None:
    artifact = build(tracking([[(10.0, 10.0)] * 4]), humans(), objects())
    observed = {(e.src, e.dst) for e in artifact.edges if e.edge_type == "observed_by"}
    assert observed == {
        ("human:cam0:p1", "cam0"),
        ("object:cam0:o1", "cam0"),
        ("background:cam0", "cam0"),
    }

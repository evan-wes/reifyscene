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


# The noise floor (entity_min_area / entity_min_support_tracks) has tests of its own; the rest
# build these few-track scenes without it.
NO_FLOOR = {"entity_min_area": 0.0, "entity_min_support_tracks": 0}


def build(track_artifact, human_artifact=None, object_artifact=None, cfg=None,
          semantics_artifact=None):
    builder = DynamicSceneGraphBuilder({**NO_FLOOR, **(cfg or {})})
    return builder.build(geometry(), track_artifact, human_artifact, object_artifact,
                         semantics_artifact=semantics_artifact)


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
    """A genuinely ambiguous track: half its inside-frames on the person, half on the mug."""
    ambiguous = [(10.0, 10.0), (10.0, 10.0), (45.0, 15.0), (45.0, 15.0)]
    loose = build(tracking([ambiguous]), humans(), objects())
    strict = build(
        tracking([ambiguous]),
        humans(),
        objects(),
        cfg={"track_membership_min_ratio": 0.75, "track_membership_min_frames": 2},
    )
    # At 0.5 the tie goes to the person (equal counts, equal ratio, smaller id).
    assert nodes_by_id(loose)["human:cam0:p1"].attrs["support_track_ids"] == [0]
    # At 0.75 neither entity owns it decisively enough, so it supports neither.
    assert nodes_by_id(strict)["human:cam0:p1"].attrs["support_track_ids"] == []
    assert nodes_by_id(strict)["object:cam0:o1"].attrs["support_track_ids"] == []


def test_a_track_inside_one_entity_only_is_its_evidence_however_briefly() -> None:
    """The ratio asks *which* entity, not how long -- min_frames asks how long.

    A point on the person for two frames of four, and inside nothing on the others, is
    unambiguously the person's. Under the old rule its denominator was the person's whole
    co-visible life, so it scored 0.5 and only survived a lenient threshold; a longer
    clip would have thrown it away entirely.
    """
    brief = [(10.0, 10.0), (10.0, 10.0), (28.0, 35.0), (28.0, 35.0)]  # (28,35) is in neither box
    strict = build(
        tracking([brief]),
        humans(),
        objects(),
        cfg={"track_membership_min_ratio": 0.9, "track_membership_min_frames": 2},
    )
    person = nodes_by_id(strict)["human:cam0:p1"]
    assert person.attrs["support_track_ids"] == [0]
    assert person.attrs["mean_support_ratio"] == 1.0      # decisive: inside nothing else
    assert person.attrs["mean_covisible_ratio"] == 0.5    # and honest about its coverage


def test_dwell_is_still_required() -> None:
    """Unambiguous is not the same as sufficient: one frame inside is still one frame."""
    glancing = [(10.0, 10.0), (28.0, 35.0), (28.0, 35.0), (28.0, 35.0)]
    artifact = build(
        tracking([glancing]), humans(), objects(),
        cfg={"track_membership_min_ratio": 0.5, "track_membership_min_frames": 2},
    )
    assert nodes_by_id(artifact)["human:cam0:p1"].attrs["support_track_ids"] == []


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


# --- what the semantics stage decided reaches the node -----------------------

def semantics(records=None, embeddings=None):
    """The little of the SemanticsArtifact the builder reads, keyed by canonical track."""
    return SimpleNamespace(
        records=records if records is not None else {},
        embeddings=embeddings if embeddings is not None else {},
    )


def _record(category="mug", score=0.4, topk=None, **kw):
    """A semantics record, with every field a real SemanticRecord carries."""
    base = dict(
        category=category, score=score,
        category_topk=topk if topk is not None else [("mug", 0.4), ("cup", 0.1)],
        caption=None, margin=None, margin_z=None, abstained=False, abstain_reason=None,
        background=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_node_carries_the_category_its_alternatives_and_its_embedding() -> None:
    recs = {"p1": _record("person", 0.5), "o1": _record("mug", 0.4)}
    embs = {"p1": np.arange(4, dtype=np.float32)}
    artifact = build(
        tracking([[(10.0, 10.0)] * 4]), humans(), objects(),
        semantics_artifact=semantics(recs, embs),
    )
    person = nodes_by_id(artifact)["human:cam0:p1"]
    assert person.attrs["category"] == "person"
    assert person.attrs["category_topk"] == [["mug", 0.4], ["cup", 0.1]]
    assert person.attrs["semantics_source"] == "semantics"
    assert artifact.embeddings["human:cam0:p1"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert "object:cam0:o1" not in artifact.embeddings  # no vector, no entry
    assert any("semantics named 2 of 2" in n for n in artifact.notes)


def test_a_human_node_is_labelled_by_semantics_not_hardcoded() -> None:
    artifact = build(
        tracking([[(10.0, 10.0)] * 4]), humans(), objects(),
        semantics_artifact=semantics({"p1": _record("cyclist", 0.6)}),
    )
    assert nodes_by_id(artifact)["human:cam0:p1"].label == "cyclist"


def test_an_abstention_leaves_the_node_labelled_human_and_says_why() -> None:
    """A refusal is a value: the node falls back, and records that it was asked."""
    refused = _record("unknown", 0.1, abstained=True, abstain_reason="low_separation",
                      margin=0.004, margin_z=0.8)
    artifact = build(
        tracking([[(10.0, 10.0)] * 4]), humans(), objects(),
        semantics_artifact=semantics({"p1": refused}),
    )
    person = nodes_by_id(artifact)["human:cam0:p1"]
    assert person.label == "human"                       # not "unknown"
    assert person.attrs["category"] == "unknown"
    assert person.attrs["abstained"] is True
    assert person.attrs["abstain_reason"] == "low_separation"
    assert person.attrs["category_margin"] == 0.004




def test_without_a_semantics_artifact_the_graph_still_builds() -> None:
    artifact = build(tracking([[(10.0, 10.0)] * 4]), humans(), objects())
    person = nodes_by_id(artifact)["human:cam0:p1"]
    mug = nodes_by_id(artifact)["object:cam0:o1"]
    assert person.label == "human"
    assert person.attrs["semantics_source"] == "none"
    assert mug.attrs["semantics_source"] == "objects"    # category came down via objects
    assert mug.label == "mug"
    assert artifact.embeddings == {}


# --- the graph says whether its vertical relations are computable -------------

def test_the_graph_records_whether_its_positions_are_gravity_aligned() -> None:
    """A missing `above` edge means *false* or *not computable*; a consumer must be able
    to tell, and the answer lives two stages upstream."""
    geo = geometry()
    geo.metadata = {
        "ground_plane_applied": True,
        "world_frame": {"type": "ground_plane_z_up", "inlier_ratio": 0.45},
    }
    builder = DynamicSceneGraphBuilder()
    artifact = builder.build(geo, tracking([[(10.0, 10.0)] * 4]), humans(), objects())
    frame = artifact.metadata["world_frame"]
    assert frame["gravity_aligned"] is True
    assert frame["up_axis"] == "+z"
    assert frame["inlier_ratio"] == 0.45


def test_a_canonicalizer_that_did_not_run_is_recorded_as_such() -> None:
    geo = geometry()
    geo.metadata = {"ground_plane_applied": False, "world_frame": {}}
    artifact = DynamicSceneGraphBuilder().build(
        geo, tracking([[(10.0, 10.0)] * 4]), humans(), objects()
    )
    frame = artifact.metadata["world_frame"]
    assert frame["gravity_aligned"] is False
    assert frame["up_axis"] is None


# --- not every candidate is a thing -------------------------------------------

def test_a_candidate_named_a_background_class_goes_to_the_background() -> None:
    recs = {"p1": _record("grass", background=True), "o1": _record("grass", background=True)}
    artifact = build(tracking([[(10.0, 10.0)] * 4]), humans(), objects(),
                     semantics_artifact=semantics(recs))
    ids = nodes_by_id(artifact)
    assert "object:cam0:o1" not in ids                  # the backdrop is the background's
    assert "human:cam0:p1" in ids                       # a person never is
    assert ids["background:cam0"].attrs["absorbed_tracks"] == ["o1"]
    assert artifact.metadata["candidates"]["absorbed"] == [
        {"node_id": "object:cam0:o1", "track_id": "o1", "category": "grass"}]


def test_a_candidate_under_the_noise_floor_is_not_an_entity() -> None:
    """By default a thing needs 5 point tracks inside it, and a mask over 0.1% of the frame."""
    in_person = [[(8.0 + i, 10.0)] * 4 for i in range(5)]
    in_mug = [[(40.0, 10.0)] * 4]
    artifact = DynamicSceneGraphBuilder().build(geometry(), tracking(in_person + in_mug),
                                                humans(), objects())
    ids = nodes_by_id(artifact)
    assert "human:cam0:p1" in ids and "object:cam0:o1" not in ids
    (gone,) = artifact.metadata["candidates"]["dropped"]
    assert gone["node_id"] == "object:cam0:o1" and "1 support track(s), under 5" in gone["reason"]

    speck = objects()
    speck.mask_arrays = {k: box(40, 10, 41, 11) for k in speck.mask_arrays}   # 1 px of 2,400
    artifact = DynamicSceneGraphBuilder({"entity_min_support_tracks": 0}).build(
        geometry(), tracking(in_person + in_mug), humans(), speck)
    (gone,) = artifact.metadata["candidates"]["dropped"]
    assert gone["node_id"] == "object:cam0:o1" and "of the frame (median)" in gone["reason"]

"""Per-frame id maps (reifyscene/graph/id_maps.py) -- the identity loss's input contract.

What these pin is what the loss would learn silently wrong: a contested pixel given to one
entity, a dropped tracklet read as background (it is unaddressed: the scene stage owns what is
seeded there by a reserved owner), a padded pixel treated as a real one, or class numbers that
shift between exports of the same graph.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reifyscene.api.run_outputs import RunOutputs
from reifyscene.geometry.image_transform import ImageTransform, Pad
from reifyscene.graph import id_maps
from reifyscene.graph.id_maps import (
    BACKGROUND,
    IGNORE,
    UNADDRESSED,
    compose_id_map,
    export_id_maps,
    has_id_maps,
    labelled,
    load_id_maps,
    masks_from_id_maps,
    round_trip_mismatches,
)
from reifyscene.graph.renderable_handle import RenderPixelMasks
from tests.conftest import write_ingest

H, W = 4, 6
A, B = "object:cam_00:sam2:cam_00:0", "human:cam_00:sam2:cam_00:1"


def _mask(*cells: tuple[int, int], shape=(H, W)) -> np.ndarray:
    m = np.zeros(shape, bool)
    for r, c in cells:
        m[r, c] = True
    return m


def test_single_entity_pixels_get_its_class_and_the_rest_is_background() -> None:
    out, counts = compose_id_map({A: _mask((0, 0), (1, 1))}, {A: 1}, shape=(H, W))
    assert out.dtype == np.uint16
    assert out[0, 0] == 1 and out[1, 1] == 1
    assert (out == BACKGROUND).sum() == H * W - 2
    assert counts == {"overlap": 0, "unmapped": 0, "padding": 0}


def test_a_pixel_two_entities_claim_is_ignored_not_given_to_either() -> None:
    out, counts = compose_id_map(
        {A: _mask((0, 0), (0, 1)), B: _mask((0, 1), (0, 2))}, {A: 1, B: 2}, shape=(H, W)
    )
    assert out[0, 0] == 1 and out[0, 2] == 2
    assert out[0, 1] == IGNORE
    assert counts["overlap"] == 1


def test_a_dropped_tracklet_is_unaddressed_not_background() -> None:
    """Background would teach the field that the thing the graph dropped is scenery."""
    out, counts = compose_id_map(
        {A: _mask((0, 0))}, {A: 1}, shape=(H, W), unmapped=_mask((2, 2), (0, 0))
    )
    assert out[2, 2] == UNADDRESSED
    assert out[0, 0] == IGNORE           # contested by the unmapped tracklet
    assert counts == {"overlap": 1, "unmapped": 1, "padding": 0}
    assert not labelled(out[2, 2]) and not labelled(out[0, 0]) and labelled(out[1, 1])


def test_padding_is_ignored() -> None:
    valid = np.ones((H, W), bool)
    valid[:, -1] = False
    out, counts = compose_id_map({}, {}, shape=(H, W), valid=valid)
    assert (out[:, -1] == IGNORE).all() and (out[:, :-1] == BACKGROUND).all()
    assert counts["padding"] == H


def test_a_mask_in_the_wrong_raster_is_refused() -> None:
    with pytest.raises(ValueError, match="raster"):
        compose_id_map({A: np.zeros((H + 1, W), bool)}, {A: 1}, shape=(H, W))


# -- export, against a fake run ---------------------------------------------------------------

NUM_FRAMES = 4


def _fake_run(tmp_path, monkeypatch, *, pad: bool = False):
    """A run whose ingest says four frames, and whose masks are given directly.

    Frame 0: A alone. Frame 1: A and B overlapping in one pixel, plus an unmapped tracklet.
    Frame 2: nothing at all. Frame 3: B alone -- A has left the clip.
    """
    run = tmp_path / "run_x"
    write_ingest(run, num_frames=NUM_FRAMES, width=W, height=H)
    if pad:
        transform = ImageTransform((W - 1, H), (Pad(size=(W, H), offset=(0, 0)),))
    else:
        transform = ImageTransform.identity((W, H))
    placed = RenderPixelMasks(
        by_frame={
            0: {A: _mask((0, 0), (0, 1))},
            1: {A: _mask((1, 0), (1, 1)), B: _mask((1, 1), (1, 2))},
            3: {B: _mask((3, 3))},
        },
        labels={A: "dog", B: "person"},
        notes=["1 tracklet(s) have no graph node and were skipped"],
        unmapped={1: _mask((2, 4))},
        transform=transform,
    )
    import reifyscene.graph.renderable_handle as rh

    monkeypatch.setattr(rh, "render_pixel_masks", lambda run, stream_id="cam_00": placed)
    return RunOutputs.open(run), placed


def test_export_writes_a_map_for_every_frame_including_empty_ones(tmp_path, monkeypatch) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch)
    manifest = export_id_maps(run)
    files = sorted(p.name for p in (run.identity_dir).glob("frame_*.png"))
    assert files == [f"frame_{i:05d}.png" for i in range(NUM_FRAMES)]
    _, maps = load_id_maps(run)
    assert (maps[2] == BACKGROUND).all()
    assert manifest["raster"] == {"width": W, "height": H}


def test_classes_are_numbered_by_sorted_node_id_and_background_is_zero(
    tmp_path, monkeypatch
) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch)
    manifest = export_id_maps(run)
    by_index = {c["index"]: c for c in manifest["classes"]}
    assert by_index[0]["node_id"] is None
    assert [by_index[i]["node_id"] for i in (1, 2)] == sorted([A, B])
    assert manifest["num_classes"] == 3
    assert manifest["encoding"] == {"dtype": "uint16", "background": 0, "unaddressed": 65534,
                                    "ignore": 65535}


def test_the_manifest_records_when_each_entity_is_in_the_clip(tmp_path, monkeypatch) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch)
    manifest = export_id_maps(run)
    spans = {c["node_id"]: (c["first_frame"], c["last_frame"], c["frames_present"])
             for c in manifest["classes"] if c["node_id"]}
    assert spans[A] == (0, 1, 2)
    assert spans[B] == (1, 3, 2)


def test_ignore_reasons_are_accounted_for(tmp_path, monkeypatch) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch)
    f = export_id_maps(run)["fractions"]
    total = NUM_FRAMES * H * W
    assert f["overlap"] == round(1 / total, 6)
    assert f["unmapped"] == round(1 / total, 6)
    assert f["padding"] == 0.0
    assert f["ignore"] == round(1 / total, 6)          # unaddressed pixels are not ignore


def test_padding_from_the_recorded_transform_is_ignored(tmp_path, monkeypatch) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch, pad=True)
    export_id_maps(run)
    _, maps = load_id_maps(run)
    assert all((m[:, -1] == IGNORE).all() for m in maps.values())


def test_round_trip_reproduces_the_source_masks_off_ignore(tmp_path, monkeypatch) -> None:
    run, placed = _fake_run(tmp_path, monkeypatch)
    export_id_maps(run)
    manifest, maps = load_id_maps(run)
    assert round_trip_mismatches(manifest, maps, placed.by_frame) == {}
    decoded = masks_from_id_maps(manifest, maps)
    assert set(decoded) == {0, 1, 3}
    assert decoded[1][A].sum() == 1 and decoded[1][B].sum() == 1   # the shared pixel went


def test_round_trip_catches_a_corrupted_map(tmp_path, monkeypatch) -> None:
    run, placed = _fake_run(tmp_path, monkeypatch)
    export_id_maps(run)
    manifest, maps = load_id_maps(run)
    maps[0] = maps[0].copy()
    maps[0][0, 0] = BACKGROUND
    assert round_trip_mismatches(manifest, maps, placed.by_frame) == {A: 1}


def test_an_unknown_format_version_is_refused_and_written_again(tmp_path, monkeypatch) -> None:
    run, _ = _fake_run(tmp_path, monkeypatch)
    export_id_maps(run)
    assert has_id_maps(run)
    path = run.identity_dir / id_maps.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["format_version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format"):
        load_id_maps(run)
    assert not has_id_maps(run)          # a cache of the graph: callers export it again


def test_a_track_the_background_absorbed_is_background_not_unaddressed() -> None:
    """``pixel_masks``: the graph's background node absorbed the lawn; its pixels are
    background, while a tracklet with no node at all stays unaddressed."""
    from types import SimpleNamespace

    from reifyscene.common.types import FrameRef
    from reifyscene.graph.renderable_handle import pixel_masks

    lawn, speck, person = np.zeros((4, 6), bool), np.zeros((4, 6), bool), np.zeros((4, 6), bool)
    lawn[3, :], speck[0, 5], person[1:3, 1:3] = True, True, True
    obs = [SimpleNamespace(frame_ref=FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=None),
                           track_id=t, instance_id=f"inst:{t}") for t in ("t0", "t1", "t2")]
    segment = SimpleNamespace(observations=obs, mask_arrays={"inst:t0": person,
                                                             "inst:t1": lawn, "inst:t2": speck})
    nodes = [{"node_id": "human:cam_00:t0", "node_type": "human", "label": "person",
              "attrs": {"stream_id": "cam_00", "track_id": "t0"}},
             {"node_id": "background:cam_00", "node_type": "region", "label": "background",
              "attrs": {"stream_id": "cam_00", "absorbed_tracks": ["t1"]}}]
    stream = SimpleNamespace(id="cam_00", transform=ImageTransform.identity((6, 4)))
    placed = pixel_masks(nodes, {}, segment, stream)
    assert set(placed.by_frame[0]) == {"human:cam_00:t0"}
    assert placed.unmapped[0].sum() == 1                    # the speck, not the lawn
    out, counts = compose_id_map(placed.by_frame[0], {"human:cam_00:t0": 1}, shape=(4, 6),
                                 unmapped=placed.unmapped[0])
    assert (out[3] == BACKGROUND).all() and out[0, 5] == UNADDRESSED and counts["unmapped"] == 1

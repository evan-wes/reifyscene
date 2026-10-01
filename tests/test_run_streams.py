"""A run's streams, asked of the run (reifyscene/ingest/streams.py).

What these pin: the stream id comes from the run -- ``cam_00`` for a video, ``0`` for a DyCheck
frames folder -- and nothing downstream assumes it; with several streams and none named, the
answer is an error listing them, not a guess; frame keys round-trip; a frame is read only where
ingest recorded it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reifyscene.api.run_outputs import NotARun, RunOutputs
from reifyscene.common.frame_keys import parse_key
from reifyscene.geometry.image_transform import ImageTransform, Resize
from reifyscene.ingest.streams import RunStreams, StreamError
from tests.conftest import write_manifest


def _run(tmp_path, streams: list[dict], transforms: dict | None = None):
    """A run whose ingest holds ``streams`` (VideoStreamRef fields; size/fps default)."""
    from reifyscene.common.types import VideoStreamRef
    from reifyscene.ingest.ingest_artifact import IngestArtifact

    run = tmp_path / "run"
    refs = [VideoStreamRef(**{"video_path": Path(s.get("video_path") or run / "clip"),
                              "fps": 30.0, "width": 8, "height": 6, "has_audio": False,
                              "metadata": {}, **{k: v for k, v in s.items() if k != "video_path"}})
            for s in streams]
    write_manifest(run)
    IngestArtifact(run_id="run", stage_name="ingest", save_dir=run / "ingest", streams=refs).save()
    if transforms is not None:
        (run / "geometry").mkdir()
        (run / "geometry" / "artifact.json").write_text(json.dumps(
            {"metadata": {"image_transforms": transforms}}))
    return run


def _frames_dir(tmp_path, n=3, w=8, h=6):
    d = tmp_path / "rgb"
    d.mkdir()
    index = {}
    for i in range(n):
        p = d / f"0_{i:05d}.png"
        Image.fromarray(np.full((h, w, 3), 10 * (i + 1), np.uint8)).save(p)
        index[str(i)] = str(p)
    return d, index


def test_a_frames_folder_stream_is_0_and_serves_its_frames_in_render_pixels(tmp_path) -> None:
    d, index = _frames_dir(tmp_path)
    t = ImageTransform((8, 6), (Resize(size=(4, 3)),))
    run = _run(tmp_path, [{"stream_id": "0", "video_path": d,
                           "fps": 30.0, "num_frames": 3, "width": 8, "height": 6,
                           "metadata": {"source": "frames", "frame_index": index}}],
               {"0": t.to_dict()})
    s = RunStreams.from_run(run).resolve(None)
    assert s.id == "0" and s.num_frames == 3 and s.size == (8, 6)
    assert s.frame(1)[0, 0].tolist() == [20, 20, 20]
    assert s.frame_in_render(1).shape == (3, 4, 3)
    assert s.key(2) == "0:2" and parse_key(s.key(2)) == ("0", 2)




def test_several_streams_and_none_named_is_an_error_not_a_guess(tmp_path) -> None:
    run = _run(tmp_path, [{"stream_id": "cam_00", "num_frames": 4},
                          {"stream_id": "cam_01", "num_frames": 4}])
    rs = RunStreams.from_run(run)
    assert rs.ids == ["cam_00", "cam_01"]
    with pytest.raises(StreamError, match="2 streams"):
        rs.resolve(None)
    assert rs.resolve("cam_01").id == "cam_01"
    with pytest.raises(StreamError, match="no stream 'cam_02'"):
        rs.get("cam_02")


def test_no_recorded_transform_is_refused_not_assumed(tmp_path) -> None:
    run = _run(tmp_path, [{"stream_id": "cam_00", "num_frames": 4}], {})
    with pytest.raises(StreamError, match="no image transform"):
        _ = RunStreams.from_run(run).only().transform


def test_frame_keys_keep_colons_in_the_stream_id() -> None:
    assert parse_key("cam_00:12") == ("cam_00", 12)
    assert parse_key("rig:left:7") == ("rig:left", 7)
    with pytest.raises(ValueError):
        parse_key("12")


def test_render_pixel_masks_finds_a_stream_that_is_not_cam_00(tmp_path) -> None:
    """The bug this module exists for: spin's stream is '0', and the mask exporter asked for
    'cam_00' and failed at the end of a 26-minute run."""
    from reifyscene.graph.renderable_handle import render_pixel_masks

    run = _run(tmp_path, [{"stream_id": "0", "num_frames": 2}],
               {"0": ImageTransform.identity((8, 6)).to_dict()})
    try:
        render_pixel_masks(RunOutputs.open(run))
    except StreamError:
        pytest.fail("the run's only stream was not found")
    except NotARun:
        pass            # past the stream: it now wants the graph and segment this fixture lacks


# -- FrameStream as a torch Dataset, and what it refuses to guess ---------------------------------

def _folder_run(tmp_path, frame_numbers, *, layout=None, fps_record=None):
    """A run over a folder of frames named by ``frame_numbers``, optionally inside a DAVIS or
    DyCheck layout."""
    if layout == "davis":
        d = tmp_path / "DAVIS" / "JPEGImages" / "480p" / "seq"
        (tmp_path / "DAVIS" / "Annotations" / "480p" / "seq").mkdir(parents=True)
    elif layout == "dycheck":
        root = tmp_path / "spin"
        d = root / "rgb" / "2x"
        (root / "camera").mkdir(parents=True)
        (root / "extra.json").write_text(json.dumps({"fps": fps_record} if fps_record else {}))
    else:
        d = tmp_path / "plain"
    d.mkdir(parents=True)
    index = {}
    for f in frame_numbers:
        p = d / f"{f:05d}.png"
        Image.fromarray(np.full((6, 8, 3), f, np.uint8)).save(p)
        index[str(f)] = str(p)
    run = _run(tmp_path, [{"stream_id": "cam_00", "video_path": str(d), "fps": fps_record,
                           "num_frames": len(index), "width": 8, "height": 6,
                           "metadata": {"source": "frames", "frame_index": index}}],
               {"cam_00": ImageTransform((8, 6), (Resize(size=(4, 3)),)).to_dict()})
    return run, d


def test_a_stream_is_a_torch_dataset_over_the_frames_that_exist(tmp_path) -> None:
    from torch.utils.data import DataLoader

    from reifyscene.ingest.streams import collate_frames

    run, _ = _folder_run(tmp_path, [3, 5, 10])        # starts above 0 and skips
    s = RunStreams.from_run(run).only()
    assert s.frames == [3, 5, 10] and len(s) == 3
    assert s.capability()["contiguous"] is False
    item = s[1]
    assert item["frame"] == 5 and item["key"] == "cam_00:5" and item["image"].shape == (3, 6, 8)
    batch = next(iter(DataLoader(s.in_render(), batch_size=3, collate_fn=collate_frames)))
    assert tuple(batch["image"].shape) == (3, 3, 3, 4)          # in the recorded render raster
    assert batch["frame"].tolist() == [3, 5, 10]
    assert batch["image"][2, 0, 0, 0].item() == 10               # frame 10's own pixels


def test_a_folder_of_frames_has_no_frame_rate_whatever_ingest_wrote(tmp_path) -> None:
    from reifyscene.common.types import UnknownFrameRate

    run, _ = _folder_run(tmp_path, [0, 1])
    s = RunStreams.from_run(run).only()
    assert s.fps is None
    assert s.capability()["physical_time"] == "unavailable"
    assert s[0]["seconds"] is None
    with pytest.raises(UnknownFrameRate, match="records no frame rate"):
        s.seconds(1)


def test_dycheck_records_its_frame_rate(tmp_path) -> None:
    run, _ = _folder_run(tmp_path, [0, 1, 2], layout="dycheck", fps_record=60.0)
    s = RunStreams.from_run(run).only()
    assert s.source.kind == "dycheck" and s.source.factor == 2 and s.source.sequence == "spin"
    assert s.fps == 60.0 and s.source.fps_source == "dataset record"
    assert s.seconds(2) == pytest.approx(2 / 60)


def test_davis_has_no_frame_rate_and_finds_annotations_by_the_frame_file(tmp_path) -> None:
    run, d = _folder_run(tmp_path, [0, 1], layout="davis")
    ann = tmp_path / "DAVIS" / "Annotations" / "480p" / "seq"
    a = np.zeros((6, 8), np.uint8)
    a[1:3, 2:5] = 2
    Image.fromarray(a).save(ann / "00001.png")               # only frame 1 is annotated
    s = RunStreams.from_run(run).only()
    assert s.source.kind == "davis" and s.source.sequence == "seq" and s.fps is None
    assert s.annotation(0) is None
    assert (s.annotation(1) == a).all()
    placed = s.mask_in_render(s.annotation(1) == 2)
    assert placed.shape == (3, 4) and placed.any()           # placed, not squashed


def test_a_bare_folder_has_no_annotations_to_score_against(tmp_path) -> None:
    run, _ = _folder_run(tmp_path, [0])
    with pytest.raises(StreamError, match="no per-frame annotations"):
        RunStreams.from_run(run).only().annotation(0)


def test_the_opencv_rung_returns_exactly_the_frame_asked_for(tmp_path, monkeypatch) -> None:
    import cv2

    from reifyscene.ingest import frame_decoders

    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (16, 8))
    for i in range(6):
        w.write(np.full((8, 16, 3), 40 * i, np.uint8))
    w.release()
    monkeypatch.setattr(frame_decoders, "video_backend", lambda: "opencv")
    for i in (0, 3, 5):
        assert abs(int(frame_decoders.read_video_frame(path, i)[0, 0, 0]) - 40 * i) <= 3
    assert frame_decoders.video_fps(path) == pytest.approx(12.0)
    with pytest.raises(frame_decoders.FrameNotFound):
        frame_decoders.read_video_frame(path, 6)

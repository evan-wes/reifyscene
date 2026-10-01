"""Tests for exporting a run as posed RGB-D and reading a reference tool's output back.

The comparison these support is only worth running if the export is faithful: a scale
factor applied to depth but not to translation, or a frame index taken as a DAVIS frame
number, would produce a number that looks like a result and means nothing.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reifyscene.eval.benchmarks.inventory.score import score_masks_by_frame
from reifyscene.eval.reference.conceptgraphs import export_run, masks_by_frame, scale_to_reference
from reifyscene.eval.reference.conceptgraphs.export import PNG_DEPTH_SCALE
from tests.conftest import write_ingest

W, H = 32, 20
INTR = {"fx": 40.0, "fy": 40.0, "cx": W / 2, "cy": H / 2}


def _pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """A camera-to-world pose with a real rotation, so a dropped inverse is visible."""
    theta = 0.3
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = (tx, ty, tz)
    return T


def _camera_points(depth: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return np.stack([
        (u - INTR["cx"]) * depth / INTR["fx"],
        (v - INTR["cy"]) * depth / INTR["fy"],
        depth,
    ], -1)


def _make_run(tmp_path: Path, frames: list[int], *, source_size=(W, H)) -> tuple[Path, Path]:
    """A minimal run: exact pinhole pointmaps, so any export error is the export's."""
    run = tmp_path / "inv_synthetic"
    (run / "geometry" / "pointmaps").mkdir(parents=True)
    (run / "geometry" / "depthmaps").mkdir(parents=True)
    src = tmp_path / "frames"
    src.mkdir()
    n = max(frames) + 1
    write_ingest(run, video_path=src, num_frames=n, width=source_size[0],
                 height=source_size[1],
                 frame_index={str(i): str(src / f"{i:05d}.jpg") for i in range(n)})
    states, point_index, selected = [], {}, []
    rng = np.random.default_rng(0)
    for i in range(max(frames) + 1):
        depth = 2.0 + rng.uniform(0, 1.0, size=(H, W))
        T = _pose(tz=0.05 * i)
        world = _camera_points(depth) @ T[:3, :3].T + T[:3, 3]
        states.append({"stream_id": "cam_00", "time_sec": i / 30, "pose": T.tolist()})
        selected.append({"stream_id": "cam_00", "frame_idx": i, "timestamp_sec": i / 30})
        if i in frames:
            np.savez(run / "geometry" / "pointmaps" / f"cam_00__{i}.npz",
                     points=world.astype(np.float32))
            point_index[f"cam_00:{i}"] = f"cam_00__{i}.npz"
    (run / "geometry" / "artifact.json").write_text(json.dumps({
        "intrinsics_by_stream": {"cam_00": INTR},
        "camera_states_by_stream": {"cam_00": states},
        "selected_frames": selected,
        "point_index": point_index,
        "metadata": {"image_transforms": {"cam_00": {
            "source_size": list(source_size),
            "ops": [{"op": "resize", "size": [W, H], "resample": "bicubic"}],
            "output_size": [W, H],
        }}},
    }))
    for i in range(n):
        Image.fromarray(
            rng.integers(0, 255, (source_size[1], source_size[0], 3), dtype=np.uint8)
        ).save(src / f"{i:05d}.jpg")
    return run, src


# --- scale ------------------------------------------------------------------

def test_scale_maps_a_clouds_own_extent_onto_the_reference() -> None:
    cloud = np.random.default_rng(0).uniform(0, 1, (5000, 3))
    scale, diagonal = scale_to_reference(cloud, reference_diagonal_m=6.0)
    assert np.isclose(diagonal * scale, 6.0)


def test_scale_ignores_a_lone_outlier() -> None:
    """A single stray point must not set the scale for a whole sequence."""
    cloud = np.random.default_rng(0).uniform(0, 1, (5000, 3))
    with_outlier = np.vstack([cloud, [[1000.0, 1000.0, 1000.0]]])
    a, _ = scale_to_reference(cloud)
    b, _ = scale_to_reference(with_outlier)
    assert np.isclose(a, b, rtol=0.02)


def test_a_degenerate_cloud_does_not_divide_by_zero() -> None:
    scale, diagonal = scale_to_reference(np.zeros((10, 3)))
    assert scale == 1.0 and diagonal == 0.0


# --- geometry round trip ----------------------------------------------------

def test_exported_depth_and_pose_reconstruct_the_world_pointmap(tmp_path: Path) -> None:
    """The whole contract: unprojecting what we wrote must land where our points are."""
    run, src = _make_run(tmp_path, [0, 1, 2])
    out = tmp_path / "export"
    result = export_run(run, out, reference_diagonal_m=6.0)

    manifest = json.loads((out / "manifest.json").read_text())
    scale = manifest["scale_factor"]
    traj = np.loadtxt(out / "traj.txt").reshape(-1, 4, 4)

    for idx, frame in enumerate(manifest["source_frame_for_index"]):
        depth_m = np.asarray(
            Image.open(out / "results" / f"depth{idx:05d}.png"), dtype=np.float64
        ) / PNG_DEPTH_SCALE
        world = _camera_points(depth_m) @ traj[idx][:3, :3].T + traj[idx][:3, 3]
        ours = np.load(run / "geometry" / "pointmaps" / f"cam_00__{frame}.npz")["points"] * scale
        # 1 mm quantisation of the uint16 depth is the only error a pinhole cloud can have
        assert np.abs(world - ours).max() < 0.01
    assert result.pinhole_residual_pct < 1e-6  # synthetic points ARE pinhole
    assert result.num_frames == 3


def test_translation_is_scaled_with_depth_and_rotation_is_not(tmp_path: Path) -> None:
    """Scaling depth but not translation would shear the whole trajectory."""
    run, src = _make_run(tmp_path, [0, 1, 2])
    out = tmp_path / "export"
    result = export_run(run, out, reference_diagonal_m=6.0)
    traj = np.loadtxt(out / "traj.txt").reshape(-1, 4, 4)
    original = np.asarray(json.loads((run / "geometry" / "artifact.json").read_text())
                          ["camera_states_by_stream"]["cam_00"][1]["pose"])
    assert np.allclose(traj[1][:3, :3], original[:3, :3])
    assert np.allclose(traj[1][:3, 3], original[:3, 3] * result.scale_factor)


def test_non_positive_depth_is_written_as_no_measurement(tmp_path: Path) -> None:
    """ConceptGraphs reads 0 as "no depth"; a negative z must not wrap around in uint16."""
    run, src = _make_run(tmp_path, [0])
    pts = np.load(run / "geometry" / "pointmaps" / "cam_00__0.npz")["points"]
    T = np.asarray(json.loads((run / "geometry" / "artifact.json").read_text())
                   ["camera_states_by_stream"]["cam_00"][0]["pose"])
    cam = (pts - T[:3, 3]) @ T[:3, :3]
    cam[0, 0, 2] = -5.0                       # one pixel behind the camera
    world = cam @ T[:3, :3].T + T[:3, 3]
    np.savez(run / "geometry" / "pointmaps" / "cam_00__0.npz", points=world.astype(np.float32))

    out = tmp_path / "export"
    result = export_run(run, out)
    depth = np.asarray(Image.open(out / "results" / "depth00000.png"))
    assert depth[0, 0] == 0
    assert result.behind_camera_pct > 0


# --- frame mapping ----------------------------------------------------------

def test_a_sparse_frame_subset_keeps_its_source_numbering(tmp_path: Path) -> None:
    """The bug this prevents: index 1 of the export is DAVIS frame 5, not frame 1."""
    run, src = _make_run(tmp_path, [0, 5, 9])
    out = tmp_path / "export"
    export_run(run, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_frame_for_index"] == [0, 5, 9]

    objects = [{"image_idx": [0, 1, 2], "mask": [np.ones((H, W), bool)] * 3,
                "class_name": ["bear"] * 3}]
    path = tmp_path / "objects.pkl.gz"
    with gzip.open(path, "wb") as fh:
        pickle.dump({"objects": objects}, fh)
    by_frame, _ = masks_by_frame(path, out / "manifest.json")
    assert sorted(by_frame) == [0, 5, 9]


def test_masks_come_back_in_source_pixels(tmp_path: Path) -> None:
    """Masks are produced in the geometry's raster; ground truth is in the source's."""
    run, src = _make_run(tmp_path, [0], source_size=(W * 2, H * 2))
    out = tmp_path / "export"
    export_run(run, out)
    objects = [{"image_idx": [0], "mask": [np.ones((H, W), bool)], "class_name": ["thing"]}]
    path = tmp_path / "objects.pkl.gz"
    with gzip.open(path, "wb") as fh:
        pickle.dump({"objects": objects}, fh)
    by_frame, _ = masks_by_frame(path, out / "manifest.json")
    assert by_frame[0]["0"].shape == (H * 2, W * 2)


def test_objects_below_the_detection_floor_are_dropped(tmp_path: Path) -> None:
    run, src = _make_run(tmp_path, [0, 1, 2])
    out = tmp_path / "export"
    export_run(run, out)
    objects = [
        {"image_idx": [0, 1, 2], "mask": [np.ones((H, W), bool)] * 3, "class_name": ["a"] * 3},
        {"image_idx": [0], "mask": [np.ones((H, W), bool)], "class_name": ["b"]},
    ]
    path = tmp_path / "objects.pkl.gz"
    with gzip.open(path, "wb") as fh:
        pickle.dump({"objects": objects}, fh)
    kept, info = masks_by_frame(path, out / "manifest.json", min_detections=3)
    assert info["objects_kept"] == 1 and info["objects_skipped"] == 1
    assert set(kept[0]) == {"0"}
    assert info["labels"]["0"] == "a"


def test_a_missing_image_transform_is_refused_not_guessed(tmp_path: Path) -> None:
    run, src = _make_run(tmp_path, [0])
    art = json.loads((run / "geometry" / "artifact.json").read_text())
    art["metadata"] = {}
    (run / "geometry" / "artifact.json").write_text(json.dumps(art))
    with pytest.raises(ValueError, match="no image transform"):
        export_run(run, tmp_path / "export")


# --- both systems reach the scorer the same way ------------------------------

def _gt_from(ann: Path):
    """The scorer's ground-truth lookup over this test's own annotation files."""
    def gt(frame: int) -> dict[int, np.ndarray]:
        path = ann / f"{frame:05d}.png"
        if not path.is_file():
            return {}
        arr = np.asarray(Image.open(path))
        return {int(i): arr == i for i in np.unique(arr) if int(i) != 0}
    return gt


def test_reference_masks_score_through_the_same_function(tmp_path: Path) -> None:
    """A comparison in which each side has its own scorer is not a comparison."""
    ann = tmp_path / "ann"
    ann.mkdir()
    gt = np.zeros((H, W), np.uint8)
    gt[5:15, 5:15] = 1
    Image.fromarray(gt).save(ann / "00000.png")

    ours = {0: {"node-1": gt.astype(bool)}}
    score = score_masks_by_frame(ours, _gt_from(ann), sequence="synthetic", frames=[0])
    assert score.recovered == 1 and score.num_gt == 1 and score.spurious == 0


# --- the scorer is comparable across systems -------------------------------

def _ann(tmp_path: Path, per_frame: dict[int, list[tuple[int, tuple[int, int, int, int]]]]) -> Path:
    """Write DAVIS-style annotations: {frame: [(gt_id, (y0, y1, x0, x1)), ...]}."""
    ann = tmp_path / "ann"
    ann.mkdir(exist_ok=True)
    for frame, objects in per_frame.items():
        arr = np.zeros((H, W), np.uint8)
        for gid, (y0, y1, x0, x1) in objects:
            arr[y0:y1, x0:x1] = gid
        Image.fromarray(arr).save(ann / f"{frame:05d}.png")
    return ann


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_two_systems_peaking_on_different_frames_share_a_denominator(tmp_path: Path) -> None:
    """The defect this pins: the scorer used to report each system's *best* frame, so the
    denominator was that frame's object count. On ``lindy-hop`` that scored us against 8
    objects and the reference against 5 -- two different questions reported as one number.
    """
    ann = _ann(tmp_path, {
        0: [(1, (2, 8, 2, 8)), (2, (10, 16, 10, 16))],   # two objects
        5: [(1, (2, 8, 2, 8))],                           # one: object 2 has left
    })
    # A peaks on frame 5 (finds the only object there); B peaks on frame 0 (finds both).
    a = {0: {"a1": _box(2, 8, 2, 8)}, 5: {"a1": _box(2, 8, 2, 8)}}
    b = {0: {"b1": _box(2, 8, 2, 8), "b2": _box(10, 16, 10, 16)}, 5: {}}

    sa = score_masks_by_frame(a, _gt_from(ann), sequence="s", frames=[0, 5])
    sb = score_masks_by_frame(b, _gt_from(ann), sequence="s", frames=[0, 5])
    assert sa.num_gt == sb.num_gt == 2, (
        "the denominator must be the frame set's objects, not a system's"
    )
    assert sa.recovered == 1 and sb.recovered == 2


def test_a_frame_the_system_missed_entirely_still_counts_against_it(tmp_path: Path) -> None:
    """Producing no masks on a frame must not excuse its objects from the denominator."""
    ann = _ann(tmp_path, {0: [(1, (2, 8, 2, 8))], 5: [(2, (10, 16, 10, 16))]})
    silent = {0: {"n1": _box(2, 8, 2, 8)}}          # nothing at all on frame 5
    score = score_masks_by_frame(silent, _gt_from(ann), sequence="s", frames=[0, 5])
    assert score.num_gt == 2 and score.recovered == 1 and score.missed_ids == [2]


def test_an_object_found_on_any_sampled_frame_counts_as_recovered(tmp_path: Path) -> None:
    """One frame of occlusion is not a miss; a miss on every sampled frame is."""
    ann = _ann(tmp_path, {0: [(1, (2, 8, 2, 8))], 5: [(1, (2, 8, 2, 8))]})
    occluded_once = {0: {"n1": _box(2, 8, 2, 8)}, 5: {"n1": _box(18, 19, 28, 29)}}
    score = score_masks_by_frame(occluded_once, _gt_from(ann), sequence="s", frames=[0, 5])
    assert score.recovered == 1 and score.num_gt == 1
    assert score.matches[0]["frame_idx"] == 0, "the reported match names the frame it came from"

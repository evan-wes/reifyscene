"""Unit tests for reifyscene/calibration/gt_scoring.py (calibration arc M0-minimal).

Pose-math trust comes from tests (design decision D6): the Umeyama fixture
below is pinned against an independent oracle — evo v1.36.5
(evo.core.geometry.umeyama_alignment), run once at build time (2026-06-12)
in a throwaway install; provenance in test_umeyama_matches_evo_oracle.
SfM-orchestration against real hloc/COLMAP is integration-scope (manual GPU
proof runs), matching the repo's established posture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import reifyscene.calibration.gt_scoring as gs
from reifyscene.calibration.model_io import ParsedColmapModel, qvec_to_rotmat
from tests.conftest import real_data

# Real-data tests read datasets from the data directory the tests were started with; they
# skip cleanly without one.
Z_NVIDIA = real_data("datasets/nvidia_dynamic/Balloon1")
Z_IPHONE = real_data("datasets/dycheck_iphone/paper-windmill")


def _rotvec_to_rotmat(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Inverse of model_io.qvec_to_rotmat (verified by round-trip below).

    Shepperd's max-diagonal construction: a fixed qw-threshold branch
    divides by ~0 near 180 degrees (float noise put qw at 1.49e-8, just
    above a 1e-8 cutoff — caught by the round-trip test).
    """
    w2 = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    x2 = 1.0 + R[0, 0] - R[1, 1] - R[2, 2]
    y2 = 1.0 - R[0, 0] + R[1, 1] - R[2, 2]
    z2 = 1.0 - R[0, 0] - R[1, 1] + R[2, 2]
    case = int(np.argmax([w2, x2, y2, z2]))
    if case == 0:
        qw = math.sqrt(w2) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    elif case == 1:
        qx = math.sqrt(x2) / 2.0
        qw = (R[2, 1] - R[1, 2]) / (4 * qx)
        qy = (R[0, 1] + R[1, 0]) / (4 * qx)
        qz = (R[0, 2] + R[2, 0]) / (4 * qx)
    elif case == 2:
        qy = math.sqrt(y2) / 2.0
        qw = (R[0, 2] - R[2, 0]) / (4 * qy)
        qx = (R[0, 1] + R[1, 0]) / (4 * qy)
        qz = (R[1, 2] + R[2, 1]) / (4 * qy)
    else:
        qz = math.sqrt(z2) / 2.0
        qw = (R[1, 0] - R[0, 1]) / (4 * qz)
        qx = (R[0, 2] + R[2, 0]) / (4 * qz)
        qy = (R[1, 2] + R[2, 1]) / (4 * qz)
    return np.array([qw, qx, qy, qz])


def test_rotmat_qvec_roundtrip() -> None:
    # includes 180-degree rotations (the qw<=1e-8 branch of _rotmat_to_qvec);
    # compare R not q since the quaternion sign is ambiguous there
    for axis, theta in [
        ((1, 0, 0), 0.3), ((1, 2, 3), 1.7), ((0, 1, -1), 2.9),
        ((1, 0, 0), math.pi), ((1, 1, 0), math.pi),
    ]:
        R = _rotvec_to_rotmat(np.array(axis, dtype=float), theta)
        assert np.allclose(qvec_to_rotmat(_rotmat_to_qvec(R)), R, atol=1e-9)


# --------------------------------------------------------------------------
# Pose math (D6/D11)
# --------------------------------------------------------------------------

_SRC = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.2, -0.1],
    [0.3, 1.1, 0.4],
    [-0.7, 0.5, 1.2],
    [0.9, -0.8, 0.6],
    [-0.2, -0.4, -0.9],
])
_NOISE = np.array([
    [0.011, -0.008, 0.004],
    [-0.006, 0.012, -0.009],
    [0.007, 0.003, 0.010],
    [-0.012, -0.005, -0.003],
    [0.005, 0.009, -0.011],
    [0.002, -0.010, 0.008],
])


def _oracle_dst() -> np.ndarray:
    R_true = _rotvec_to_rotmat(np.array([1.0, 2.0, 3.0]), 0.7)
    return (2.5 * (R_true @ _SRC.T)).T + np.array([0.4, -1.2, 2.0]) + _NOISE


def test_umeyama_matches_evo_oracle() -> None:
    # Oracle: evo v1.36.5 umeyama_alignment(src.T, dst.T, with_scale=True),
    # generated 2026-06-12 from these exact literals (see module docstring).
    s, R, t = gs.umeyama_sim3(_SRC, _oracle_dst())
    assert s == pytest.approx(2.502050822832259, abs=1e-12)
    R_oracle = np.array([
        [0.7799465626119249, -0.48481759065040875, 0.3957717312615649],
        [0.5521328871537398, 0.830779640031825, -0.07038795800325319],
        [-0.2946737762223155, 0.27341743452919215, 0.9156474605995237],
    ])
    t_oracle = np.array([0.401630138493866, -1.2015016395126548, 1.9999889381212286])
    assert np.allclose(R, R_oracle, atol=1e-12)
    assert np.allclose(t, t_oracle, atol=1e-12)
    res = _oracle_dst() - ((s * (R @ _SRC.T)).T + t)
    res_oracle = [
        0.012087663589914949, 0.010991572161101155, 0.014326302854847171,
        0.018016912894420516, 0.008431138849088771, 0.010740107855249514,
    ]
    assert np.allclose(np.linalg.norm(res, axis=1), res_oracle, atol=1e-12)


def test_umeyama_exact_roundtrip_and_guards() -> None:
    R_true = _rotvec_to_rotmat(np.array([0.2, -1.0, 0.5]), 1.1)
    dst = (0.7 * (R_true @ _SRC.T)).T + np.array([-3.0, 0.5, 8.0])
    s, R, t = gs.umeyama_sim3(_SRC, dst)
    assert s == pytest.approx(0.7, abs=1e-12)
    assert np.allclose(R, R_true, atol=1e-12)
    with pytest.raises(ValueError):
        gs.umeyama_sim3(_SRC[:2], _SRC[:2])  # <3 points
    with pytest.raises(ValueError):
        gs.umeyama_sim3(np.zeros((4, 3)), np.ones((4, 3)))  # coincident src


def test_umeyama_reflection_yields_proper_rotation() -> None:
    # Mirrored data must not produce det(R) = -1 (the Umeyama S-correction).
    dst = _SRC.copy()
    dst[:, 2] *= -1.0
    _, R, _ = gs.umeyama_sim3(_SRC, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_geodesic_boundaries() -> None:
    R = _rotvec_to_rotmat(np.array([0.0, 0.0, 1.0]), math.radians(34.0))
    assert gs.geodesic_deg(R, np.eye(3)) == pytest.approx(34.0, abs=1e-9)
    assert gs.geodesic_deg(R, R) == pytest.approx(0.0, abs=1e-9)  # clamp at cos=1
    R180 = np.diag([1.0, -1.0, -1.0])
    assert gs.geodesic_deg(R180, np.eye(3)) == pytest.approx(180.0, abs=1e-9)


def test_center_conditioning_flags() -> None:
    line = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    flags = gs.center_conditioning(line)
    assert flags["colinear"] and flags["planar"]
    square = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    flags = gs.center_conditioning(square)
    assert flags["planar"] and not flags["colinear"]
    flags = gs.center_conditioning(_SRC)
    assert not flags["planar"] and not flags["colinear"]


def _make_rig(n: int = 4) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rig = {}
    for i in range(n):
        R = _rotvec_to_rotmat(np.array([0.1 + i, 1.0, -0.5 * i + 0.2]), 0.4 + 0.3 * i)
        C = np.array([1.5 * i, 0.7 * (i % 2), 0.3 * i * i + 0.1])
        rig[f"{i}_00000.png"] = (R, C)
    return rig


def _gauge_transform(
    rig: dict[str, tuple[np.ndarray, np.ndarray]], s: float, G: np.ndarray, t: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # world x -> s·G·x + t  ⇒  centers map through; w2c rotations pick up Gᵀ
    return {k: (R @ G.T, s * G @ C + t) for k, (R, C) in rig.items()}


def test_pairwise_errors_are_gauge_invariant() -> None:
    gt = _make_rig()
    G = _rotvec_to_rotmat(np.array([2.0, -1.0, 0.4]), 0.9)
    est = _gauge_transform(gt, 3.3, G, np.array([5.0, -2.0, 1.0]))
    out = gs.pairwise_relative_errors(est, gt)
    # acos amplifies float error near 0deg (~sqrt(eps)); 1e-5 deg is the
    # practical noise floor for an exactly-gauge-equivalent rig
    assert out["rot_deg"]["max"] == pytest.approx(0.0, abs=1e-5)
    assert out["trans_dir_deg"]["max"] == pytest.approx(0.0, abs=1e-5)
    assert out["baseline_ratio_err"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert out["rot_deg"]["count"] == 6  # C(4,2) pairs


def test_pairwise_errors_localize_a_perturbed_camera() -> None:
    gt = _make_rig()
    est = {k: (R.copy(), C.copy()) for k, (R, C) in gt.items()}
    bump = _rotvec_to_rotmat(np.array([0.0, 0.0, 1.0]), math.radians(2.0))
    name = "2_00000.png"
    est[name] = (bump @ est[name][0], est[name][1])
    out = gs.pairwise_relative_errors(est, gt)
    assert out["rot_deg"]["max"] == pytest.approx(2.0, abs=1e-6)
    bad_pairs = [p for p in out["pairs"] if name in p["pair"]]
    good_pairs = [p for p in out["pairs"] if name not in p["pair"]]
    assert all(p["rot_deg"] == pytest.approx(2.0, abs=1e-6) for p in bad_pairs)
    assert all(p["rot_deg"] == pytest.approx(0.0, abs=1e-9) for p in good_pairs)


def test_pairwise_degenerate_baseline_excluded() -> None:
    gt = _make_rig(3)
    dup_R = _rotvec_to_rotmat(np.array([1.0, 1.0, 0.0]), 0.2)
    gt["dup.png"] = (dup_R, gt["0_00000.png"][1] + 1e-9)  # same center
    est = {k: (R.copy(), C.copy()) for k, (R, C) in gt.items()}
    out = gs.pairwise_relative_errors(est, gt)
    assert ["0_00000.png", "dup.png"] in out["degenerate_pairs"]
    # degenerate pairs are excluded from ALL headline stats (they measure
    # registration repeatability, not rig geometry) and reported separately
    assert out["trans_dir_deg"]["count"] == out["rot_deg"]["count"]
    assert out["rot_deg_degenerate_pairs"]["count"] == len(out["degenerate_pairs"])


def test_sim3_ape_gauge_recovery_and_guards() -> None:
    gt = _make_rig()
    G = _rotvec_to_rotmat(np.array([0.3, 0.9, -1.2]), 1.4)
    est = _gauge_transform(gt, 0.25, G, np.array([-1.0, 4.0, 2.5]))
    out = gs.sim3_ape(est, gt)
    assert out["rot_deg"]["max"] == pytest.approx(0.0, abs=1e-7)
    assert out["trans"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert out["scale"] == pytest.approx(1 / 0.25, rel=1e-9)

    two = {k: gt[k] for k in list(gt)[:2]}
    assert "skipped" in gs.sim3_ape(two, two)

    line_gt = {
        f"c{i}": (np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(4)
    }
    out = gs.sim3_ape(line_gt, line_gt)
    assert "skipped" in out and out["conditioning"]["colinear"]


# --------------------------------------------------------------------------
# Adapters + collector (D3/D7)
# --------------------------------------------------------------------------


def _write_synthetic_sequence(root: Path, frames: list[str], focal: float = 500.0) -> None:
    (root / "rgb" / "1x").mkdir(parents=True)
    (root / "camera").mkdir(parents=True)
    for i, frame in enumerate(frames):
        (root / "rgb" / "1x" / f"{frame}.png").write_bytes(b"fakepng")
        R = _rotvec_to_rotmat(np.array([0.2, 1.0, 0.1 + i]), 0.3 + 0.2 * i)
        (root / "camera" / f"{frame}.json").write_text(json.dumps({
            "orientation": R.tolist(),
            "position": [float(i), 0.5 * i, 0.25 * i * i],
            "focal_length": focal,
            "principal_point": [320.0, 240.0],
            "image_size": [640, 480],
            "radial_distortion": [0, 0, 0],
        }))


def test_adapter_enumerate_select_and_load(tmp_path: Path) -> None:
    frames = [f"{c}_{t:05d}" for c in range(2) for t in range(6)]
    _write_synthetic_sequence(tmp_path, frames)
    enumerated = gs.enumerate_frames("nvidia", tmp_path)
    assert enumerated == sorted(frames)
    selected = gs.select_frames("nvidia", enumerated, timesteps=2)
    assert len(selected) == 4  # 2 cams x 2 timesteps
    assert {f.split("_")[0] for f in selected} == {"0", "1"}
    cams = gs.load_gt_cameras("nvidia", tmp_path, selected)
    assert len(cams) == 4
    assert gs.shared_intrinsics(cams) is not None

    sel_iphone = gs.select_frames("iphone", enumerated, num_frames=5)
    assert len(sel_iphone) == 5


def test_adapter_rejects_malformed_gt(tmp_path: Path) -> None:
    _write_synthetic_sequence(tmp_path, ["0_00000"])
    cam = tmp_path / "camera" / "0_00000.json"

    cam.write_text("not json")
    with pytest.raises(gs.GTSchemaError, match="invalid JSON"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    cam.write_text(json.dumps({"orientation": np.eye(3).tolist()}))
    with pytest.raises(gs.GTSchemaError, match="missing keys"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    blob = {
        "orientation": (np.eye(3) * 1.2).tolist(),  # det != 1
        "position": [0, 0, 0], "focal_length": 500.0,
        "principal_point": [1, 1], "image_size": [10, 10],
    }
    cam.write_text(json.dumps(blob))
    with pytest.raises(gs.GTSchemaError, match="not a rotation"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    with pytest.raises(gs.GTSchemaError, match="no frames selected"):
        gs.load_gt_cameras("nvidia", tmp_path, [])
    with pytest.raises(gs.GTSchemaError, match="has no rgb"):
        gs.load_gt_cameras("nvidia", tmp_path, ["9_99999"])


def test_shared_intrinsics_detects_mismatch(tmp_path: Path) -> None:
    _write_synthetic_sequence(tmp_path / "a", ["0_00000"], focal=500.0)
    _write_synthetic_sequence(tmp_path / "b", ["1_00000"], focal=999.0)
    cams = gs.load_gt_cameras("nvidia", tmp_path / "a", ["0_00000"]) + gs.load_gt_cameras(
        "nvidia", tmp_path / "b", ["1_00000"]
    )
    assert gs.shared_intrinsics(cams) is None


def test_collector_manifest_roundtrip(tmp_path: Path) -> None:
    frames = ["0_00000", "1_00000", "1_00001"]
    _write_synthetic_sequence(tmp_path / "seq", frames)
    cams = gs.load_gt_cameras("nvidia", tmp_path / "seq", frames)
    manifest_path = gs.collect_frames(cams, tmp_path / "images")
    assert manifest_path.name == gs.GT_MANIFEST_NAME
    # NEVER inside the image dir — hloc/COLMAP sweep every file in there
    assert manifest_path.parent == tmp_path
    images = gs.load_manifest(manifest_path)
    assert set(images) == {f"{f}.png" for f in frames}
    for cam in cams:
        blob = images[cam.image_name]
        assert np.allclose(np.asarray(blob["R_w2c"]).reshape(3, 3), cam.R_w2c)
        assert np.allclose(blob["center"], cam.center)
        assert (tmp_path / "images" / cam.image_name).exists()
    # idempotent recollect; duplicate image names rejected
    gs.collect_frames(cams, tmp_path / "images")
    dup = cams + [cams[0]]
    with pytest.raises(gs.GTSchemaError, match="duplicate"):
        gs.collect_frames(dup, tmp_path / "images2")
    # a smaller re-collection PURGES stale images — hloc sweeps every file
    # in the dir, so leftovers would silently join the reconstruction
    gs.collect_frames(cams[:2], tmp_path / "images")
    remaining = {p.name for p in (tmp_path / "images").iterdir()}
    assert remaining == {c.image_name for c in cams[:2]}


# --------------------------------------------------------------------------
# Scoring orchestration (D5)
# --------------------------------------------------------------------------


def _model_from_poses(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    components: list[list[str]] | None = None,
) -> ParsedColmapModel:
    records = {}
    for i, (name, (R, C)) in enumerate(sorted(poses.items())):
        records[name] = {
            "image_id": i + 1,
            "camera_id": 1,
            "qvec": _rotmat_to_qvec(R).tolist(),
            "tvec": (-R @ C).tolist(),
        }
    return ParsedColmapModel(
        intrinsics_by_camera_id={},
        image_records=records,
        point_records={},
        connected_components=components if components is not None else [sorted(records)],
        reprojection_stats={},
    )


def _manifest_from_rig(rig: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict]:
    return {
        name: {
            "source_frame": name.removesuffix(".png"),
            "R_w2c": R.reshape(-1).tolist(),
            "center": C.tolist(),
            "focal": 500.0,
            "principal_point": [320.0, 240.0],
            "image_size": [640, 480],
        }
        for name, (R, C) in rig.items()
    }


def test_score_against_gt_happy_path() -> None:
    gt = _make_rig()
    G = _rotvec_to_rotmat(np.array([1.0, 0.2, 0.5]), 0.8)
    est = _gauge_transform(gt, 2.0, G, np.array([0.5, 1.5, -2.0]))
    result = gs.score_against_gt(_model_from_poses(est), _manifest_from_rig(gt))
    assert result["alignable"] is True
    assert result["registration"]["cameras_registered"] == 4
    assert result["registration"]["cameras_total"] == 4
    assert result["fragmentation"]["num_components"] == 1
    assert result["scored_coverage"] == pytest.approx(1.0)
    # qvec round-trip quantization adds ~1e-6 deg; 1e-4 is still far below
    # any meaningful calibration error
    assert result["metrics"]["pairwise"]["rot_deg"]["max"] == pytest.approx(0.0, abs=1e-4)
    assert result["metrics"]["sim3_ape"]["scale"] == pytest.approx(0.5, rel=1e-6)
    assert set(result["per_camera_images"]) == {"0", "1", "2", "3"}


def test_score_against_gt_fragmentation_and_unregistered() -> None:
    gt = _make_rig(6)
    est = {k: (R.copy(), C.copy()) for k, (R, C) in gt.items()}
    names = sorted(gt)
    dropped = names[5]
    del est[dropped]  # COLMAP failed to register one image
    components = [names[0:3], names[3:5]]  # fragmented model
    model = _model_from_poses(est, components=components)
    result = gs.score_against_gt(model, _manifest_from_rig(gt))
    assert result["alignable"] is True
    assert result["fragmentation"]["num_components"] == 2
    assert result["scored_component_size"] == 3
    assert result["scored_coverage"] == pytest.approx(3 / 6)
    assert dropped in result["registration"]["unregistered_images"]

    # largest component below 3 matched cameras -> structured non-alignable
    tiny = gs.score_against_gt(
        _model_from_poses({n: est[n] for n in names[0:2]}, components=[names[0:2]]),
        _manifest_from_rig(gt),
    )
    assert tiny["alignable"] is False
    assert "fewer than 3" in tiny["error"]


def test_score_against_gt_extraneous_images_listed() -> None:
    gt = _make_rig(3)
    est = dict(gt)
    est["mystery.png"] = (np.eye(3), np.zeros(3))
    result = gs.score_against_gt(_model_from_poses(est), _manifest_from_rig(gt))
    assert result["registration"]["extraneous_images"] == ["mystery.png"]


# --------------------------------------------------------------------------
# Gap-closure tests (ship review round)
# --------------------------------------------------------------------------


def test_center_conditioning_small_n_degrades_not_crashes() -> None:
    # svd of (N,3) yields min(N,3) singular values; N<3 must degrade to
    # "degenerate", never IndexError (verified crash pre-fix).
    for pts in (np.zeros((1, 3)), np.array([[0.0, 0, 0], [1, 0, 0]])):
        flags = gs.center_conditioning(pts)
        assert flags["colinear"] and flags["planar"]
    coincident = np.ones((4, 3))
    flags = gs.center_conditioning(coincident)
    assert flags["colinear"] and flags["planar"]


def test_build_sfm_cfg_payload_exact() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_calibration_gt_payload",
        Path(__file__).parents[1] / "scripts" / "score_calibration_gt.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cam = gs.GTCamera(
        image_name="0_00000.png", source_frame="0_00000", rgb_path=Path("x.png"),
        R_w2c=np.eye(3), center=np.zeros(3),
        focal=804.5, principal_point=(470.05, 248.0), image_size=(940, 496),
    )
    cfg = mod.build_sfm_cfg(True, [cam])
    # the exact injection payload — a swapped cx/cy would only surface on GPU
    assert cfg == {
        "camera_model": "SIMPLE_PINHOLE",
        "camera_params": "804.5,470.05,248.0",
        "fix_intrinsics": True,
        "camera_mode": "SINGLE",
    }
    # free mode pins camera_mode explicitly (flat dir = shared camera; a
    # silent PER_FOLDER default would mean the same thing undisclosed)
    assert mod.build_sfm_cfg(False, [cam]) == {"camera_mode": "SINGLE"}
    drift = gs.GTCamera(
        image_name="1_00000.png", source_frame="1_00000", rgb_path=Path("y.png"),
        R_w2c=np.eye(3), center=np.zeros(3),
        focal=999.0, principal_point=(470.05, 248.0), image_size=(940, 496),
    )
    with pytest.raises(gs.GTSchemaError, match="shared intrinsics"):
        mod.build_sfm_cfg(True, [cam, drift])


def test_adapter_schema_guard_branches(tmp_path: Path) -> None:
    _write_synthetic_sequence(tmp_path, ["0_00000"])
    cam = tmp_path / "camera" / "0_00000.json"

    cam.write_text("[1, 2, 3]")  # top level not an object
    with pytest.raises(gs.GTSchemaError, match="not an object"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    base = {
        "position": [0, 0, 0], "focal_length": 500.0,
        "principal_point": [1, 1], "image_size": [10, 10],
    }
    cam.write_text(json.dumps({**base, "orientation": [[1, 0], [0, 1]]}))
    with pytest.raises(gs.GTSchemaError, match="not 3x3"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    cam.write_text(json.dumps({
        **base, "orientation": np.eye(3).tolist(), "position": [0, 0, 0, 0],
    }))
    with pytest.raises(gs.GTSchemaError, match="not a 3-vector"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    # orthonormal but det = -1 (pure reflection) must be rejected too
    cam.write_text(json.dumps({**base, "orientation": np.diag([1.0, 1.0, -1.0]).tolist()}))
    with pytest.raises(gs.GTSchemaError, match="not a rotation"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00000"])

    # rgb present but camera json missing -> the json-specific branch
    (tmp_path / "rgb" / "1x" / "0_00001.png").write_bytes(b"fakepng")
    with pytest.raises(gs.GTSchemaError, match="has no camera json"):
        gs.load_gt_cameras("nvidia", tmp_path, ["0_00001"])


def test_enumerate_and_select_guard_branches(tmp_path: Path) -> None:
    with pytest.raises(gs.GTSchemaError, match="unknown dataset"):
        gs.enumerate_frames("bogus", tmp_path)
    # rgb and camera populated but DISJOINT -> empty intersection
    (tmp_path / "rgb" / "1x").mkdir(parents=True)
    (tmp_path / "camera").mkdir()
    (tmp_path / "rgb" / "1x" / "0_00000.png").write_bytes(b"x")
    (tmp_path / "camera" / "0_99999.json").write_text("{}")
    with pytest.raises(gs.GTSchemaError, match="no frames with both"):
        gs.enumerate_frames("nvidia", tmp_path)

    # selection clamps: asking for more than exists returns each frame once
    frames = [f"0_{t:05d}" for t in range(3)]
    assert gs.select_frames("nvidia", frames, timesteps=99) == frames
    assert gs.select_frames("iphone", frames, num_frames=99) == frames


def test_shared_intrinsics_all_mismatch_axes(tmp_path: Path) -> None:
    def cam(focal=500.0, pp=(320.0, 240.0), size=(640, 480)):
        return gs.GTCamera(
            image_name="x.png", source_frame="0_00000", rgb_path=Path("x"),
            R_w2c=np.eye(3), center=np.zeros(3),
            focal=focal, principal_point=pp, image_size=size,
        )

    assert gs.shared_intrinsics([cam(), cam(pp=(321.0, 240.0))]) is None
    assert gs.shared_intrinsics([cam(), cam(size=(641, 480))]) is None
    with pytest.raises(gs.GTSchemaError, match="at least one camera"):
        gs.shared_intrinsics([])


def test_load_manifest_rejects_non_manifests(tmp_path: Path) -> None:
    bad = tmp_path / "m.json"
    for content in ('[1,2]', '{"schema": "other", "images": {}}', '{"schema": "gt_manifest_v1"}'):
        bad.write_text(content)
        with pytest.raises(gs.GTSchemaError, match="not a gt_manifest_v1"):
            gs.load_manifest(bad)


def test_pairwise_guards_and_est_collapse_flagging() -> None:
    one = {"a": (np.eye(3), np.zeros(3))}
    with pytest.raises(ValueError, match="at least 2"):
        gs.pairwise_relative_errors(one, one)

    # all GT baselines degenerate -> headline stats are None; the
    # repeatability stats carry the rotations
    rig = {f"c{i}": (np.eye(3), np.zeros(3)) for i in range(3)}
    out = gs.pairwise_relative_errors(rig, rig)
    assert out["rot_deg"] is None
    assert out["trans_dir_deg"] is None
    assert out["baseline_ratio_err"] is None
    assert out["rot_deg_degenerate_pairs"]["count"] == 3

    # healthy GT baseline but COLLAPSED estimate -> flagged, never silent
    gt = _make_rig(3)
    est = {k: (R.copy(), C.copy()) for k, (R, C) in gt.items()}
    names = sorted(gt)
    est[names[1]] = (est[names[1]][0], est[names[0]][1].copy())  # same center
    out = gs.pairwise_relative_errors(est, gt)
    assert [names[0], names[1]] in out["est_degenerate_pairs"]


def test_score_against_gt_nothing_registered() -> None:
    gt = _make_rig(3)
    empty_model = ParsedColmapModel(
        intrinsics_by_camera_id={}, image_records={}, point_records={},
        connected_components=[], reprojection_stats={},
    )
    result = gs.score_against_gt(empty_model, _manifest_from_rig(gt))
    assert result["alignable"] is False
    assert result["registration"]["images_registered"] == 0


def test_ensure_text_model_search_and_exit(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_calibration_gt_etm",
        Path(__file__).parents[1] / "scripts" / "score_calibration_gt.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # nothing parseable -> clear exit
    empty = tmp_path / "sparse"
    empty.mkdir()
    with pytest.raises(SystemExit, match="nothing parseable"):
        mod.ensure_text_model(empty, tmp_path / "out")

    # text model under models/0/ is discovered
    nested = empty / "models" / "0"
    nested.mkdir(parents=True)
    (nested / "images.txt").write_text("# empty\n")
    assert mod.ensure_text_model(empty, tmp_path / "out") == nested


def test_cli_not_alignable_exit_and_output_override(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    frames = [f"{c}_00000" for c in range(2)]  # only 2 cameras -> <3 matched
    _write_synthetic_sequence(tmp_path / "datasets" / "seq", frames)
    cams = gs.load_gt_cameras("nvidia", tmp_path / "datasets" / "seq", frames)
    rig = {c.image_name: (c.R_w2c, c.center) for c in cams}
    _write_colmap_text_model(tmp_path / "mock_model", rig)
    with pytest.raises(SystemExit) as exc:
        _run_cli([
            "--dataset", "nvidia",
            "--data-dir", "seq",
            "--run-dir", "run",
            "--timesteps", "1",
            "--mock-sparse-model-dir", str(tmp_path / "mock_model"),
        ])
    assert "NOT ALIGNABLE" in str(exc.value)

    # --output override with an alignable 4-camera rig
    frames4 = [f"{c}_00000" for c in range(4)]
    _write_synthetic_sequence(tmp_path / "datasets" / "seq4", frames4)
    cams4 = gs.load_gt_cameras("nvidia", tmp_path / "datasets" / "seq4", frames4)
    rig4 = {c.image_name: (c.R_w2c, c.center) for c in cams4}
    _write_colmap_text_model(tmp_path / "mock4", rig4)
    custom = tmp_path / "runs" / "elsewhere" / "score.json"
    _run_cli([
        "--dataset", "nvidia",
        "--data-dir", "seq4",
        "--run-dir", "run4",
        "--timesteps", "1",
        "--output", "elsewhere/score.json",
        "--mock-sparse-model-dir", str(tmp_path / "mock4"),
    ])
    assert json.loads(custom.read_text())["alignable"] is True


# --------------------------------------------------------------------------
# Runner injection (D4/D12) + CLI end-to-end via the mock SfM path
# --------------------------------------------------------------------------


def test_runner_image_options_injection() -> None:
    from reifyscene.calibration.hloc_colmap import HlocColmapSfMRunner

    # no cfg -> empty (existing video path unchanged: run() passes
    # `image_options or None`, so {} still becomes None downstream)
    assert HlocColmapSfMRunner({})._image_options() == {}
    assert HlocColmapSfMRunner({"camera_model": "PINHOLE"})._image_options() == {
        "camera_model": "PINHOLE"
    }
    opts = HlocColmapSfMRunner(
        {"camera_model": "SIMPLE_PINHOLE", "camera_params": "500,320,240"}
    )._image_options()
    assert opts == {"camera_model": "SIMPLE_PINHOLE", "camera_params": "500,320,240"}


def _write_colmap_text_model(
    model_dir: Path, poses: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n1 SIMPLE_PINHOLE 640 480 500 320 240\n"
    )
    lines = ["# Image list"]
    for i, (name, (R, C)) in enumerate(sorted(poses.items())):
        q = _rotmat_to_qvec(R)
        t = -R @ C
        lines.append(
            f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}"
        )
        lines.append("100 100 1")  # one shared observation keeps one component
    (model_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (model_dir / "points3D.txt").write_text(
        "# 3D point list\n1 0.0 0.0 3.0 255 0 0 0.5 "
        + " ".join(f"{i + 1} 0" for i in range(len(poses)))
        + "\n"
    )


def _run_cli(argv: list[str]) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_calibration_gt", Path(__file__).parents[1] / "scripts" / "score_calibration_gt.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys as _sys

    old = _sys.argv
    _sys.argv = ["score_calibration_gt.py", *argv]
    try:
        mod.main()
    finally:
        _sys.argv = old


def test_cli_end_to_end_with_mock_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    frames = [f"{c}_{t:05d}" for c in range(4) for t in range(3)]
    _write_synthetic_sequence(tmp_path / "datasets" / "seq", frames)
    cams = gs.load_gt_cameras(
        "nvidia", tmp_path / "datasets" / "seq", gs.select_frames("nvidia", frames, timesteps=1)
    )
    # the mock SfM "recovers" a gauge-transformed version of GT
    gt_rig = {c.image_name: (c.R_w2c, c.center) for c in cams}
    G = _rotvec_to_rotmat(np.array([0.5, 1.0, -0.3]), 0.6)
    est = _gauge_transform(gt_rig, 1.8, G, np.array([2.0, -1.0, 0.5]))
    _write_colmap_text_model(tmp_path / "mock_model", est)

    run_dir = tmp_path / "runs" / "run"
    _run_cli([
        "--dataset", "nvidia",
        "--data-dir", "seq",
        "--run-dir", "run",
        "--timesteps", "1",
        "--mock-sparse-model-dir", str(tmp_path / "mock_model"),
    ])
    out = capsys.readouterr().out
    assert "Calibration GT score written" in out
    result = json.loads((run_dir / "calibration_gt_score.json").read_text())
    assert result["schema"] == gs.SCORE_SCHEMA
    assert result["alignable"] is True
    assert result["intrinsics_mode"] == "free"
    assert result["registration"]["cameras_registered"] == 4
    assert result["metrics"]["pairwise"]["rot_deg"]["max"] < 1e-4
    assert result["metrics"]["sim3_ape"]["scale"] == pytest.approx(1 / 1.8, rel=1e-6)


def test_cli_fix_intrinsics_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    frames = [f"{c}_{t:05d}" for c in range(4) for t in range(1)]
    _write_synthetic_sequence(tmp_path / "datasets" / "seq", frames)
    cams = gs.load_gt_cameras("nvidia", tmp_path / "datasets" / "seq", frames)
    gt_rig = {c.image_name: (c.R_w2c, c.center) for c in cams}
    _write_colmap_text_model(tmp_path / "mock_model", gt_rig)

    # shared intrinsics -> fixed-gt mode recorded in the result
    _run_cli([
        "--dataset", "nvidia",
        "--data-dir", "seq",
        "--run-dir", "run-fixed",
        "--timesteps", "1",
        "--fix-intrinsics",
        "--mock-sparse-model-dir", str(tmp_path / "mock_model"),
    ])
    result = json.loads((tmp_path / "runs" / "run-fixed" / "calibration_gt_score.json").read_text())
    assert result["intrinsics_mode"] == "fixed-gt"

    # per-frame intrinsics (iphone-style drift) -> --fix-intrinsics refuses
    drift = tmp_path / "datasets" / "drift"
    _write_synthetic_sequence(drift, ["0_00000"], focal=500.0)
    (drift / "camera" / "0_00001.json").write_text(
        (drift / "camera" / "0_00000.json").read_text().replace("500.0", "555.0")
    )
    (drift / "rgb" / "1x" / "0_00001.png").write_bytes(b"fakepng")
    with pytest.raises(SystemExit) as exc:
        _run_cli([
            "--dataset", "iphone",
            "--data-dir", "drift",
            "--run-dir", "run-refuse",
            "--num-frames", "2",
            "--fix-intrinsics",
            "--mock-sparse-model-dir", str(tmp_path / "mock_model"),
        ])
    assert "shared intrinsics" in str(exc.value)


def test_cli_rejects_bad_dataset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REIFYSCENE_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        _run_cli([
            "--dataset", "nvidia",
            "--data-dir", "nope",
            "--run-dir", "run",
        ])
    assert "GT collection failed" in str(exc.value)


# --------------------------------------------------------------------------
# Real-data adapter checks (data on disk; cheap — no SfM)
# --------------------------------------------------------------------------


@pytest.mark.skipif(Z_NVIDIA is None, reason="Balloon1 not in the data directory")
def test_real_balloon1_adapter() -> None:
    frames = gs.enumerate_frames("nvidia", Z_NVIDIA)
    assert {f.split("_")[0] for f in frames} == {"0", "1", "2", "3"}
    selected = gs.select_frames("nvidia", frames, timesteps=2)
    cams = gs.load_gt_cameras("nvidia", Z_NVIDIA, selected)
    assert len(cams) == 8
    rep = gs.shared_intrinsics(cams)
    assert rep is not None and rep.focal == pytest.approx(804.5333, abs=1e-3)


@pytest.mark.skipif(Z_IPHONE is None, reason="paper-windmill not in the data directory")
def test_real_paper_windmill_adapter() -> None:
    frames = gs.enumerate_frames("iphone", Z_IPHONE)
    selected = gs.select_frames("iphone", frames, num_frames=6)
    cams = gs.load_gt_cameras("iphone", Z_IPHONE, selected)
    assert len(cams) == 6
    # Real-data fact: DyCheck/ARKit re-estimates intrinsics per frame
    # (focal drifts 719.9 -> 724.9 across the trajectory), so the
    # shared-intrinsics gate correctly refuses --fix-intrinsics on iphone
    # sequences (D12 global-injection limit).
    assert gs.shared_intrinsics(cams) is None

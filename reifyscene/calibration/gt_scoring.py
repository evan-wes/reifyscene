"""Score recovered calibration extrinsics against dataset ground truth.

Design: the calibration-arc design document, kept outside the repository (eng-review decisions
D3–D12). M0-minimal scope.

    dataset seq ──adapter──► GTCamera[] ──collector──► image_dir/ + gt_manifest.json
                                                            │
                                            HlocColmapSfMRunner (existing)
                                                            │
                                            parse_colmap_text_model (existing)
                                                            │
        gt_manifest ∩ image_records ──join──► matched (est, gt) pose pairs
                                                            │
                    ┌── pairwise relative errors   (PRIMARY — gauge-free)
                    └── Sim(3) APE on centers      (secondary — guarded)

Conventions: GT camera JSONs store ``orientation`` = R_w2c (world→camera
rotation) and ``position`` = camera center in world coordinates — the same
convention the SoM dataloaders use (w2c = [R | -R·C]). COLMAP image records
carry qvec/tvec in the same world→camera sense; centers are -Rᵀt.

The PRIMARY metrics are pairwise-relative errors (D11): with N as small as
4 cameras, fitting Sim(3)'s 7 DOF to the centers absorbs a flattering share
of the true error, so the gauge-free pair metrics lead and the aligned
absolute errors are secondary, with a conditioning guard for near-planar /
near-colinear rigs.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reifyscene.calibration.model_io import ParsedColmapModel, qvec_to_rotmat
from reifyscene.ingest.datasets import DatasetLayoutError, DyCheckSequence

GT_MANIFEST_NAME = "gt_manifest.json"
MANIFEST_SCHEMA = "gt_manifest_v1"
SCORE_SCHEMA = "calibration_gt_score_v1"

# Frame-selection defaults — single source for both the library signature
# and the CLI argparse defaults.
DEFAULT_TIMESTEPS = 5
DEFAULT_NUM_FRAMES = 24

# Singular-value ratio thresholds on centered GT centers, used to flag
# alignments whose Sim(3) fit is ill-conditioned (D11 guard).
_PLANAR_RATIO = 1e-3
_COLINEAR_RATIO = 1e-3
# A camera pair closer than this fraction of the mean baseline has no
# meaningful direction; the pair is reported but excluded from direction
# statistics (near-duplicate views from a static camera, for instance).
_MIN_BASELINE_FRACTION = 1e-3


class GTSchemaError(ValueError):
    """Dataset GT does not match the expected schema/content."""


@dataclass
class GTCamera:
    """One collected image with its ground-truth pose and intrinsics."""

    image_name: str  # filename inside the collected image dir
    source_frame: str  # dataset frame name (e.g. "1_00042")
    rgb_path: Path
    R_w2c: np.ndarray  # (3, 3)
    center: np.ndarray  # (3,) camera center, world coords
    focal: float
    principal_point: tuple[float, float]
    image_size: tuple[int, int]


# --------------------------------------------------------------------------
# GT adapters (D7: self-contained — own all per-dataset scoring knowledge,
# no imports from the benchmark wrapper or third_party)
# --------------------------------------------------------------------------

# Both supported datasets share the camera-json schema; they differ only in
# frame organization (nvidia: static rig, frames "{cam}_{t:05d}" across 4
# cameras; iphone/DyCheck: one moving camera "0_{t:05d}" — note its poses
# are ARKit odometry, not survey-grade GT).
DATASETS = ("iphone", "nvidia")


camera_of = DyCheckSequence.camera_of  # the dataset's frame-name convention, not ours


def _evenly_spaced(items: list[str], count: int) -> list[str]:
    count = min(count, len(items))
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def _load_camera_json(path: Path) -> dict:
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise GTSchemaError(f"{path}: unreadable or invalid JSON: {exc}") from exc
    if not isinstance(blob, dict):
        raise GTSchemaError(f"{path}: top level is not an object")
    missing = [k for k in ("orientation", "position", "focal_length", "principal_point", "image_size") if k not in blob]
    if missing:
        raise GTSchemaError(f"{path}: missing keys: {missing}")
    return blob


def _gt_camera_from_json(frame_name: str, cam_json: Path, rgb_path: Path) -> GTCamera:
    blob = _load_camera_json(cam_json)
    R = np.asarray(blob["orientation"], dtype=np.float64)
    if R.shape != (3, 3):
        raise GTSchemaError(f"{cam_json}: orientation is not 3x3")
    # A GT rotation that is not orthonormal would silently corrupt every
    # downstream error number — reject it here, loudly.
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or not math.isclose(
        float(np.linalg.det(R)), 1.0, abs_tol=1e-4
    ):
        raise GTSchemaError(f"{cam_json}: orientation is not a rotation (RRᵀ≠I or det≠1)")
    center = np.asarray(blob["position"], dtype=np.float64)
    if center.shape != (3,):
        raise GTSchemaError(f"{cam_json}: position is not a 3-vector")
    pp = blob["principal_point"]
    size = blob["image_size"]
    return GTCamera(
        image_name=f"{frame_name}.png",
        source_frame=frame_name,
        rgb_path=rgb_path,
        R_w2c=R,
        center=center,
        focal=float(blob["focal_length"]),
        principal_point=(float(pp[0]), float(pp[1])),
        image_size=(int(size[0]), int(size[1])),
    )


def enumerate_frames(dataset: str, data_dir: Path) -> list[str]:
    """Sorted frame names that have BOTH an rgb/1x image and a camera json.

    1x is the resolution the GT intrinsics are expressed in (image_size in
    the camera jsons matches rgb/1x), so scoring collects 1x frames.
    """
    if dataset not in DATASETS:
        raise GTSchemaError(f"unknown dataset {dataset!r} (expected one of {DATASETS})")
    try:
        return DyCheckSequence(data_dir).frames(1)
    except DatasetLayoutError as exc:
        raise GTSchemaError(str(exc)) from exc


def select_frames(
    dataset: str,
    frames: list[str],
    *,
    timesteps: int = DEFAULT_TIMESTEPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> list[str]:
    """Default frame selection per dataset.

    nvidia: every camera × ``timesteps`` evenly spaced times. Static-rig
    note (D11/outside-voice): more timesteps per static camera adds
    near-duplicate images, so the default stays small and registration is
    always reported per unique CAMERA, never per frame.
    iphone: ``num_frames`` evenly spaced along the moving trajectory.
    """
    if dataset == "nvidia":
        by_cam: dict[str, list[str]] = {}
        for f in frames:
            by_cam.setdefault(camera_of(f), []).append(f)
        selected: list[str] = []
        for cam in sorted(by_cam):
            selected.extend(_evenly_spaced(sorted(by_cam[cam]), timesteps))
        return selected
    return _evenly_spaced(frames, num_frames)


def load_gt_cameras(dataset: str, data_dir: Path, frame_names: list[str]) -> list[GTCamera]:
    if not frame_names:
        raise GTSchemaError("no frames selected — nothing to score")
    cameras = []
    for frame in frame_names:
        seq = DyCheckSequence(data_dir)
        rgb, cam_json = seq.rgb_path(frame, 1), seq.camera_path(frame)
        if not rgb.is_file():
            raise GTSchemaError(f"selected frame {frame!r} has no rgb at {rgb}")
        if not cam_json.is_file():
            raise GTSchemaError(f"selected frame {frame!r} has no camera json at {cam_json}")
        cameras.append(_gt_camera_from_json(frame, cam_json, rgb))
    return cameras


def shared_intrinsics(cameras: list[GTCamera], rel_tol: float = 1e-4) -> GTCamera | None:
    """The representative camera when all GT intrinsics agree, else None.

    D12 limit: hloc applies image_options globally (one camera for the whole
    import), so GT-intrinsics injection is only expressible when the
    collected cameras share intrinsics. Callers must check this before
    requesting the known-intrinsics protocol.
    """
    if not cameras:
        raise GTSchemaError("shared_intrinsics needs at least one camera")
    first = cameras[0]
    for cam in cameras[1:]:
        if not math.isclose(cam.focal, first.focal, rel_tol=rel_tol):
            return None
        if not all(
            math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-6)
            for a, b in zip(cam.principal_point, first.principal_point)
        ) or cam.image_size != first.image_size:
            return None
    return first


# --------------------------------------------------------------------------
# Collector (D3: correspondence is constructed ONCE, at collection time,
# and written into gt_manifest.json — the scorer never re-derives it)
# --------------------------------------------------------------------------


def collect_frames(cameras: list[GTCamera], image_dir: Path) -> Path:
    """Materialize frames into ``image_dir`` and write the GT manifest.

    Frames are COPIED, never symlinked: pycolmap >= 3.13 resolves symlinks
    during import and stores the relative path THROUGH the link target
    (e.g. ``../../nvidia/Balloon1/rgb/1x/x.png``), which no longer matches
    hloc's listing-name h5 keys — every keypoint lookup then KeyErrors.
    The manifest is written NEXT TO the image dir, never inside it, because
    hloc/COLMAP sweep every file in the image dir.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    expected = {cam.image_name for cam in cameras}
    # Purge anything a previous (larger/different) collection left behind:
    # hloc sweeps EVERY file in the image dir, so a stale image would be
    # reconstructed and silently change the model the metrics describe.
    stale = [p for p in image_dir.iterdir() if p.is_file() and p.name not in expected]
    for path in stale:
        path.unlink()
    if stale:
        print(f"[collect] purged {len(stale)} stale file(s) from {image_dir}", flush=True)
    images: dict[str, dict] = {}
    for cam in cameras:
        if cam.image_name in images:
            raise GTSchemaError(f"duplicate collected image name {cam.image_name!r}")
        # Always copy — an exists-check would let a re-collection from a
        # DIFFERENT sequence (same frame names) silently score fresh GT
        # against stale images.
        dest = image_dir / cam.image_name
        shutil.copy2(cam.rgb_path, dest)
        images[cam.image_name] = {
            "source_frame": cam.source_frame,
            "R_w2c": cam.R_w2c.reshape(-1).tolist(),
            "center": cam.center.tolist(),
            "focal": cam.focal,
            "principal_point": list(cam.principal_point),
            "image_size": list(cam.image_size),
        }
    manifest_path = image_dir.parent / GT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "images": images}, indent=2) + "\n"
    )
    return manifest_path


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    blob = json.loads(manifest_path.read_text())
    if (
        not isinstance(blob, dict)
        or blob.get("schema") != MANIFEST_SCHEMA
        or "images" not in blob
    ):
        raise GTSchemaError(f"{manifest_path}: not a {MANIFEST_SCHEMA} manifest")
    return blob["images"]


# --------------------------------------------------------------------------
# Pose math (D6: hand-rolled, oracle-proven in tests; D11: pairwise primary)
# --------------------------------------------------------------------------


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, degrees, clamped at 0°/180°."""
    cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos)))))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) minimizing ‖dst - (s·R·src + t)‖².

    Umeyama (1991) with the determinant correction, so a reflection in the
    data cannot masquerade as a rotation.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("umeyama_sim3 needs matching (N>=3, 3) arrays")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / src.shape[0]
    if var_src <= 0.0:
        raise ValueError("umeyama_sim3: source points are coincident")
    s = float((D * np.diag(S)).sum() / var_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t


def center_conditioning(centers: np.ndarray) -> dict:
    """Singular-value shape of the centered camera centers (D11 guard).

    svd of an (N, 3) matrix yields only min(N, 3) singular values — ratios
    are padded so N < 3 degrades to "degenerate" instead of IndexError.
    """
    c = centers - centers.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    s1 = float(sv[0]) if sv.size and sv[0] > 0 else 0.0
    ratios = [float(v) / s1 if s1 > 0 else 0.0 for v in sv]
    ratios += [0.0] * (3 - len(ratios))
    return {
        "singular_ratios": ratios,
        "colinear": s1 == 0.0 or ratios[1] < _COLINEAR_RATIO,
        "planar": s1 == 0.0 or ratios[2] < _PLANAR_RATIO,
    }


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def pairwise_relative_errors(
    est: dict[str, tuple[np.ndarray, np.ndarray]],
    gt: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """PRIMARY metric (D11): gauge-free per-pair errors.

    For each camera pair (i, j), with world→camera rotations R and centers C:
    - relative rotation error: geodesic(R_i·R_jᵀ |est vs gt) — gauge-free.
    - baseline DIRECTION error: the baseline expressed in camera-i's frame,
      d = R_i·(C_j − C_i), is frame-independent; the angle between est and
      gt directions needs no alignment.
    - baseline RATIO error: |b_ij|/mean(|b|) compared est vs gt — scale-free.
    Pairs with near-zero GT baseline are excluded from direction/ratio stats
    and reported (static-rig duplicate timesteps would otherwise pollute).
    """
    names = sorted(set(est) & set(gt))
    if len(names) < 2:
        raise ValueError("pairwise_relative_errors needs at least 2 matched cameras")
    gt_baselines = {}
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            gt_baselines[(a, b)] = float(np.linalg.norm(gt[b][1] - gt[a][1]))

    # Two-pass classification: degeneracy is judged against a MEDIAN-scaled
    # threshold (a mean would itself be deflated by the zero baselines it is
    # trying to detect), and the ratio-normalization means are then computed
    # over NON-degenerate pairs only — otherwise GT contributes exact zeros
    # where est contributes small nonzero baselines, biasing every ratio.
    med_gt_b = float(np.median(list(gt_baselines.values()))) or 1.0
    nd_pairs = [
        pair for pair, b in gt_baselines.items()
        if b >= _MIN_BASELINE_FRACTION * med_gt_b
    ]
    est_norms = {
        pair: float(np.linalg.norm(est[pair[1]][1] - est[pair[0]][1]))
        for pair in gt_baselines
    }
    if nd_pairs:
        mean_gt_b = float(np.mean([gt_baselines[p] for p in nd_pairs])) or 1.0
        mean_est_b = float(np.mean([est_norms[p] for p in nd_pairs])) or 1.0
    else:
        mean_gt_b = mean_est_b = 1.0

    rot_nd, rot_degen, direction, ratio, degenerate = [], [], [], [], []
    pairs = []
    for (a, b), gt_b in gt_baselines.items():
        Ra_e, Ca_e = est[a]
        Rb_e, Cb_e = est[b]
        Ra_g, Ca_g = gt[a]
        Rb_g, Cb_g = gt[b]
        rot_err = geodesic_deg(Ra_e @ Rb_e.T, Ra_g @ Rb_g.T)
        entry = {"pair": [a, b], "rot_deg": rot_err}
        if (a, b) not in set(nd_pairs):
            # Same-position pairs (e.g. one static camera at two timesteps)
            # measure registration REPEATABILITY, not rig geometry — kept
            # out of the headline stats so they cannot flatter the verdict.
            rot_degen.append(rot_err)
            degenerate.append([a, b])
            entry["degenerate_baseline"] = True
        else:
            rot_nd.append(rot_err)
            # Direction error is computed in BOTH anchor frames and averaged:
            # a single-anchor version leaks only camera a's rotation error
            # into the stat and never camera b's.
            dir_errs = []
            est_degen = False
            for (Rx_e, Cx_e, Cy_e, Rx_g, Cx_g, Cy_g) in (
                (Ra_e, Ca_e, Cb_e, Ra_g, Ca_g, Cb_g),
                (Rb_e, Cb_e, Ca_e, Rb_g, Cb_g, Ca_g),
            ):
                d_e = Rx_e @ (Cy_e - Cx_e)
                d_g = Rx_g @ (Cy_g - Cx_g)
                n_e, n_g = np.linalg.norm(d_e), np.linalg.norm(d_g)
                if n_e > 0 and n_g > 0:
                    cos = float(d_e @ d_g / (n_e * n_g))
                    dir_errs.append(
                        math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                    )
                else:
                    est_degen = True
            if dir_errs and not est_degen:
                dir_err = float(np.mean(dir_errs))
                direction.append(dir_err)
                entry["trans_dir_deg"] = dir_err
            else:
                # The GT baseline is healthy but the ESTIMATE collapsed two
                # cameras onto one point — a catastrophic failure that must
                # be flagged, never silently dropped from direction stats.
                entry["est_degenerate_baseline"] = True
            ratio_err = abs(est_norms[(a, b)] / mean_est_b - gt_b / mean_gt_b)
            ratio.append(ratio_err)
            entry["baseline_ratio_err"] = ratio_err
        pairs.append(entry)
    result = {
        # headline stats: non-degenerate (cross-position) pairs only
        "rot_deg": _stats(rot_nd) if rot_nd else None,
        "rot_deg_degenerate_pairs": _stats(rot_degen) if rot_degen else None,
        "trans_dir_deg": _stats(direction) if direction else None,
        "baseline_ratio_err": _stats(ratio) if ratio else None,
        "degenerate_pairs": degenerate,
        "est_degenerate_pairs": [
            p["pair"] for p in pairs if p.get("est_degenerate_baseline")
        ],
        "pairs": pairs,
    }
    return result


def sim3_ape(
    est: dict[str, tuple[np.ndarray, np.ndarray]],
    gt: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """SECONDARY metric: Sim(3)-aligned absolute pose errors (guarded).

    At small N the 7-DOF alignment absorbs real error (D11) — these numbers
    are reported alongside, never instead of, the pairwise metrics.
    """
    names = sorted(set(est) & set(gt))
    if len(names) < 3:
        return {"skipped": f"needs >=3 matched cameras, have {len(names)}"}
    est_c = np.stack([est[n][1] for n in names])
    gt_c = np.stack([gt[n][1] for n in names])
    cond = center_conditioning(gt_c)
    if cond["colinear"]:
        return {"skipped": "GT centers are near-colinear — Sim(3) is degenerate", "conditioning": cond}
    # The ESTIMATE can be degenerate too (collapsed/colinear recovered
    # centers with healthy GT) — fitting through it would report a garbage
    # alignment as a normal result, and exactly-coincident est centers
    # would raise out of umeyama AFTER the expensive SfM already ran.
    est_cond = center_conditioning(est_c)
    if est_cond["colinear"]:
        return {
            "skipped": "ESTIMATED centers are near-colinear/collapsed — "
            "Sim(3) fit would be garbage",
            "conditioning": cond,
            "est_conditioning": est_cond,
        }
    try:
        s, R, t = umeyama_sim3(est_c, gt_c)
    except ValueError as exc:
        return {"skipped": f"alignment failed: {exc}", "conditioning": cond}
    rot, trans = [], []
    for n in names:
        R_e, C_e = est[n]
        R_g, C_g = gt[n]
        # world transforms as x_gt ≈ s·R·x_est + t ⇒ w2c rotations relate by
        # R_g ≈ R_e·Rᵀ; centers map directly through the similarity.
        rot.append(geodesic_deg(R_e @ R.T, R_g))
        trans.append(float(np.linalg.norm(s * R @ C_e + t - C_g)))
    return {
        "rot_deg": _stats(rot),
        "trans": _stats(trans),
        "scale": s,
        "conditioning": cond,
    }


# --------------------------------------------------------------------------
# Scoring orchestration (D5: fragmentation is a result, not an error)
# --------------------------------------------------------------------------


def _est_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    R = qvec_to_rotmat(np.asarray(record["qvec"], dtype=np.float64))
    t = np.asarray(record["tvec"], dtype=np.float64)
    return R, -R.T @ t


def score_against_gt(model: ParsedColmapModel, manifest_images: dict[str, dict]) -> dict:
    """Join recovered poses with the GT manifest and compute both metric families."""
    gt_all: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cam_of: dict[str, str] = {}
    for name, blob in manifest_images.items():
        gt_all[name] = (
            np.asarray(blob["R_w2c"], dtype=np.float64).reshape(3, 3),
            np.asarray(blob["center"], dtype=np.float64),
        )
        cam_of[name] = camera_of(blob["source_frame"])

    registered = set(model.image_records) & set(gt_all)
    unregistered = sorted(set(gt_all) - set(model.image_records))
    extraneous = sorted(set(model.image_records) - set(gt_all))

    # parse_colmap_text_model partitions every image into a component, so
    # registered images always intersect at least one component; an empty
    # list here means nothing registered, and the <3 guard below reports it.
    components = [sorted(set(c) & registered) for c in model.connected_components]
    components = sorted((c for c in components if c), key=len, reverse=True)

    unique_cams = sorted({cam_of[n] for n in gt_all})
    registered_cams = sorted({cam_of[n] for n in registered})

    result = {
        "registration": {
            # per unique CAMERA, never per frame (static-rig duplicates
            # must not inflate this number)
            "cameras_total": len(unique_cams),
            "cameras_registered": len(registered_cams),
            "images_total": len(gt_all),
            "images_registered": len(registered),
            "unregistered_images": unregistered,
            "extraneous_images": extraneous,
        },
        "fragmentation": {
            "num_components": len(components),
            "component_sizes": [len(c) for c in components],
        },
        "alignable": False,
        "metrics": None,
    }
    if not components or len(components[0]) < 3:
        result["error"] = (
            "largest connected component has fewer than 3 matched cameras — "
            "nothing alignable to score"
        )
        return result

    scored = components[0]
    est = {n: _est_pose(model.image_records[n]) for n in scored}
    gt = {n: gt_all[n] for n in scored}
    per_camera: dict[str, list[str]] = {}
    for n in scored:
        per_camera.setdefault(cam_of[n], []).append(n)

    result["alignable"] = True
    result["scored_component_size"] = len(scored)
    result["scored_coverage"] = len(scored) / len(gt_all)
    result["metrics"] = {
        "pairwise": pairwise_relative_errors(est, gt),
        "sim3_ape": sim3_ape(est, gt),
    }
    result["per_camera_images"] = {cam: sorted(v) for cam, v in per_camera.items()}
    return result

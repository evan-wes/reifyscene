"""FP6b step-1 validation: COLMAP-SfM pose rung as a native swap vs GT extrinsics.

Productionizes the FP6b egomotion spike through the shipped
`ColmapPoseInitializer.reconstruct`: monocular hloc+COLMAP with intrinsics fixed from the
GT focal, scored vs the dataset GT camera poses. Reports registered-frame fraction +
gauge-free pairwise-relative rotation (the M0 metric that caught the calibration arc's
silent cam-12 miss) + Sim(3)-aligned APE (monocular SfM is up-to-scale).

Run (GPU; hloc/pycolmap via setup_reifyscene.sh --with-calibration):
    python scripts/fp6b_validate_colmap.py --dataset-root dycheck_iphone/spin --max-frames 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from reifyscene.common import data_dir
from reifyscene.geometry.colmap_wrapper import ColmapPoseInitializer
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def umeyama(X: np.ndarray, Y: np.ndarray):
    """Sim(3): (s, R, t) with Y ~= s R X + t."""
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    Sigma = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((Xc ** 2).sum() / len(X)))
    return s, R, muY - s * R @ muX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dycheck_iphone/spin",
                    help="the sequence's name in the data directory's datasets/")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--work-dir", default="fp6b_colmap_validate",
                    help="a name in the data directory's cache/")
    args = ap.parse_args()
    args.dataset_root = str(data_dir.path(data_dir.DATASETS, args.dataset_root))

    ds = DatasetGeometryInitializer({"dataset_root": args.dataset_root, "factor": args.factor, "split": args.split})
    specs = ds.split_frames()
    if len(specs) > args.max_frames:
        idx = np.linspace(0, len(specs) - 1, args.max_frames).astype(int)
        specs = [specs[i] for i in idx]
    names = [ds.frame_name(c, t) for c, t in specs]

    # intrinsics fixed from the GT focal (calibration-arc lesson)
    intr, _, _ = ds.load_camera(names[0])
    cam_params = f"{intr.fx},{intr.fy},{intr.cx},{intr.cy}"

    # materialize index-named frames for the shipped reconstruct() core
    work = data_dir.path(data_dir.CACHE, args.work_dir)
    img_dir = work / "images"
    if img_dir.exists():
        import shutil
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.fromarray(ds.load_rgb(name)).save(img_dir / f"{i:06d}.png")

    print(f"\n=== FP6b step-1 COLMAP pose rung vs GT ({Path(args.dataset_root).name}, {len(names)} frames, fixed intrinsics) ===")
    rung = ColmapPoseInitializer({"camera_params": cam_params, "fix_intrinsics": True, "colmap_work_dir": args.work_dir})
    poses = rung.reconstruct(img_dir)  # {index: (T_world_cam, intr)}

    common = [i for i in range(len(names)) if i in poses]
    print(f"registered {len(common)}/{len(names)} ({100 * len(common) / len(names):.0f}%)")
    if len(common) < 4:
        print("too few registered frames to score.")
        return

    def R_and_C(t_world_cam):
        w2c = np.linalg.inv(t_world_cam)
        return w2c[:3, :3], np.asarray(t_world_cam)[:3, 3]

    est = {i: R_and_C(poses[i][0]) for i in common}
    gt = {i: R_and_C(ds.load_camera(names[i])[1]) for i in common}

    pair = [geodesic_deg(est[b][0] @ est[a][0].T, gt[b][0] @ gt[a][0].T) for a, b in zip(common[:-1], common[1:])]
    Xc = np.stack([est[i][1] for i in common])
    Yc = np.stack([gt[i][1] for i in common])
    s, R, t = umeyama(Xc, Yc)
    rot = [geodesic_deg(R @ est[i][0].T, gt[i][0].T) for i in common]
    trans = [float(np.linalg.norm(s * R @ est[i][1] + t - gt[i][1])) for i in common]
    extent = float(np.linalg.norm(Yc.max(0) - Yc.min(0))) or 1.0

    print(f"pairwise-rel rotation  med/max = {np.median(pair):.2f} / {np.max(pair):.2f} deg   (gauge-free, the honest motion signal)")
    print(f"Sim(3)-APE  rotation med = {np.median(rot):.2f} deg   translation med = {100 * np.median(trans) / extent:.1f}% of extent")
    print(f"(Sim3 scale = {s:.4g}; monocular SfM is up-to-scale — resolved by COLMAP<->MoGe reconciliation at step 5.)")


if __name__ == "__main__":
    main()

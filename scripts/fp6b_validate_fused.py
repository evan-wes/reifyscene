"""FP6b step-2 validation: fused COLMAP poses + MoGe depth, scale-reconciled, vs GT.

The payoff of the pose/depth split (design §4A.1): COLMAP (scale-free egomotion) + MoGe
(metric depth) fuse into **GT-free, VGGT-free metric geometry** via a single global scale.
This script runs the shipped rungs + reconciler on a dataset and answers three questions:

1. **Does the per-scene scale offset collapse to ONE global scale?** FP6a needed a *per-frame*
   oracle scale to align MoGe to GT (spin +13% / windmill −35%). Here a single reconciled
   scale explains all frames; the per-frame residual spread is what remains for F2. A **tight**
   residual band is the headline result.
2. **Are the reconciled poses metric?** Scored with a **rigid SE(3)** alignment to GT
   (rotation+translation, scale fixed = 1) — unlike step 1's Sim(3), which absorbed scale.
   The residual Sim(3) scale vs GT (reported alongside) ≈ MoGe's own per-scene bias — the part
   F2's person-anchored scale later removes, *not* a reconciliation failure.
3. Depth vs GT for context (identical to FP6a — reconciliation moves poses, not depth).

Reuses the shipped `ColmapPoseInitializer.reconstruct`, `MoGeInitializer.infer_frame`, and
`room4d.geometry.scale_reconcile` so it exercises production code, not a re-implementation.

Run (GPU; hloc/pycolmap + MoGe via setup_room4d.sh --with-calibration --with-moge):
    python scripts/fp6b_validate_fused.py --dataset-root ../data/shape_of_motion/iphone/spin --max-frames 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from room4d.geometry.colmap_wrapper import ColmapPoseInitializer
from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.geometry.moge_wrapper import MoGeInitializer
from room4d.geometry.scale_reconcile import extract_colmap_observations, solve_global_scale


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


def rigid_align(X: np.ndarray, Y: np.ndarray):
    """SE(3): (R, t) with Y ~= R X + t (scale FIXED = 1) — Kabsch."""
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, _, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, muY - R @ muX


def R_and_C(T_world_cam: np.ndarray):
    w2c = np.linalg.inv(T_world_cam)
    return w2c[:3, :3], np.asarray(T_world_cam)[:3, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../data/shape_of_motion/iphone/spin")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-obs", type=int, default=64)
    ap.add_argument("--work-dir", default="/tmp/fp6b_fused_validate")
    args = ap.parse_args()

    ds = DatasetGeometryInitializer({"dataset_root": args.dataset_root, "factor": args.factor, "split": args.split})
    specs = ds.split_frames()
    if len(specs) > args.max_frames:
        idx = np.linspace(0, len(specs) - 1, args.max_frames).astype(int)
        specs = [specs[i] for i in idx]
    names = [ds.frame_name(c, t) for c, t in specs]

    intr_gt0, _, _ = ds.load_camera(names[0])
    cam_params = f"{intr_gt0.fx},{intr_gt0.fy},{intr_gt0.cx},{intr_gt0.cy}"

    work = Path(args.work_dir)
    img_dir = work / "images"
    if img_dir.exists():
        import shutil
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.fromarray(ds.load_rgb(name)).save(img_dir / f"{i:06d}.png")

    print(f"\n=== FP6b step-2 fused (COLMAP poses + MoGe depth, reconciled) vs GT "
          f"({Path(args.dataset_root).name}, {len(names)} frames) ===")

    # ---- pose channel: COLMAP with fixed intrinsics -------------------------
    rung = ColmapPoseInitializer({"camera_params": cam_params, "fix_intrinsics": True, "colmap_work_dir": str(work)})
    poses = rung.reconstruct(img_dir)                 # {index: (T_world_cam, intr)}
    model_dir = rung._last_model_dir
    common = sorted(i for i in range(len(names)) if i in poses)
    print(f"COLMAP registered {len(common)}/{len(names)} ({100 * len(common) / len(names):.0f}%)")
    if len(common) < 4 or model_dir is None:
        print("too few registered frames / no sparse model to fuse.")
        return

    # ---- depth channel: MoGe per registered frame ---------------------------
    moge = MoGeInitializer({"device": args.device})
    depth_by_key, mask_by_key, gt_depth_by_key = {}, {}, {}
    for i in common:
        res = moge.infer_frame(ds.load_rgb(names[i]))
        key = f"f:{i}"
        depth_by_key[key] = res["depth"].astype(np.float32)
        if res["mask"] is not None:
            mask_by_key[key] = res["mask"]
        gt_depth_by_key[key] = np.squeeze(ds.load_depth(names[i])).astype(np.float32)

    # ---- reconcile the single global scale ----------------------------------
    index_to_key = {i: f"f:{i}" for i in common}
    obs = extract_colmap_observations(model_dir, index_to_key)
    rec = solve_global_scale(obs, depth_by_key, min_obs=args.min_obs, mask_by_key=mask_by_key or None)
    print(f"\n[1] scale reconciliation: {rec.note()}")
    print(f"    per-frame residual  median={rec.residual_median:.3f}  "
          f"p10/p90=[{rec.residual_p10:.3f}, {rec.residual_p90:.3f}]  "
          f"(tight band → one global scale suffices; spread → F2)")

    s = rec.scale
    est_C = {i: R_and_C(poses[i][0]) for i in common}          # (R_wc, C) scale-free
    est_C_metric = np.stack([s * est_C[i][1] for i in common])  # metric camera centers
    est_R = [est_C[i][0] for i in common]
    gt = {i: R_and_C(ds.load_camera(names[i])[1]) for i in common}
    gt_C = np.stack([gt[i][1] for i in common])
    gt_R = [gt[i][0] for i in common]

    # ---- [2] pose accuracy: metric SE(3) vs step-1's Sim(3) -----------------
    R_se3, t_se3 = rigid_align(est_C_metric, gt_C)
    trans_se3 = [float(np.linalg.norm(R_se3 @ est_C_metric[k] + t_se3 - gt_C[k])) for k in range(len(common))]
    rot_se3 = [geodesic_deg(R_se3 @ est_R[k].T, gt_R[k].T) for k in range(len(common))]
    s_sim, R_sim, t_sim = umeyama(est_C_metric, gt_C)
    trans_sim = [float(np.linalg.norm(s_sim * R_sim @ est_C_metric[k] + t_sim - gt_C[k])) for k in range(len(common))]
    extent = float(np.linalg.norm(gt_C.max(0) - gt_C.min(0))) or 1.0
    print(f"\n[2] pose vs GT")
    print(f"    metric SE(3)-APE  rot med={np.median(rot_se3):.2f} deg  "
          f"trans med={100 * np.median(trans_se3) / extent:.1f}% of extent   (scale fixed=1 — the honest metric test)")
    print(f"    Sim(3)-APE        trans med={100 * np.median(trans_sim) / extent:.1f}% of extent  "
          f"residual scale vs GT={s_sim:.3f}  (≈ MoGe per-scene bias → removed by F2 person-anchor)")

    # ---- [3] depth vs GT for context ----------------------------------------
    absrel, absrel_a, scales = [], [], []
    for key in depth_by_key:
        pred, g = depth_by_key[key], gt_depth_by_key[key]
        if pred.shape != g.shape:
            import cv2
            pred = cv2.resize(pred, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(g) & (g > 1e-3) & np.isfinite(pred) & (pred > 1e-3)
        m = mask_by_key.get(key)
        if m is not None:
            if m.shape != g.shape:
                import cv2
                m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid &= m > 0.5
        if int(valid.sum()) < 100:
            continue
        sc = float(np.median(g[valid] / pred[valid]))
        absrel.append(float(np.mean(np.abs(pred[valid] - g[valid]) / g[valid])))
        absrel_a.append(float(np.mean(np.abs(pred[valid] * sc - g[valid]) / g[valid])))
        scales.append(sc)
    if absrel:
        print(f"\n[3] MoGe depth vs GT (context; unchanged from FP6a)")
        print(f"    metric absRel={np.mean(absrel):.3f}   scale-aligned absRel*={np.mean(absrel_a):.3f}   "
              f"median metric scale={np.median(scales):.3f}")


if __name__ == "__main__":
    main()

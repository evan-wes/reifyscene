"""FP6b step-3 validation: does size-gated dyn_mask reduce COLMAP's degenerate-draw rate?

Step 2 found monocular COLMAP is intermittently degenerate on large-dynamic clips. The
step-3 hypothesis (from the spike, with oracle masks): masking the dynamic region is a
**size-gated robustness** help. This tests it with the REAL detector rung + the shipped
health-check:

1. Per sequence, compute the detector rung's median dynamic fraction and whether it clears
   the size-gate (spin/person → large, should mask; windmill/pinwheel → small, gated off).
2. Run COLMAP N times per arm (unmasked vs detector-masked) and report the **degenerate-draw
   rate** (flagged by the GT-free per-stream health-check) + pairwise-rel rotation tail +
   Sim(3)-APE. Masking should lower the degenerate rate on spin; windmill should be neutral.

Reuses the shipped `DetectorDynMask.mask_frame`, `ColmapPoseInitializer.reconstruct`, and
`assess_pose_health`. GPU (Mask R-CNN + hloc/COLMAP).

Run:
    python scripts/fp6b_validate_dynmask.py --dataset-root ../data/shape_of_motion/iphone/spin --runs 5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from room4d.common.types import CameraIntrinsics, CameraPose, TimedCameraState
from room4d.geometry.colmap_wrapper import ColmapPoseInitializer
from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.geometry.reconstruction_health import assess_pose_health
from room4d.perception.dyn_mask.rungs import DetectorDynMask


def geodesic_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def umeyama(X, Y):
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((Xc ** 2).sum() / len(X)))
    return s, R, muY - s * R @ muX


def _materialize(ds, names, masks, img_dir):
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        rgb = ds.load_rgb(name)
        if masks is not None:
            rgb = rgb.copy()
            rgb[np.asarray(masks[i]) > 0.5] = 0
        Image.fromarray(rgb).save(img_dir / f"{i:06d}.png")


def _score(poses, ds, names):
    common = sorted(i for i in range(len(names)) if i in poses)
    if len(common) < 4:
        return None
    intr = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0)
    states = {"cam": [TimedCameraState("cam", float(i), intr, CameraPose(T_world_cam=poses[i][0])) for i in common]}
    health = assess_pose_health(states)
    def RC(T):
        w2c = np.linalg.inv(T)
        return w2c[:3, :3], w2c[:3, 3] * 0 + np.asarray(T)[:3, 3]
    est = {i: RC(poses[i][0]) for i in common}
    gt = {i: (np.linalg.inv(ds.load_camera(names[i])[1])[:3, :3], np.asarray(ds.load_camera(names[i])[1])[:3, 3]) for i in common}
    pair = [geodesic_deg(est[b][0] @ est[a][0].T, gt[b][0] @ gt[a][0].T) for a, b in zip(common[:-1], common[1:])]
    Xc = np.stack([est[i][1] for i in common]); Yc = np.stack([gt[i][1] for i in common])
    s, R, t = umeyama(Xc, Yc)
    trans = [float(np.linalg.norm(s * R @ est[i][1] + t - gt[i][1])) for i in common]
    extent = float(np.linalg.norm(Yc.max(0) - Yc.min(0))) or 1.0
    return {
        "reg": len(common), "healthy": health["healthy"],
        "pair_med": float(np.median(pair)), "pair_max": float(np.max(pair)),
        "ape_trans_pct": 100 * float(np.median(trans)) / extent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../data/shape_of_motion/iphone/spin")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gate", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-dir", default="/tmp/fp6b_dynmask")
    args = ap.parse_args()

    ds = DatasetGeometryInitializer({"dataset_root": args.dataset_root})
    specs = ds.split_frames()
    if len(specs) > args.max_frames:
        specs = [specs[i] for i in np.linspace(0, len(specs) - 1, args.max_frames).astype(int)]
    names = [ds.frame_name(c, t) for c, t in specs]
    intr0, _, _ = ds.load_camera(names[0])
    cam_params = f"{intr0.fx},{intr0.fy},{intr0.cx},{intr0.cy}"
    seq = Path(args.dataset_root).name
    work = Path(args.work_dir) / seq

    print(f"\n=== FP6b step-3 dyn_mask vs COLMAP instability ({seq}, {len(names)} frames, {args.runs} runs/arm) ===")

    # [1] detector rung: masks + gate decision
    masker = DetectorDynMask({"device": args.device})
    masks, fracs = {}, []
    for i, name in enumerate(names):
        m, f = masker.mask_frame(ds.load_rgb(name))
        masks[i] = m
        fracs.append(f)
    masker._release()
    med_frac = float(np.median(fracs))
    gated_on = med_frac >= args.gate
    print(f"[1] detector median dynamic fraction = {med_frac:.3f}  → size-gate ({args.gate:.2f}): "
          f"{'MASK (large-dynamic)' if gated_on else 'UNMASKED (small-dynamic, gated off)'}")

    # [2] N runs per arm
    for arm, use_masks in (("unmasked", None), ("detector-masked", masks)):
        degen = 0
        pair_maxes, apes = [], []
        for r in range(args.runs):
            _materialize(ds, names, use_masks, work / arm / "images")
            rung = ColmapPoseInitializer({"camera_params": cam_params, "fix_intrinsics": True,
                                          "colmap_work_dir": str(work / arm)})
            poses = rung.reconstruct(work / arm / "images")
            sc = _score(poses, ds, names)
            if sc is None:
                degen += 1
                continue
            if not sc["healthy"]:
                degen += 1
            pair_maxes.append(sc["pair_max"])
            apes.append(sc["ape_trans_pct"])
        pm = f"{np.median(pair_maxes):.2f}" if pair_maxes else "n/a"
        ap_ = f"{np.median(apes):.1f}" if apes else "n/a"
        print(f"[2] {arm:<16} degenerate {degen}/{args.runs}   median pairwise-max {pm}°   median Sim3-APE trans {ap_}%")

    print("\nExpect: large-dynamic (spin) → masking lowers the degenerate rate; "
          "small-dynamic (windmill) → gated off, neutral.")


if __name__ == "__main__":
    main()

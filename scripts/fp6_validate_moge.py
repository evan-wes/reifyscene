"""FP6a validation: MoGe-2 metric depth as a native swap vs the GT-depth oracle.

MoGe-2 is the FP6a depth-ladder rung (design §4A.1): a single-image *metric* point
map. This scores its depth against the dataset GT depth on spin / paper-windmill — a
direct depth-map comparison (no back-projection, no poses needed), the systematic
version of the native-swap "score vs the oracle" habit. It reports both the METRIC
error (tests MoGe's absolute-scale claim) and the SCALE-INVARIANT error (median-ratio
aligned, tests depth *structure*), plus focal agreement.

Reuses `DatasetGeometryInitializer`'s loaders for GT rgb/depth/camera and the actual
`MoGeInitializer.infer_frame` inference core, so it exercises the shipped rung.

Run (GPU; weights via `setup_reifyscene.sh --with-moge`):
    python scripts/fp6_validate_moge.py --dataset-root dycheck_iphone/spin --max-frames 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from reifyscene.common import data_dir
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.geometry.moge_wrapper import MoGeInitializer


def _depth_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    """(AbsRel, RMSE, δ<1.25) over valid pixels."""
    p, g = pred[valid], gt[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    delta = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    return absrel, rmse, delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='dycheck_iphone/spin',
                    help='a sequence name in datasets/')
    ap.add_argument('--factor', type=int, default=1)
    ap.add_argument('--split', default='train')
    ap.add_argument('--camera', default=None, help='restrict to one camera id (e.g. 0)')
    ap.add_argument('--max-frames', type=int, default=12)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None,
                    help='optional JSON for per-frame + aggregate results: a name in runs/')
    args = ap.parse_args()
    dataset = args.dataset_root
    args.dataset_root = str(data_dir.path(data_dir.DATASETS, dataset))

    ds = DatasetGeometryInitializer({'dataset_root': args.dataset_root, 'factor': args.factor, 'split': args.split})
    specs = ds.split_frames()
    if args.camera is not None:
        specs = [(c, t) for (c, t) in specs if str(c) == str(args.camera)]
    specs = specs[:: max(1, args.stride)][: args.max_frames]
    if not specs:
        raise SystemExit(f"No frames for split '{args.split}' under {args.dataset_root}")

    moge = MoGeInitializer({'device': args.device})

    per_frame: list[dict] = []
    print(f"\n=== FP6a MoGe-2 native-swap vs GT depth ({Path(args.dataset_root).name}, up to {len(specs)} frames) ===")
    print(f"{'frame':<16}{'absRel':>8}{'d<1.25':>8}  |{'absRel*':>9}{'d<1.25*':>9}{'scale':>8}{'foc':>7}")
    for cam, time in specs:
        name = ds.frame_name(cam, time)
        rgb = ds.load_rgb(name)
        gt = np.squeeze(ds.load_depth(name)).astype(np.float32)
        intr_gt, _, _ = ds.load_camera(name)
        res = moge.infer_frame(rgb)
        pred = res['depth'].astype(np.float32)
        mask = res['mask']
        if pred.shape != gt.shape:  # safety: align MoGe output to the GT grid
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            if mask is not None:
                mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(gt) & (gt > 1e-3) & np.isfinite(pred) & (pred > 1e-3)
        if mask is not None:
            valid &= mask > 0.5
        if int(valid.sum()) < 100:
            continue
        absrel, rmse, delta = _depth_metrics(pred, gt, valid)
        scale = float(np.median(gt[valid] / pred[valid]))  # best single scalar aligning pred->gt
        absrel_a, rmse_a, delta_a = _depth_metrics(pred * scale, gt, valid)
        foc = float(res['intrinsics'].fx / intr_gt.fx)
        per_frame.append({
            'frame': name, 'absrel': absrel, 'delta': delta, 'rmse': rmse,
            'absrel_aligned': absrel_a, 'delta_aligned': delta_a, 'rmse_aligned': rmse_a,
            'scale': scale, 'focal_ratio': foc, 'valid_px': int(valid.sum()),
        })
        print(f"{name:<16}{absrel:>8.3f}{delta:>8.3f}  |{absrel_a:>9.3f}{delta_a:>9.3f}{scale:>8.3f}{foc:>7.3f}")

    if not per_frame:
        raise SystemExit("No frames had enough valid pixels to score.")

    keys = ('absrel', 'delta', 'rmse', 'absrel_aligned', 'delta_aligned', 'rmse_aligned', 'scale', 'focal_ratio')
    agg = {k: float(np.mean([r[k] for r in per_frame])) for k in keys}
    print("-" * 66)
    print(f"{f'MEAN (n={len(per_frame)})':<16}{agg['absrel']:>8.3f}{agg['delta']:>8.3f}  |"
          f"{agg['absrel_aligned']:>9.3f}{agg['delta_aligned']:>9.3f}{agg['scale']:>8.3f}{agg['focal_ratio']:>7.3f}")
    print("\nmetric absRel/d test the absolute-scale claim; *-aligned (median-ratio) test depth structure.")
    print(f"median metric scale = {agg['scale']:.3f} (->1.0 = MoGe metric ~ GT); focal ratio = {agg['focal_ratio']:.3f}")

    if args.out:
        data_dir.path(data_dir.RUNS, args.out).write_text(
            json.dumps({'dataset': dataset, 'per_frame': per_frame, 'aggregate': agg}, indent=2))
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()

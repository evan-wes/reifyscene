"""Validate the native 3D-track lift against SoM's cached lifted tracks.

Runs reifyscene's native lift (``reifyscene.lifting.native_track_lift``) on a DyCheck
sequence and compares the foreground trajectories + scene normalization to the
Shape-of-Motion cache that produced the *borrowed* init bundle
(``flow3d_preprocessed/cache/tracks_3d_40000.pth`` + ``scene_norm_dict.pth``).

The up-axis rotation of the scene transform is deterministic (camera-derived) and
should match the cache near-exactly; the scene scale, track count, and xyz extent
come from a random track subsample, so they're checked for agreement within a
tolerance rather than equality. This is the PR-A acceptance check; the full
mPSNR≈20.7 validation lands with the initializer in PR-B.

Example (reifyscene env):
    python scripts/check_native_lift.py \
        --dataset-root ../data/shape_of_motion/iphone/paper-windmill \
        --num-fg 40000 --num-bg 100000 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from reifyscene.lifting.native_track_lift import build_native_init_inputs


def _bbox(xyz: torch.Tensor) -> str:
    lo = xyz.reshape(-1, 3).min(0).values
    hi = xyz.reshape(-1, 3).max(0).values
    return f"min={lo.tolist()} max={hi.tolist()}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate native track lift vs SoM cache")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--num-fg", type=int, default=40_000)
    ap.add_argument("--num-bg", type=int, default=100_000)
    ap.add_argument("--camera-type", default="refined", choices=["refined", "original"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"native lift: {args.dataset_root} (factor={args.factor}, num_fg={args.num_fg}, camera={args.camera_type})")
    tracks, bg, cameras, norm = build_native_init_inputs(
        args.dataset_root, factor=args.factor, num_fg=args.num_fg,
        num_bg=args.num_bg, camera_type=args.camera_type, device=args.device, seed=args.seed,
    )
    assert tracks.check_sizes(), "Tracks3D failed check_sizes()"
    assert bg.check_sizes(), "StaticPoints3D failed check_sizes()"

    print("\n-- native --")
    print(f"  fg tracks : N={tracks.num_tracks}  T={tracks.num_frames}")
    print(f"  bg points : M={bg.num_points}")
    print(f"  scene scale: {norm.scale:.6f}")
    print(f"  fg xyz    : {_bbox(tracks.xyz)}")
    print(f"  train cams: {len(cameras['train'].frame_names)}  val cams: {len(cameras['val'].frame_names)}")

    cache_dir = Path(args.dataset_root) / "flow3d_preprocessed" / "cache"
    snd_path = cache_dir / "scene_norm_dict.pth"
    trk_path = cache_dir / f"tracks_3d_{args.num_fg}.pth"
    if not (snd_path.exists() and trk_path.exists()):
        print(f"\n[skip] no SoM cache at {cache_dir} (lift shapes OK; nothing to compare).")
        return

    snd = torch.load(snd_path, weights_only=False)
    cached_scale = float(snd["scale"])
    cached_transfm = torch.as_tensor(np.asarray(snd["transfm"]), dtype=torch.float32)
    cached = torch.load(trk_path, weights_only=False)
    cached_xyz = cached["tracks_3d"].float()

    R_diff = float((norm.transfm[:3, :3] - cached_transfm[:3, :3]).abs().max())
    t_diff = float((norm.transfm[:3, 3] - cached_transfm[:3, 3]).abs().max())
    scale_rel = abs(norm.scale - cached_scale) / max(cached_scale, 1e-9)
    count_ratio = tracks.num_tracks / max(cached_xyz.shape[0], 1)

    print("\n-- cached (SoM) --")
    print(f"  fg tracks : N={cached_xyz.shape[0]}  T={cached_xyz.shape[1]}")
    print(f"  scene scale: {cached_scale:.6f}")
    print(f"  fg xyz    : {_bbox(cached_xyz)}")

    print("\n-- comparison --")
    print(f"  transfm R max|diff| : {R_diff:.2e}   (deterministic -> expect ~0)")
    print(f"  transfm t max|diff| : {t_diff:.4f}   (random subsample -> small)")
    print(f"  scene scale rel diff: {scale_rel:.3f}")
    print(f"  track count ratio   : {count_ratio:.3f}  ({tracks.num_tracks} vs {cached_xyz.shape[0]})")

    checks = {
        "rotation matches cache (R<1e-3)": R_diff < 1e-3,
        "scale within 25%": scale_rel < 0.25,
        "track count within 2x": 0.5 <= count_ratio <= 2.0,
        "T == cached T": tracks.num_frames == cached_xyz.shape[1],
    }
    print("\n-- checks --")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{'ALL PASS' if all(checks.values()) else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()

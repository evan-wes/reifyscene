"""Dump room4d's *native* motion init to an .npz (parity with the SoM borrow).

Builds the motion init natively (`NativeMotionInitializer`: room4d lifts the 2D
tracks to 3D, then clusters + Procrustes + places Gaussians) and writes the same
bundle layout as ``scripts/borrow_som_motion_init.py`` — so the native bundle can
be inspected, diffed against the borrowed one, and loaded back through
``render/gaussian4d/motion_init.py::load_borrowed_init``.

Unlike the borrow script, this runs in the plain ``room4d`` env (no SoM reference
env, no cuML): the only borrowed input is the 2D point tracker output.

Example (room4d env):
    python scripts/native_motion_init.py \
        --dataset-root ../data/shape_of_motion/iphone/paper-windmill \
        --num-fg 40000 --num-bg 100000 --num-bases 10 --seed 0 \
        --out ../data/shape_of_motion/native_init/paper-windmill.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from room4d.render.gaussian4d.motion_init import NativeMotionInitializer


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump room4d's native motion init (no SoM env / cuML)")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--num-fg", type=int, default=40_000)
    ap.add_argument("--num-bg", type=int, default=100_000)
    ap.add_argument("--num-bases", type=int, default=10)
    ap.add_argument("--camera-type", default="refined", choices=["refined", "original"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Build on CPU (the lift holds all full-res frames; KMeans/kNN are CPU anyway).
    scene, cams = NativeMotionInitializer(
        dataset_root=args.dataset_root, factor=1, num_fg=args.num_fg, num_bg=args.num_bg,
        num_bases=args.num_bases, camera_type=args.camera_type, seed=args.seed,
    ).build(device="cpu")

    fg, bases, bg = scene.fg, scene.motion_bases, scene.bg
    dump: dict[str, np.ndarray] = {
        "fg_means": _np(fg.means),
        "fg_quats": _np(fg.quats),
        "fg_scales": _np(fg.log_scales),          # log-space
        "fg_colors": _np(fg.colors_logit),        # logit
        "fg_opacities": _np(fg.logit_opacities),  # logit
        "fg_motion_coefs": _np(fg.motion_coefs),  # raw (softmax at use)
        "motion_rots": _np(bases.rots),           # (K, T, 6)
        "motion_transls": _np(bases.transls),     # (K, T, 3)
        "num_bases": np.asarray(bases.num_bases),
        "factor": np.asarray(1),                  # full-res frame, like the borrow
    }
    if bg is not None:
        dump.update({
            "bg_means": _np(bg.means),
            "bg_quats": _np(bg.quats),
            "bg_scales": _np(bg.log_scales),
            "bg_colors": _np(bg.colors_logit),
            "bg_opacities": _np(bg.logit_opacities),
        })
    for split, cameras in cams.items():
        dump[f"{split}_w2cs"] = np.stack([c.w2c for c in cameras]).astype(np.float32)
        dump[f"{split}_Ks"] = np.stack([c.K for c in cameras]).astype(np.float32)
        dump[f"{split}_time_ids"] = np.asarray([c.time_id for c in cameras])
        dump[f"{split}_frame_names"] = np.asarray([c.frame_name for c in cameras])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **dump)
    print(
        f"native init -> {out}\n"
        f"  fg={dump['fg_means'].shape[0]}  bg={dump.get('bg_means', np.zeros((0,))).shape[0]}  "
        f"bases={int(dump['num_bases'])}  train_cams={dump['train_w2cs'].shape[0]}  "
        f"val_cams={dump['val_w2cs'].shape[0]}"
    )


if __name__ == "__main__":
    main()

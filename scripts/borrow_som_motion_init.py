"""Borrow Shape-of-Motion's track-based motion initialization (STOPGAP).

Runs in the SoM reference env (room4d_som_ref): builds the iPhone dataset, runs
SoM's `init_model_from_tracks` (3D tracks -> cuML clustering + Procrustes ->
motion bases + coefs + canonical fg + static bg), and dumps the init tensors +
the normalized train/val cameras to an .npz that room4d loads via
`render/gaussian4d/motion_init.py`.

This borrows the heavy motion-init machinery the way we borrow poses/depth — it
is NOT room4d's own implementation; replacing it with a native MotionInitializer
is tracked follow-up work. SoM stores params as means(raw)/scales(log)/
colors(logit)/opacities(logit)/coefs(raw), which match room4d's fields exactly.

Example (in room4d_som_ref):
    python scripts/borrow_som_motion_init.py \
        --data-dir ../data/shape_of_motion/iphone/paper-windmill \
        --factor 2 --out ../data/shape_of_motion/borrowed_init/paper-windmill.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SOM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "shape-of-motion"
sys.path.insert(0, str(SOM_DIR))

from flow3d.data import get_train_val_datasets, iPhoneDataConfig  # noqa: E402
from run_training import init_model_from_tracks  # noqa: E402


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump SoM's motion init for room4d (stopgap borrow)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-fg", type=int, default=40_000)
    parser.add_argument("--num-bg", type=int, default=100_000)
    parser.add_argument("--num-bases", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # SoM's iphone loader runs at factor 1 (full res); cameras/Gaussians are in
    # that frame. room4d scales the dumped Ks when it renders at a coarser factor.
    data_cfg = iPhoneDataConfig(data_dir=args.data_dir)
    train_ds, _train_video, val_ds, _val_kpt = get_train_val_datasets(data_cfg, load_val=True)

    fg, bases, bg, _tracks = init_model_from_tracks(
        train_ds, args.num_fg, args.num_bg, args.num_bases
    )

    dump: dict[str, np.ndarray] = {
        # foreground (deformable) canonical params + motion coefficients
        "fg_means": _np(fg.params["means"]),
        "fg_quats": _np(fg.params["quats"]),
        "fg_scales": _np(fg.params["scales"]),       # log-space (matches room4d)
        "fg_colors": _np(fg.params["colors"]),       # logit (matches room4d)
        "fg_opacities": _np(fg.params["opacities"]), # logit
        "fg_motion_coefs": _np(fg.params["motion_coefs"]),  # raw (softmax at use)
        # motion bases
        "motion_rots": _np(bases.params["rots"]),     # (K, T, 6)
        "motion_transls": _np(bases.params["transls"]),  # (K, T, 3)
        "num_bases": np.asarray(bases.num_bases),
        "factor": np.asarray(1),  # SoM iphone full-res frame
    }
    if bg is not None:
        dump.update({
            "bg_means": _np(bg.params["means"]),
            "bg_quats": _np(bg.params["quats"]),
            "bg_scales": _np(bg.params["scales"]),
            "bg_colors": _np(bg.params["colors"]),
            "bg_opacities": _np(bg.params["opacities"]),
        })

    for name, ds in (("train", train_ds), ("val", val_ds)):
        dump[f"{name}_w2cs"] = _np(ds.w2cs)            # (N, 4, 4) world->cam, normalized
        dump[f"{name}_Ks"] = _np(ds.Ks)               # (N, 3, 3)
        dump[f"{name}_time_ids"] = _np(ds.time_ids)   # (N,) basis frame index
        dump[f"{name}_frame_names"] = np.asarray(list(ds.frame_names))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **dump)
    print(
        f"borrowed init -> {out}\n"
        f"  fg={dump['fg_means'].shape[0]}  bg={dump.get('bg_means', np.zeros((0,))).shape[0]}  "
        f"bases={int(dump['num_bases'])}  train_cams={dump['train_w2cs'].shape[0]}  "
        f"val_cams={dump['val_w2cs'].shape[0]}"
    )


if __name__ == "__main__":
    main()

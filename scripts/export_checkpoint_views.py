#!/usr/bin/env python
"""Export held-out views and a replay video from a run's trained dynamic scene.

Renders straight from ``<run_dir>/render/trained_dynamic.npz``; nothing is
retrained, so this works on old runs too. Writes, under ``--out`` (default
``<run_dir>/render/exports``):

    heldout/<n>_<frame>_{render,photo,mask}.png + heldout/index.json
        evenly spaced views from the benchmark's held-out split, each with its
        masked PSNR, placed exactly as the masked-NVS benchmark places them
    replay.mp4 + replay.json
        the scene re-rendered along the captured camera path (fidelity, not
        novel views)

Held-out views need the benchmark dataset (``--dataset-root``); the replay does
not. Needs CUDA (the renderer is gsplat-only).

Usage:
    python scripts/export_checkpoint_views.py RUN_DIR --dataset-root DATASET [--views 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reifyscene.render.eval.checkpoint_views import export_heldout_views, export_replay
from reifyscene.render.eval.masked_nvs_benchmark import eval_dataset
from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
from reifyscene.render.gaussian4d.rasterizer import build_rasterizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="default: RUN_DIR/render/trained_dynamic.npz")
    ap.add_argument("--out", type=Path, help="default: RUN_DIR/render/exports")
    ap.add_argument("--dataset-root", help="benchmark sequence dir; omit to skip held-out views")
    ap.add_argument("--factor", type=int, default=2, help="dataset resolution factor (as scored)")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--views", type=int, default=8, help="held-out views to export")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--replay-fps", type=int, default=30)
    ap.add_argument("--replay-stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    checkpoint = a.checkpoint or a.run_dir / "render" / "trained_dynamic.npz"
    if not checkpoint.exists():
        sys.exit(f"no trained scene at {checkpoint}")
    out = a.out or a.run_dir / "render" / "exports"

    data = np.load(checkpoint, allow_pickle=True)
    scene, cams = bundle_to_scene(data, device=a.device)
    rasterizer = build_rasterizer({"device": a.device})

    if a.dataset_root:
        eval_cfg = {"dataset_root": a.dataset_root, "factor": a.factor, "eval_split": a.eval_split}
        records = export_heldout_views(
            scene,
            data,
            cams,
            eval_cfg,
            out / "heldout",
            backend=eval_dataset(eval_cfg),
            rasterizer=rasterizer,
            count=a.views,
            device=a.device,
        )
        for r in records:
            name, t, psnr = r["frame_name"], r["time_id"], r["masked_psnr"]
            print(f"  held-out {name:>10}  t={t:<4} masked PSNR {psnr:6.2f}")
        print(f"held-out views -> {out / 'heldout'}")

    if not a.no_replay:
        meta = export_replay(
            scene,
            cams,
            out / "replay.mp4",
            rasterizer=rasterizer,
            fps=a.replay_fps,
            stride=a.replay_stride,
        )
        size = f"{meta['width']}x{meta['height']}"
        print(f"replay: {len(meta['time_ids'])} frames at {size} -> {out / 'replay.mp4'}")


if __name__ == "__main__":
    main()

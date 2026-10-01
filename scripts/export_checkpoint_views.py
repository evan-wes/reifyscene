#!/usr/bin/env python
"""Export held-out views, a score and a replay video from a run's trained scene.

Renders straight from one of the run's checkpoints (``--checkpoint``, the render stage's own
by default) -- the weights and the scene kind its render record names; nothing is retrained,
and the run's own records are never modified. Writes, under ``--out`` (default ``exports/``
inside the checkpoint's folder):

    heldout/<n>_<frame>_{render,photo,mask}.png + heldout/index.json
        evenly spaced views from the benchmark's held-out split, each with its
        masked PSNR, placed exactly as the masked-NVS benchmark places them
    score.json
        masked mPSNR / mSSIM / mLPIPS over the held-out split, with how the
        held-out cameras were placed (or why an alignment was refused)
    replay.mp4 + replay.json  (dynamic scenes only)
        the scene re-rendered along the captured camera path (fidelity, not
        novel views)

Held-out cameras are placed per ``--align``: ``auto`` (default) does what the
pipeline does -- for a dynamic scene it reads the run's geometry backend from its
manifest, ``none`` for the dataset (GT oracle) backend and ``sim3`` (fit on the
training cameras) for self-reconstructed geometry; every static scene gets
``sim3``, since it lives in the ground-plane-canonicalized frame. A static scene's
training cameras are read from the run's scene record.

Held-out views and the score need the benchmark dataset (``--dataset-root``); the
replay does not. Needs CUDA (the renderer is gsplat-only).

Usage:
    python scripts/export_checkpoint_views.py RUN_DIR --dataset-root DATASET [--views 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.api.run_outputs import Checkpoint, RunOutputs
from reifyscene.render.eval.checkpoint_views import (
    export_heldout_views,
    export_replay,
    export_static_heldout_views,
)
from reifyscene.render.eval.masked_nvs_benchmark import (
    AlignmentFailed,
    default_alignment,
    eval_dataset,
    load_trained_cloud,
    score_dynamic_bundle,
    score_static_cloud,
)
from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
from reifyscene.render.gaussian4d.rasterizer import build_rasterizer


def find_checkpoint(run: RunOutputs, name: str) -> Checkpoint:
    """The checkpoint to render; its weights and kind are what its render record says."""
    checkpoint = run.checkpoint(name)
    if not checkpoint.exists():
        sys.exit(f"{run.name}: no trained checkpoint {name!r}")
    return checkpoint


def read_train_views(run: RunOutputs) -> list[dict[str, Any]]:
    """A static scene's training cameras (frame, time, pose) from its scene record."""
    if not run.has("scene"):
        sys.exit(f"{run.name}: a static scene's held-out cameras are placed through its scene "
                 "record, which is missing")
    views = run.record("scene")["train_views"]
    return [
        {"frame_key": v["frame_key"], "time_id": v["time_id"], "t_world_cam": v["t_world_cam"]}
        for v in views
    ]


def resolve_alignment(run: RunOutputs, requested: str, kind: str = "dynamic") -> str:
    """``requested`` unless ``auto``: then the pipeline's default for the run's scene."""
    if requested != "auto":
        return requested
    if kind == "static":
        return default_alignment(None, kind)  # every static scene, whatever its geometry
    return default_alignment((run.config.get("geometry") or {}).get("backend_name"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", default="render",
                    help="the checkpoint to render: the render stage's own, or a named retrain")
    ap.add_argument("--out", type=Path, help="default: exports/ in the checkpoint's folder")
    ap.add_argument("--dataset-root", help="benchmark sequence dir; omit to skip held-out views")
    ap.add_argument("--factor", type=int, default=2, help="dataset resolution factor (as scored)")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--align", choices=["auto", "none", "sim3"], default="auto")
    ap.add_argument("--views", type=int, default=8, help="held-out views to export")
    ap.add_argument("--no-score", action="store_true", help="skip scoring the full split")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--replay-fps", type=int, default=30)
    ap.add_argument("--replay-stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    run = RunOutputs.open(a.run_dir)
    ckpt = find_checkpoint(run, a.checkpoint)
    checkpoint, kind = ckpt.weights, ckpt.kind
    out = a.out or ckpt.dir / "exports"
    rasterizer = build_rasterizer({"device": a.device})
    if kind == "dynamic":
        data = np.load(checkpoint, allow_pickle=True)
        scene, cams = bundle_to_scene(data, device=a.device)
    else:
        cloud = load_trained_cloud(checkpoint, a.device)

    if a.dataset_root:
        eval_cfg = {
            "dataset_root": a.dataset_root,
            "factor": a.factor,
            "eval_split": a.eval_split,
            "align": resolve_alignment(run, a.align, kind),
        }
        backend = eval_dataset(eval_cfg)
        train_views = read_train_views(run) if kind == "static" else []
        print(f"{kind} scene: held-out cameras placed with align={eval_cfg['align']}")
        if not a.no_score:
            if kind == "dynamic":
                score = score_dynamic_bundle(
                    data, scene, cams, eval_cfg, backend=backend, device=a.device,
                    rasterizer=rasterizer,
                )
            else:
                score = score_static_cloud(
                    cloud, train_views, eval_cfg, backend=backend, device=a.device,
                    rasterizer=rasterizer,
                )
            out.mkdir(parents=True, exist_ok=True)
            (out / "score.json").write_text(json.dumps(score, indent=2) + "\n")
            if "masked_mpsnr" in score:
                print(
                    f"score: masked mPSNR {score['masked_mpsnr']:.2f} over {score['num_views']} "
                    f"views -> {out / 'score.json'}"
                )
            else:
                print(f"score: none -> {out / 'score.json'} ({score.get('alignment', {})})")
        export = dict(backend=backend, rasterizer=rasterizer, count=a.views, device=a.device)
        try:
            if kind == "dynamic":
                records = export_heldout_views(
                    scene, data, cams, eval_cfg, out / "heldout", **export
                )
            else:
                records = export_static_heldout_views(
                    cloud, train_views, eval_cfg, out / "heldout", **export
                )
        except AlignmentFailed as exc:
            sys.exit(f"held-out views not exported: alignment refused: {exc}")
        for r in records:
            name, t, psnr = r["frame_name"], r["time_id"], r["masked_psnr"]
            print(f"  held-out {name:>10}  t={t:<4} masked PSNR {psnr:6.2f}")
        print(f"held-out views -> {out / 'heldout'}")

    if a.no_replay:
        return
    if kind == "static":
        print("replay: skipped (the replay is for dynamic scenes)")
        return
    meta = export_replay(
        scene,
        cams,
        out / "replay.mp4",
        rasterizer=rasterizer,
        raster=tuple(int(v) for v in data["train_raster_wh"]),  # what it was trained at
        fps=a.replay_fps,
        stride=a.replay_stride,
    )
    size = f"{meta['width']}x{meta['height']}"
    print(f"replay: {len(meta['time_ids'])} frames at {size} -> {out / 'replay.mp4'}")


if __name__ == "__main__":
    main()

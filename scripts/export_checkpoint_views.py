#!/usr/bin/env python
"""Export held-out views, a score and a replay video from a run's trained scene.

Renders straight from the run's checkpoint -- ``<run_dir>/render/trained_dynamic.npz``
for a dynamic scene, else ``<run_dir>/render/trained_gaussians.npz`` for a static
one; nothing is retrained, so this works on old runs too, and the run's own records
are never modified. Writes, under ``--out`` (default ``<run_dir>/render/exports``):

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
training cameras are read from ``<run_dir>/scene/artifact.json``.

``--checkpoint`` reads another checkpoint; its kind comes from its contents (a
dynamic bundle carries ``fg_means``, a static cloud does not).

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

CHECKPOINTS = ("trained_dynamic.npz", "trained_gaussians.npz")


def checkpoint_kind(path: Path) -> str:
    """``dynamic`` for a dynamic bundle (it carries ``fg_means``), else ``static``."""
    with np.load(path, allow_pickle=True) as data:
        return "dynamic" if "fg_means" in data.files else "static"


def find_checkpoint(run_dir: Path, override: Path | None) -> tuple[Path, str]:
    """The checkpoint to render and its kind: ``override``, else the run's own."""
    candidates = [override] if override else [run_dir / "render" / name for name in CHECKPOINTS]
    for path in candidates:
        if path.exists():
            return path, checkpoint_kind(path)
    sys.exit(f"no trained scene at {' or '.join(str(p) for p in candidates)}")


def read_train_views(run_dir: Path) -> list[dict[str, Any]]:
    """A static scene's training cameras (frame, time, pose) from its scene artifact."""
    path = run_dir / "scene" / "artifact.json"
    if not path.exists():
        sys.exit(f"a static scene's held-out cameras are placed through {path}, which is missing")
    views = json.loads(path.read_text()).get("train_views") or []
    return [
        {"frame_key": v["frame_key"], "time_id": v["time_id"], "t_world_cam": v["t_world_cam"]}
        for v in views
    ]


def resolve_alignment(run_dir: Path, requested: str, kind: str = "dynamic") -> str:
    """``requested`` unless ``auto``: then the pipeline's default for the run's scene."""
    if requested != "auto":
        return requested
    if kind == "static":
        return default_alignment(None, kind)  # every static scene, whatever its geometry
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        sys.exit(f"--align auto needs {manifest} to read the geometry backend; pass --align")
    cfg = json.loads(manifest.read_text()).get("config") or {}
    return default_alignment((cfg.get("geometry") or {}).get("backend_name"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--checkpoint", type=Path, help="default: RUN_DIR/render/trained_{dynamic,gaussians}.npz"
    )
    ap.add_argument("--out", type=Path, help="default: RUN_DIR/render/exports")
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

    checkpoint, kind = find_checkpoint(a.run_dir, a.checkpoint)
    out = a.out or a.run_dir / "render" / "exports"
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
            "align": resolve_alignment(a.run_dir, a.align, kind),
        }
        backend = eval_dataset(eval_cfg)
        train_views = read_train_views(a.run_dir) if kind == "static" else []
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
        fps=a.replay_fps,
        stride=a.replay_stride,
    )
    size = f"{meta['width']}x{meta['height']}"
    print(f"replay: {len(meta['time_ids'])} frames at {size} -> {out / 'replay.mp4'}")


if __name__ == "__main__":
    main()

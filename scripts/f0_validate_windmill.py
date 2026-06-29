"""F0b validation: in-pipeline dynamic render on DyCheck paper-windmill.

Drives the F0b wiring end to end -- build a ``DynamicScene`` from oracle geometry,
serialize it into a ``SceneArtifact``, reload it, train + render with the new
dynamic render backend, and score covisibility-masked NVS with the new dynamic
scorer -- so the masked mPSNR is directly comparable to the standalone
``run_render_proof.py cmd_dynamic`` path (the known-good ~20.7).

Two scene sources (the only thing that differs from the reference):

* ``borrowed-init`` -- load the pre-dumped SoM init bundle (no TAPIR; a fast
  smoke of the serialize -> reload -> render -> score wiring).
* ``own-tracker`` -- the gate (A): oracle ``DyCheckGeometryProvider`` geometry +
  in-memory packed all-pairs tracks (room4d's own tapnet tracker) ->
  ``NativeMotionInitializer``. Reproduces the native+own-tracker number through
  the F0b artifact seam.

Everything downstream (factor-2 training views, k_scale, masked-NVS) mirrors
``cmd_dynamic`` exactly, so a delta here is a wiring delta, not a metric one.

Example (gate):
    python scripts/f0_validate_windmill.py \
        --dataset-root ../data/shape_of_motion/iphone/paper-windmill \
        --config configs/render/render_proof_windmill_dynamic.yaml \
        --source own-tracker --out-dir /tmp/f0_validate/windmill_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.lifting.geometry_provider import DyCheckGeometryProvider
from room4d.render.eval.masked_nvs_benchmark import score_render_artifact_masked_nvs_dynamic
from room4d.render.gaussian4d.motion_init import (
    NativeMotionInitializer,
    dump_bundle,
    load_borrowed_init,
)
from room4d.render.gaussian4d.render_backend import GaussianRenderBackend
from room4d.scene.scene_artifact import SceneArtifact
from room4d.tracking.tapir_wrapper import TAPIRWrapper


def _build_borrowed(init_npz: str, device: str):
    """Pre-dumped SoM init bundle -> (scene, cams, scene_norm=None)."""
    scene, cams = load_borrowed_init(init_npz, device=device)
    return scene, cams, None


def _build_own_tracker(dataset_root: str, device: str, *, num_fg: int, num_bg: int, num_bases: int, seed: int):
    """Oracle geometry + in-memory packed all-pairs tracks -> (scene, cams, norm).

    This is the F0b in-memory track path: room4d's tapnet tracker produces the
    packed AllPairsTracks once (no per-pair files on disk) and the lift reads
    pairs straight from it via ``pair_loader``.
    """
    provider = DyCheckGeometryProvider(dataset_root, factor=1, depth_type="depth_anything_colmap", camera_type="refined")
    stack = provider.frame_stack("train", device="cpu")
    print(f"own-tracker: tracking all-pairs over {stack.num_frames} train frames (tapnet)...")
    all_pairs = TAPIRWrapper({"device": device}).compute_all_pairs(
        stack.rgbs.cpu().numpy(), stack.fg.cpu().numpy(), list(stack.cameras.frame_names), factor=1
    )
    scene, cams, norm = NativeMotionInitializer(
        dataset_root=dataset_root, factor=1, num_fg=num_fg, num_bg=num_bg, num_bases=num_bases,
        seed=seed, provider=provider, pair_loader=all_pairs.pair,
    ).build(device=device, return_norm=True)
    return scene, cams, norm


def _train_views_factor2(cams_train, dataset_root: str, factor: int):
    """Factor-`factor` training views (full-res cameras scaled by 1/factor + GT path)."""
    backend = DatasetGeometryInitializer({"dataset_root": dataset_root, "factor": factor, "split": "train"})
    k_scale = 1.0 / factor
    views = []
    for cam in cams_train:
        gt_u8 = backend.load_rgb(cam.frame_name)
        height, width = gt_u8.shape[:2]
        K = np.asarray(cam.K, dtype=np.float32).copy()
        K[:2] *= k_scale  # full-res K -> factor resolution (mirrors render_camera)
        views.append({
            "frame_key": str(cam.frame_name),
            "time_id": int(cam.time_id),
            "t_world_cam": np.linalg.inv(np.asarray(cam.w2c, dtype=np.float32)).astype(np.float32),
            "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
            "width": int(width),
            "height": int(height),
            "image_path": str(backend.rgb_dir / f"{cam.frame_name}.png"),
        })
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description="F0b in-pipeline dynamic render validation (paper-windmill).")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", default="configs/render/render_proof_windmill_dynamic.yaml")
    ap.add_argument("--source", choices=["borrowed-init", "own-tracker"], default="own-tracker")
    ap.add_argument("--init", default=None, help="borrowed-init bundle npz (for --source borrowed-init)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=None, help="override training iters (else config)")
    ap.add_argument("--num-fg", type=int, default=40_000)
    ap.add_argument("--num-bg", type=int, default=100_000)
    ap.add_argument("--num-bases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["render_proof"]
    device = args.device or cfg.get("device", "cuda")
    factor = int(cfg["dataset"]["factor"])
    iters = args.iters or int(cfg["dynamic"]["iters"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the DynamicScene + cameras (the only source-specific step).
    if args.source == "borrowed-init":
        init = args.init or str(Path(args.dataset_root).parent.parent / "borrowed_init" / f"{Path(args.dataset_root).name}.npz")
        print(f"source=borrowed-init  init={init}")
        scene, cams, norm = _build_borrowed(init, device)
    else:
        print("source=own-tracker  (oracle geometry + in-memory all-pairs)")
        scene, cams, norm = _build_own_tracker(
            args.dataset_root, device, num_fg=args.num_fg, num_bg=args.num_bg,
            num_bases=args.num_bases, seed=args.seed,
        )
    print(
        f"init: fg={scene.fg.num_gaussians} bg={scene.bg.num_gaussians if scene.bg else 0} "
        f"bases={scene.fg.num_bases} frames={scene.num_frames}"
    )

    # 2. Serialize into a SceneArtifact and reload (exercises piece (a) on disk).
    bundle = dump_bundle(scene, cams, factor=1, scene_norm=norm)
    train_views = _train_views_factor2(cams["train"], args.dataset_root, factor)
    art = SceneArtifact(
        run_id="f0_validate", stage_name="scene", save_dir=out_dir / "scene",
        kind="dynamic", render_seed={}, train_views=train_views, render_trajectory=[],
        dynamic_bundle=bundle,
    )
    art.save()
    art = SceneArtifact.load(out_dir / "scene")
    print(f"scene artifact: kind={art.kind} train_views={len(art.train_views)} (saved + reloaded)")

    # 3. Train + render with the new dynamic render backend.
    render_cfg = {"device": device, "iters": iters, "dynamic": cfg["dynamic"], "densify": cfg.get("densify")}
    print(f"training {iters} iters...")
    render_art = GaussianRenderBackend(render_cfg).fit(art, save_dir=out_dir / "render")
    print(f"render: {render_art.scene_state}")

    # 4. Score covisibility-masked NVS with the new dynamic scorer (the gate number).
    bm = cfg.get("benchmark", {}) or {}
    eval_cfg = {
        "dataset_root": args.dataset_root, "factor": factor, "split": "train", "eval_split": "val",
        "lpips_net": bm.get("lpips_net", "alex"), "max_eval_frames": bm.get("max_eval_frames"),
    }
    metrics = score_render_artifact_masked_nvs_dynamic(render_art, eval_cfg, device=device)
    print("=" * 60)
    print(f"F0b GATE  source={args.source}  iters={iters}")
    print(
        f"  masked mPSNR={metrics.get('masked_mpsnr'):.4f}  mSSIM={metrics.get('masked_mssim'):.4f}  "
        f"mLPIPS={metrics.get('masked_mlpips'):.4f}  (n={metrics.get('num_views')})"
    )
    print("=" * 60)
    summary = {
        "source": args.source, "iters": iters, "factor": factor,
        "final_gaussians": int(render_art.scene_state.get("num_gaussians", 0)),
        "metrics": {k: metrics.get(k) for k in ("masked_mpsnr", "masked_mssim", "masked_mlpips", "num_views")},
    }
    (out_dir / "f0_validate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'f0_validate_summary.json'}")


if __name__ == "__main__":
    main()

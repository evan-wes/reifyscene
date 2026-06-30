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


def _slice_cameras(c, n: int):
    from room4d.lifting.native_track_lift import CameraRecords
    return CameraRecords(frame_names=c.frame_names[:n], time_ids=c.time_ids[:n], w2cs=c.w2cs[:n], Ks=c.Ks[:n])


def _slice_stack(s, n: int):
    from room4d.lifting.native_track_lift import FrameStack
    return FrameStack(cameras=_slice_cameras(s.cameras, n), depths=s.depths[:n], rgbs=s.rgbs[:n], fg=s.fg[:n], valid=s.valid[:n])


class _WindmillProvider:
    """DyCheckGeometryProvider wrapper: limit train frames + optionally override fg."""

    def __init__(self, inner, max_frames: int = 0, train_fg=None) -> None:
        self.inner = inner
        self.max_frames = int(max_frames)
        self.train_fg = train_fg  # (N, H, W) torch foreground, or None to keep dataset fg

    def cameras(self, split, *, device="cpu"):
        c = self.inner.cameras(split, device=device)
        return _slice_cameras(c, self.max_frames) if (split == "train" and self.max_frames) else c

    def frame_stack(self, split, *, device="cpu"):
        s = self.inner.frame_stack(split, device=device)
        if self.max_frames:
            s = _slice_stack(s, self.max_frames)
        if self.train_fg is not None:
            s.fg = self.train_fg.to(s.fg)
        return s


def _sam2_fg(rgbs_np: np.ndarray, device: str) -> "torch.Tensor":
    """Run the SAM 2 segmenter over the frames -> per-frame union foreground (N, H, W)."""
    from room4d.common.types import FrameRef
    from room4d.perception.instance.sam2_segment import SAM2InstanceSegmenter

    images = [rgbs_np[i] for i in range(rgbs_np.shape[0])]

    class _D:
        def get_frame(self, stream, idx):
            return images[int(idx)]

    class _S:
        stream_id = "cam_00"

    class _Det:
        backend_used = "none"
        def detect(self, frame):
            return []  # no person detector needed for the windmill scene

    class _I:
        streams = [_S()]

    seg = SAM2InstanceSegmenter(_D(), _Det(), {})
    refs = [FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) * 0.05) for i in range(len(images))]
    obs, masks = seg.run(_I(), refs)
    h, w = images[0].shape[:2]
    fg = np.zeros((len(images), h, w), dtype=np.float32)
    for o in obs:
        i = int(o.frame_ref.frame_idx)
        fg[i] = np.maximum(fg[i], masks[o.instance_id].astype(np.float32))
    print(f"sam2 fg: tracks={len(set(o.track_id for o in obs))}  coverage={fg.mean():.3f}")
    return torch.as_tensor(fg, device="cpu")


def _build_own_tracker(
    dataset_root: str, device: str, *, num_fg: int, num_bg: int, num_bases: int, seed: int,
    fg_source: str = "dataset", max_frames: int = 0, motion_gate: dict | None = None,
    grid_size: int = 4,
):
    """Oracle geometry + in-memory packed all-pairs tracks -> (scene, cams, norm).

    ``fg_source`` picks the foreground the lift seeds + masks from: ``dataset`` (the
    DyCheck oracle fg, the F0b gate) or ``sam2`` (the SAM 2 segment union — broad,
    includes static background). With ``motion_gate`` enabled, the lift then keeps
    only ever-moved tracks as deformable fg and reroutes static ones to background.
    """
    inner = DyCheckGeometryProvider(dataset_root, factor=1, depth_type="depth_anything_colmap", camera_type="refined")
    stack = inner.frame_stack("train", device="cpu")
    if max_frames:
        stack = _slice_stack(stack, max_frames)
    frame_names = list(stack.cameras.frame_names)
    rgbs_np = stack.rgbs.cpu().numpy()

    if fg_source == "sam2":
        print(f"sam2 fg: segmenting {len(frame_names)} frames...")
        train_fg = _sam2_fg(rgbs_np, device)
        fg_np = train_fg.cpu().numpy()
    else:
        train_fg = None
        fg_np = stack.fg.cpu().numpy()

    print(f"own-tracker: all-pairs over {len(frame_names)} frames (fg={fg_source}, grid={grid_size}, gate={bool(motion_gate and motion_gate.get('enabled'))})...")
    all_pairs = TAPIRWrapper({"device": device, "grid_size": grid_size}).compute_all_pairs(rgbs_np, fg_np, frame_names, factor=1)
    provider = _WindmillProvider(inner, max_frames=max_frames, train_fg=train_fg)
    scene, cams, norm = NativeMotionInitializer(
        dataset_root=dataset_root, factor=1, num_fg=num_fg, num_bg=num_bg, num_bases=num_bases,
        seed=seed, provider=provider, pair_loader=all_pairs.pair, motion_gate=motion_gate,
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
    ap.add_argument("--fg-source", choices=["dataset", "sam2"], default="dataset",
                    help="foreground source for the lift (own-tracker): DyCheck oracle vs SAM 2 segment")
    ap.add_argument("--motion-gate", action="store_true", help="keep only ever-moved tracks as fg (static -> bg)")
    ap.add_argument("--motion-gate-percentile", type=float, default=95.0)
    ap.add_argument("--motion-gate-threshold", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=0, help="limit train frames (0 = all); for smoke runs")
    ap.add_argument("--tracker-grid-size", type=int, default=4,
                    help="all-pairs query-grid stride (coarsen for broad sam2 fg to stay tractable)")
    args = ap.parse_args()
    motion_gate = {
        "enabled": bool(args.motion_gate),
        "percentile": float(args.motion_gate_percentile),
        "threshold": float(args.motion_gate_threshold),
    }

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
        print(f"source=own-tracker  (oracle geometry + in-memory all-pairs, fg={args.fg_source})")
        scene, cams, norm = _build_own_tracker(
            args.dataset_root, device, num_fg=args.num_fg, num_bg=args.num_bg,
            num_bases=args.num_bases, seed=args.seed,
            fg_source=args.fg_source, max_frames=args.max_frames, motion_gate=motion_gate,
            grid_size=args.tracker_grid_size,
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
    gate_on = bool(motion_gate.get("enabled"))
    print(f"F0b NVS  source={args.source}  fg={args.fg_source}  motion_gate={gate_on}  iters={iters}")
    print(
        f"  masked mPSNR={metrics.get('masked_mpsnr'):.4f}  mSSIM={metrics.get('masked_mssim'):.4f}  "
        f"mLPIPS={metrics.get('masked_mlpips'):.4f}  (n={metrics.get('num_views')})"
    )
    print("=" * 60)
    summary = {
        "source": args.source, "fg_source": args.fg_source, "motion_gate": motion_gate if gate_on else None,
        "max_frames": args.max_frames, "iters": iters, "factor": factor,
        "fg_gaussians": int(render_art.scene_state.get("num_fg", 0)),
        "bg_gaussians": int(render_art.scene_state.get("num_bg", 0)),
        "final_gaussians": int(render_art.scene_state.get("num_gaussians", 0)),
        "metrics": {k: metrics.get(k) for k in ("masked_mpsnr", "masked_mssim", "masked_mlpips", "num_views")},
    }
    (out_dir / "f0_validate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'f0_validate_summary.json'}")


if __name__ == "__main__":
    main()

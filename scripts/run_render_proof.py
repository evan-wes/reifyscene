"""Render-proof harness for the reifyscene Gaussian renderer (Track 2).

Two modes:

  checkpoint0  Coordinate sanity. Seed an untrained colored point-cloud splat
               from one view's depth and render at real dataset cameras; write
               GT-vs-render panels. PASS = recognizable, correctly placed.

  train        Static Gaussian training (Checkpoints 1 & 2). Seed -> train the
               GaussianField (Adam + L1/D-SSIM + densification) -> render train
               and held-out views, report PSNR/SSIM, write panels. All knobs
               come from a config file (configs/render/render_proof.yaml).

Examples:
    python scripts/run_render_proof.py checkpoint0 \
        --dataset-root .../Balloon1 --factor 2 \
        --seed-frames 0:0 --render-targets 0:0,1:0,2:0,3:0 --out-dir ck0

    python scripts/run_render_proof.py train \
        --dataset-root .../Balloon1 --config configs/render/render_proof.yaml \
        --iters 7000 --out-dir ck1_balloon1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from reifyscene.common.io import read_yaml
from reifyscene.common.types import FrameRef
from reifyscene.eval.benchmarks.covisibility_masked_novel_view_synthesis import (
    build_record,
    compare,
    write_record,
)
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.render.camera_paths import orbit_cameras
from reifyscene.render.eval.novel_view import score_masked_dynamic_views, score_masked_novel_views
from reifyscene.render.export import write_video
from reifyscene.render.gaussian4d.densify import DensifyConfig
from reifyscene.render.gaussian4d.dynamic_trainer import DynamicGaussianTrainer, DynamicTrainConfig
from reifyscene.render.gaussian4d.gaussian_field import GaussianField
from reifyscene.render.gaussian4d.gaussian_trainer import StaticGaussianTrainer, TrainConfig
from reifyscene.render.gaussian4d.motion_init import build_native_init, load_borrowed_init
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer
from reifyscene.render.gaussian4d.seed import seed_cloud_from_geometry


# -- shared helpers -------------------------------------------------------

def _parse_frames(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        cam, _, time_id = token.partition(":")
        out.append((cam.strip(), int(time_id)))
    return out


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def _label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, max(60, 7 * len(text)), 14), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return img


def _panel(gt: np.ndarray, render: np.ndarray, title: str, metrics: str) -> Image.Image:
    diff = np.abs(gt.astype(np.int16) - render.astype(np.int16)).clip(0, 255).astype(np.uint8)
    gt_img = _label(Image.fromarray(gt), "GT")
    render_img = _label(Image.fromarray(render), metrics)
    diff_img = _label(Image.fromarray(diff), "|diff|")
    sep = np.full((gt.shape[0], 4, 3), 255, dtype=np.uint8)
    strip = np.hstack([np.asarray(gt_img), sep, np.asarray(render_img), sep, np.asarray(diff_img)])
    return _label(Image.fromarray(strip), title)


def _load_view(backend: DatasetGeometryInitializer, cam: str, time_id: int, device: str):
    """Return (RenderCamera, gt_tensor[H,W,3] in [0,1], gt_uint8[H,W,3])."""
    name = backend.frame_name(cam, time_id)
    intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
    camera = RenderCamera.from_world_cam(t_world_cam, intrinsics, width, height)
    gt_u8 = backend.load_rgb(name)
    gt = torch.as_tensor(gt_u8, dtype=torch.float32, device=device) / 255.0
    return camera, gt, gt_u8


# -- checkpoint 0 ---------------------------------------------------------

def cmd_checkpoint0(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = DatasetGeometryInitializer(
        {"dataset_root": args.dataset_root, "factor": args.factor, "split": args.split}
    )

    seed_specs = _parse_frames(args.seed_frames)
    seed_refs = [FrameRef(stream_id=c, frame_idx=t, timestamp_sec=float(t)) for c, t in seed_specs]
    geometry = backend.initialize(seed_refs)
    rgb_by_frame = {f"{c}:{t}": backend.load_rgb(backend.frame_name(c, t)) for c, t in seed_specs}
    cloud = seed_cloud_from_geometry(
        geometry, rgb_by_frame, device=args.device,
        pixel_stride=args.pixel_stride, max_points=args.max_points, scale_mult=args.scale_mult,
    )
    print(f"seed cloud: {len(cloud)} gaussians from {len(seed_specs)} view(s) {seed_specs}")

    rasterizer = build_rasterizer({"device": args.device})
    seed_keys = {f"{c}:{t}" for c, t in seed_specs}
    summary = {"mode": "checkpoint0", "num_seed_gaussians": len(cloud), "targets": []}
    for cam, time_id in _parse_frames(args.render_targets):
        camera, _gt_t, gt_u8 = _load_view(backend, cam, time_id, args.device)
        out = rasterizer.render(cloud, camera)
        render_u8 = np.clip(out["rgb"] * 255.0, 0, 255).astype(np.uint8)
        coverage = float(np.mean(out["alpha"] > 0.5))
        psnr = _psnr(gt_u8, render_u8)
        kind = "reconstruction" if f"{cam}:{time_id}" in seed_keys else "novel-view"
        title = f"cam{cam} t{time_id} [{kind}]"
        _panel(gt_u8, render_u8, title, f"render psnr={psnr:.1f} cov={coverage:.2f}").save(
            out_dir / f"checkpoint0_cam{cam}_t{time_id}.png"
        )
        print(f"  {title}: psnr={psnr:.2f} dB  coverage={coverage:.3f}")
        summary["targets"].append({"camera": cam, "time": time_id, "kind": kind, "psnr": psnr, "coverage": coverage})
    (out_dir / "checkpoint0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'checkpoint0_summary.json'}")


# -- train (Checkpoints 1 & 2) -------------------------------------------

def cmd_train(args) -> None:
    cfg = read_yaml(args.config)["render_proof"]
    ds, device = cfg["dataset"], (args.device or cfg.get("device", "cuda"))
    iters = args.iters or cfg["trainer"]["iters"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = DatasetGeometryInitializer(
        {"dataset_root": args.dataset_root, "factor": ds["factor"], "split": ds["split"]}
    )

    # Seed the field from the train camera(s) at one or more seed times. A
    # moving monocular camera (e.g. paper-windmill) needs several seed views to
    # cover the scene; a static rig camera needs only one.
    seed_times = ds.get("seed_times") or [ds["seed_time"]]
    seed_specs = [(cam, int(t)) for cam in ds["train_cameras"] for t in seed_times]
    seed_refs = [FrameRef(stream_id=c, frame_idx=t, timestamp_sec=float(t)) for c, t in seed_specs]
    geometry = backend.initialize(seed_refs)
    rgb_by_frame = {f"{c}:{t}": backend.load_rgb(backend.frame_name(c, t)) for c, t in seed_specs}
    seed_cloud = seed_cloud_from_geometry(geometry, rgb_by_frame, device=device, **cfg["seed"])
    field = GaussianField.from_cloud(seed_cloud, color_model=cfg["trainer"]["color_model"])
    print(f"seed cloud: {len(seed_cloud)} gaussians; training {iters} iters on {device}")

    # Training views: train_cameras x train_times.
    train_times = ds["train_times"]
    if train_times is None:
        train_set = set(ds["train_cameras"])
        train_times = sorted({t for c, t in backend.split_frames(ds["split"]) if c in train_set})
    if ds.get("max_train_views"):
        train_times = train_times[: int(ds["max_train_views"])]
    train_views = []
    for cam in ds["train_cameras"]:
        for t in train_times:
            camera, gt, _u8 = _load_view(backend, cam, int(t), device)
            train_views.append((camera, gt))
    print(f"train views: {len(train_views)} ({len(ds['train_cameras'])} cam x {len(train_times)} times)")

    train_cfg = TrainConfig.from_cfg({**cfg["trainer"], "iters": iters})
    densify_cfg = DensifyConfig.from_cfg(cfg["densify"])
    rasterizer = build_rasterizer({**cfg["rasterizer"], "device": device})
    trainer = StaticGaussianTrainer(field, train_cfg, densify_cfg, device=device, rasterizer=rasterizer)

    def _log(record: dict) -> None:
        msg = f"  iter {record['step']:>6}  loss={record['loss']:.4f}  psnr={record['psnr']:.2f}  N={record['num_gaussians']}"
        if "densify" in record:
            d = record["densify"]
            msg += f"  [+{d['cloned']}c/{d['split']}s -{d['pruned']}p]"
        print(msg)

    trainer.fit(train_views, on_log=_log)

    # Checkpoint 1 (train-view) + Checkpoint 2 (held-out novel-view).
    summary = {"mode": "train", "iters": iters, "final_gaussians": field.num_gaussians,
               "checkpoint1_train_view": [], "checkpoint2_novel_view": []}

    def _eval_and_panel(cam: str, time_id: int, tag: str, store: list) -> None:
        camera, gt, gt_u8 = _load_view(backend, cam, int(time_id), device)
        res = trainer.evaluate([(camera, gt)])[0]
        render_u8 = np.clip(res["rgb"] * 255.0, 0, 255).astype(np.uint8)
        title = f"cam{cam} t{time_id} [{tag}]"
        if cfg["eval"].get("save_panels", True):
            _panel(gt_u8, render_u8, title, f"psnr={res['psnr']:.2f} ssim={res['ssim']:.3f}").save(
                out_dir / f"{tag}_cam{cam}_t{time_id}.png"
            )
        print(f"  {title}: psnr={res['psnr']:.2f} dB  ssim={res['ssim']:.3f}")
        store.append({"camera": cam, "time": int(time_id), "psnr": res["psnr"], "ssim": res["ssim"]})

    print("Checkpoint 1 (train-view):")
    for cam in ds["train_cameras"]:
        for t in ds.get("eval_train_times", [ds["seed_time"]]):
            _eval_and_panel(cam, t, "checkpoint1", summary["checkpoint1_train_view"])
    # Held-out cameras don't all share the same timestep, so pick a valid val
    # time per camera (eval_time if it exists for that camera, else its first).
    val_times_by_cam: dict[str, list[int]] = {}
    for cam, t in backend.split_frames("val"):
        val_times_by_cam.setdefault(str(cam), []).append(int(t))
    print("Checkpoint 2 (held-out novel-view):")
    for cam in ds["eval_cameras"]:
        times = val_times_by_cam.get(str(cam), [])
        if not times:
            continue
        t = ds["eval_time"] if ds["eval_time"] in times else times[0]
        _eval_and_panel(cam, t, "checkpoint2", summary["checkpoint2_novel_view"])

    # Covisibility-masked NVS benchmark (DyCheck protocol) over the full val split.
    bm = cfg.get("benchmark", {}) or {}
    if bm.get("enabled", True):
        eval_split = bm.get("eval_split", "val")
        val_specs = backend.split_frames(eval_split)
        if bm.get("max_eval_frames"):
            val_specs = val_specs[: int(bm["max_eval_frames"])]
        print(f"Benchmark: covisibility-masked NVS over {len(val_specs)} {eval_split} frames...")

        def _val_views():
            for cam, t in val_specs:
                name = backend.frame_name(cam, int(t))
                intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
                camera = RenderCamera.from_world_cam(t_world_cam, intrinsics, width, height)
                gt = torch.as_tensor(backend.load_rgb(name), dtype=torch.float32, device=device) / 255.0
                mask = torch.as_tensor(backend.load_covisible(name, eval_split), device=device)
                yield camera, gt, mask

        metrics = score_masked_novel_views(
            rasterizer, field.to_cloud(), _val_views(), device=device, lpips_net=bm.get("lpips_net", "alex")
        )
        record = build_record(
            method=bm.get("method_name", "reifyscene"),
            sequence=Path(args.dataset_root).name,
            dataset_format=bm.get("dataset_format", "iphone"),
            metrics=metrics,
            eval={"source": "reifyscene_render_proof", "resolution_factor": ds["factor"]},
            extra={"run_dir": str(out_dir), "iters": iters, "num_gaussians": field.num_gaussians},
        )
        bench_path = write_record(record, out_dir / "reference_benchmark.json")
        summary["benchmark"] = {k: metrics[k] for k in ("masked_mpsnr", "masked_mssim", "masked_mlpips", "num_views")}
        print(
            f"  masked mPSNR={metrics['masked_mpsnr']:.4f}  mSSIM={metrics['masked_mssim']:.4f}  "
            f"mLPIPS={metrics['masked_mlpips']:.4f}  (n={metrics['num_views']})"
        )
        print(f"  reference_benchmark_v1 -> {bench_path}")
        compare_to = args.compare_to or bm.get("compare_to")
        if compare_to:
            compare(compare_to, bench_path)

    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'train_summary.json'}")


# -- cli ------------------------------------------------------------------

def cmd_dynamic(args) -> None:
    cfg = read_yaml(args.config)["render_proof"]
    ds = cfg["dataset"]
    device = args.device or cfg.get("device", "cuda")
    iters = args.iters or cfg["dynamic"]["iters"]
    factor = int(ds["factor"])
    k_scale = 1.0 / factor  # borrowed cameras are full-res (factor 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.native:
        # Native init lifts at full res (factor 1) — 2D tracks + aligned depth
        # ship only at 1x and the bundle is full-res like the borrow; the render
        # factor is applied via k_scale below, exactly as for borrowed cameras.
        tracks_dir = None
        if args.own_tracker:
            # Drop the last borrowed input: generate the 2D tracks with reifyscene's
            # own tapnet tracker. Reuse an existing dir (the all-pairs pass is
            # expensive) unless it is empty.
            from reifyscene.lifting.native_track_lift import generate_native_tracks

            tracks_root = Path(args.own_tracks_dir) if args.own_tracks_dir else out_dir / "own_tracks"
            factor_dir = tracks_root / "1x"
            if factor_dir.exists() and any(factor_dir.glob("*.npy")):
                print(f"own-tracker: reusing existing tracks at {factor_dir}")
                tracks_dir = factor_dir
            else:
                print(f"own-tracker: generating 2D tracks -> {factor_dir}")
                tracks_dir = generate_native_tracks(
                    args.dataset_root, tracks_root, factor=1,
                    camera_type=args.camera_type, tracker_cfg={"device": device},
                )
            tracks_dir = str(tracks_dir)
        scene, cams = build_native_init(
            args.dataset_root, factor=1, num_fg=args.num_fg, num_bg=args.num_bg,
            num_bases=args.num_bases, camera_type=args.camera_type, device=device, seed=args.seed,
            tracks_dir=tracks_dir,
        )
        kind = "native+own-tracker" if args.own_tracker else "native"
    else:
        if not args.init:
            raise SystemExit("dynamic mode needs --init <bundle.npz> (or --native)")
        scene, cams = load_borrowed_init(args.init, device=device)
        kind = "borrowed"
    print(
        f"{kind} init: fg={scene.fg.num_gaussians} bg={scene.bg.num_gaussians if scene.bg else 0} "
        f"bases={scene.fg.num_bases} frames={scene.num_frames}"
    )
    backend = DatasetGeometryInitializer({"dataset_root": args.dataset_root, "factor": factor, "split": "train"})
    rasterizer = build_rasterizer({"device": device})

    def _view(cam):
        gt_u8 = backend.load_rgb(cam.frame_name)
        height, width = gt_u8.shape[:2]
        camera = cam.render_camera(width, height, k_scale=k_scale)
        gt = torch.as_tensor(gt_u8, dtype=torch.float32, device=device) / 255.0
        return camera, gt, gt_u8

    train_views = [(_view(cam)[0], _view(cam)[1], cam.time_id) for cam in cams["train"]]
    print(f"train views: {len(train_views)}; training {iters} iters")

    train_cfg = DynamicTrainConfig.from_cfg({**cfg["dynamic"], "iters": iters})
    # Schedule fields the config omits are derived from `iters` (SoM-shaped), so
    # the opacity reset always lands with recovery + polish — never on the final step.
    densify_cfg = DensifyConfig.from_cfg(cfg.get("densify"), max_iters=iters, num_frames=scene.num_frames)
    trainer = DynamicGaussianTrainer(scene, train_cfg, densify_cfg, device=device, rasterizer=rasterizer)
    trainer.fit(train_views, on_log=lambda r: print(f"  iter {r['step']:>6}  loss={r['loss']:.4f}  psnr={r['psnr']:.2f}  N={r['num_gaussians']}"))

    summary = {"mode": "dynamic", "iters": iters, "final_gaussians": scene.num_gaussians, "checkpoint3": []}

    def _panel_for(cam, tag):
        camera, gt, gt_u8 = _view(cam)
        pred = trainer.render_at(camera, cam.time_id)
        render_u8 = np.clip(pred.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        psnr = _psnr(gt_u8, render_u8)
        _panel(gt_u8, render_u8, f"{cam.frame_name} [{tag}]", f"psnr={psnr:.2f}").save(
            out_dir / f"checkpoint3_{tag}_{cam.frame_name}.png"
        )
        print(f"  {tag} {cam.frame_name}: psnr={psnr:.2f} dB")
        summary["checkpoint3"].append({"frame": cam.frame_name, "kind": tag, "psnr": psnr})

    print("Checkpoint 3 (dynamic train-view + held-out):")
    _panel_for(cams["train"][0], "train")
    for cam in cams["val"][:2]:
        _panel_for(cam, "novel")

    bm = cfg.get("benchmark", {}) or {}
    if bm.get("enabled", True):
        val_cams = cams["val"]
        if bm.get("max_eval_frames"):
            val_cams = val_cams[: int(bm["max_eval_frames"])]
        print(f"Benchmark: covisibility-masked NVS over {len(val_cams)} val frames...")

        def _val_views():
            for cam in val_cams:
                gt_u8 = backend.load_rgb(cam.frame_name)
                height, width = gt_u8.shape[:2]
                camera = cam.render_camera(width, height, k_scale=k_scale)
                gt = torch.as_tensor(gt_u8, dtype=torch.float32, device=device) / 255.0
                mask = torch.as_tensor(backend.load_covisible(cam.frame_name, "val"), device=device)
                yield camera, gt, mask, cam.time_id

        metrics = score_masked_dynamic_views(rasterizer, scene, _val_views(), device=device, lpips_net=bm.get("lpips_net", "alex"))
        record = build_record(
            method=bm.get("method_name", "reifyscene-dynamic"),
            sequence=Path(args.dataset_root).name,
            dataset_format=bm.get("dataset_format", "iphone"),
            metrics=metrics,
            eval={"source": "reifyscene_render_proof_dynamic", "resolution_factor": factor},
            extra={"run_dir": str(out_dir), "iters": iters, "num_gaussians": scene.num_gaussians},
        )
        bench_path = write_record(record, out_dir / "reference_benchmark.json")
        summary["benchmark"] = {k: metrics[k] for k in ("masked_mpsnr", "masked_mssim", "masked_mlpips", "num_views")}
        print(
            f"  masked mPSNR={metrics['masked_mpsnr']:.4f}  mSSIM={metrics['masked_mssim']:.4f}  "
            f"mLPIPS={metrics['masked_mlpips']:.4f}  (n={metrics['num_views']})"
        )
        compare_to = args.compare_to or bm.get("compare_to")
        if compare_to:
            compare(compare_to, bench_path)

    # Flythrough: render the dynamic scene along a hand-authored path while time
    # advances, and encode flythrough.mp4 (the renderer-track deliverable).
    fly = cfg.get("flythrough", {}) or {}
    if fly.get("enabled", True):
        fps = int(fly.get("fps", 24))
        num = int(fly.get("num_frames", 120))
        with torch.no_grad():
            center = scene.cloud_at(0).means.mean(dim=0).cpu().numpy()
        centers, ups = [], []
        for c in cams["train"]:
            rot, trans = c.w2c[:3, :3].astype(np.float32), c.w2c[:3, 3].astype(np.float32)
            centers.append(-rot.T @ trans)
            ups.append(rot.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32))
        centers = np.stack(centers)
        up = np.mean(ups, axis=0)
        radius = float(np.mean(np.linalg.norm(centers - center, axis=1)))
        start_dir = np.mean(centers, axis=0) - center

        ref = cams["train"][len(cams["train"]) // 2]
        K = ref.K.astype(np.float32).copy()
        K[:2] *= k_scale
        sample = backend.load_rgb(cams["train"][0].frame_name)
        height, width = sample.shape[:2]

        if fly.get("path", "orbit") == "replay":
            path = []
            for c in cams["train"]:
                rot, trans = c.w2c[:3, :3].astype(np.float32), c.w2c[:3, 3].astype(np.float32)
                eye = -rot.T @ trans
                path.append({
                    "eye": eye.tolist(),
                    "target": (eye + rot.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).tolist(),
                    "up": (rot.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)).tolist(),
                })
        else:
            path = orbit_cameras(center, radius, num, up=up, start_dir=start_dir, arc_degrees=float(fly.get("arc_degrees", 90.0)))

        print(f"flythrough: rendering {len(path)} frames ({fly.get('path', 'orbit')} path)...")
        frames = []
        with torch.no_grad():
            for i, pose in enumerate(path):
                time_id = int(round(i / max(len(path) - 1, 1) * (scene.num_frames - 1)))
                camera = RenderCamera.from_look_at(pose["eye"], pose["target"], pose["up"], width, height, intrinsics=K)
                cloud = scene.cloud_at(time_id)
                out = rasterizer.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera)
                frames.append(np.clip(out.rgb.cpu().numpy() * 255.0, 0, 255).astype(np.uint8))
        fly_path = write_video(frames, out_dir / "flythrough.mp4", fps=fps)
        summary["flythrough"] = {"path": fly.get("path", "orbit"), "frames": len(frames), "fps": fps}
        print(f"  flythrough -> {fly_path}")

    (out_dir / "dynamic_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'dynamic_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="reifyscene render-proof harness (Track 2)")
    sub = parser.add_subparsers(dest="mode", required=True)

    c0 = sub.add_parser("checkpoint0", help="coordinate sanity (untrained splat)")
    c0.add_argument("--dataset-root", required=True)
    c0.add_argument("--factor", type=int, default=2)
    c0.add_argument("--split", default="train")
    c0.add_argument("--seed-frames", default="0:0")
    c0.add_argument("--render-targets", default="0:0,1:0,2:0,3:0")
    c0.add_argument("--pixel-stride", type=int, default=1)
    c0.add_argument("--scale-mult", type=float, default=1.5)
    c0.add_argument("--max-points", type=int, default=1_500_000)
    c0.add_argument("--device", default="cuda")
    c0.add_argument("--out-dir", default="checkpoint0_out")
    c0.set_defaults(func=cmd_checkpoint0)

    tr = sub.add_parser("train", help="static Gaussian training (Checkpoints 1 & 2)")
    tr.add_argument("--dataset-root", required=True)
    tr.add_argument("--config", default="configs/render/render_proof.yaml")
    tr.add_argument("--iters", type=int, default=None, help="override trainer.iters (smoke)")
    tr.add_argument("--device", default=None)
    tr.add_argument("--out-dir", default="train_out")
    tr.add_argument("--compare-to", default=None, help="SoM reference_benchmark.json to --compare against")
    tr.set_defaults(func=cmd_train)

    dy = sub.add_parser("dynamic", help="dynamic Gaussian training from a borrowed or native init (Checkpoint 3)")
    dy.add_argument("--dataset-root", required=True)
    dy.add_argument("--init", default=None, help="borrowed motion-init .npz (scripts/borrow_som_motion_init.py)")
    dy.add_argument("--native", action="store_true", help="build the motion init natively instead of loading --init")
    dy.add_argument("--own-tracker", action="store_true",
                    help="native: produce the 2D tracks with reifyscene's own tapnet tracker (drops the last borrowed input)")
    dy.add_argument("--own-tracks-dir", default=None,
                    help="native+own-tracker: dir to write/reuse generated tracks (default: <out-dir>/own_tracks)")
    dy.add_argument("--num-fg", type=int, default=40_000, help="native: foreground tracks to lift")
    dy.add_argument("--num-bg", type=int, default=100_000, help="native: background points to sample")
    dy.add_argument("--num-bases", type=int, default=10, help="native: SE(3) motion bases (clusters)")
    dy.add_argument("--camera-type", default="refined", choices=["refined", "original"], help="native: camera source")
    dy.add_argument("--seed", type=int, default=None, help="native: RNG seed (track sampling + KMeans)")
    dy.add_argument("--config", default="configs/render/render_proof_windmill_dynamic.yaml")
    dy.add_argument("--iters", type=int, default=None)
    dy.add_argument("--device", default=None)
    dy.add_argument("--out-dir", default="dynamic_out")
    dy.add_argument("--compare-to", default=None)
    dy.set_defaults(func=cmd_dynamic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

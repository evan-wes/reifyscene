"""Render-proof harness for the room4d Gaussian renderer (Track 2).

Checkpoint 0 (coordinate sanity): load a DyCheck-format sequence's poses/depth
through the real DatasetGeometryInitializer -> GeometryArtifact, seed an
untrained colored point-cloud splat from one (or more) views' depth, then
render at real dataset cameras and write GT-vs-render side-by-side panels.

PASS = the render is recognizable and correctly placed (not mirrored, flipped,
or wrong-scale). The seed-camera target is a same-view reconstruction round-trip;
a held-out camera target is the real extrinsic/world-frame convention test.

Example (NVIDIA Balloon1, factor 2, seed cam0@t0, render cam0@t0 + cam1@t0):

    python scripts/run_render_proof.py \
        --dataset-root ../data/shape_of_motion/nvidia/Balloon1 \
        --factor 2 --seed-frames 0:0 --render-targets 0:0,1:0 \
        --out-dir checkpoint0_balloon1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from room4d.common.types import FrameRef
from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer
from room4d.render.gaussian4d.seed import seed_cloud_from_geometry


def _parse_frames(spec: str) -> list[tuple[str, int]]:
    """Parse "cam:time,cam:time" into [(camera_id, time_id), ...]."""
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
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def _label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, max(60, 7 * len(text)), 14), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return img


def _panel(gt: np.ndarray, render: np.ndarray, title: str, psnr: float, coverage: float) -> Image.Image:
    diff = np.abs(gt.astype(np.int16) - render.astype(np.int16)).clip(0, 255).astype(np.uint8)
    gt_img = _label(Image.fromarray(gt), "GT")
    render_img = _label(Image.fromarray(render), f"render psnr={psnr:.1f} cov={coverage:.2f}")
    diff_img = _label(Image.fromarray(diff), "|diff|")
    h = gt.shape[0]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    strip = np.hstack([np.asarray(gt_img), sep, np.asarray(render_img), sep, np.asarray(diff_img)])
    out = Image.fromarray(strip)
    return _label(out, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="room4d render-proof Checkpoint 0 (coordinate sanity)")
    parser.add_argument("--dataset-root", required=True, help="DyCheck-format sequence directory")
    parser.add_argument("--factor", type=int, default=2, help="resolution downsample factor (rgb/depth {factor}x)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed-frames", default="0:0", help='"cam:time,..." views to backproject for the seed cloud')
    parser.add_argument("--render-targets", default="0:0,1:0", help='"cam:time,..." cameras to render + compare')
    parser.add_argument("--pixel-stride", type=int, default=1, help="subsample stride when backprojecting depth")
    parser.add_argument("--scale-mult", type=float, default=1.5, help="splat radius multiplier (pixel footprint)")
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="checkpoint0_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = DatasetGeometryInitializer(
        {"dataset_root": args.dataset_root, "factor": args.factor, "split": args.split}
    )

    # Seed: build a GeometryArtifact over the seed frames and backproject depth.
    seed_specs = _parse_frames(args.seed_frames)
    seed_frame_refs = [
        FrameRef(stream_id=cam, frame_idx=time_id, timestamp_sec=float(time_id))
        for cam, time_id in seed_specs
    ]
    geometry = backend.initialize(seed_frame_refs)
    rgb_by_frame = {
        f"{cam}:{time_id}": backend.load_rgb(backend.frame_name(cam, time_id))
        for cam, time_id in seed_specs
    }
    cloud = seed_cloud_from_geometry(
        geometry,
        rgb_by_frame,
        device=args.device,
        pixel_stride=args.pixel_stride,
        max_points=args.max_points,
        scale_mult=args.scale_mult,
    )
    print(f"seed cloud: {len(cloud)} gaussians from {len(seed_specs)} view(s) {seed_specs}")

    rasterizer = build_rasterizer({"device": args.device})

    seed_keys = {f"{cam}:{time_id}" for cam, time_id in seed_specs}
    summary = {
        "dataset_root": str(args.dataset_root),
        "factor": args.factor,
        "seed_frames": args.seed_frames,
        "num_seed_gaussians": len(cloud),
        "targets": [],
    }
    for cam, time_id in _parse_frames(args.render_targets):
        name = backend.frame_name(cam, time_id)
        intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
        camera = RenderCamera.from_world_cam(t_world_cam, intrinsics, width, height)
        out = rasterizer.render(cloud, camera)
        render_rgb = np.clip(out["rgb"] * 255.0, 0, 255).astype(np.uint8)
        gt_rgb = backend.load_rgb(name)
        coverage = float(np.mean(out["alpha"] > 0.5))
        psnr = _psnr(gt_rgb, render_rgb)
        kind = "reconstruction" if f"{cam}:{time_id}" in seed_keys else "novel-view"
        title = f"cam{cam} t{time_id} [{kind}]"
        panel = _panel(gt_rgb, render_rgb, title, psnr, coverage)
        panel_path = out_dir / f"checkpoint0_cam{cam}_t{time_id}.png"
        panel.save(panel_path)
        print(f"  {title}: psnr={psnr:.2f} dB  coverage={coverage:.3f}  -> {panel_path}")
        summary["targets"].append(
            {"camera": cam, "time": time_id, "kind": kind, "psnr": psnr, "coverage": coverage, "panel": str(panel_path)}
        )

    (out_dir / "checkpoint0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'checkpoint0_summary.json'}")


if __name__ == "__main__":
    main()

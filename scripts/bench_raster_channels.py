#!/usr/bin/env python
"""Time the rasterizer with D feature channels alongside RGB -- forward + backward.

    python scripts/bench_raster_channels.py                                  # synthetic cloud
    python scripts/bench_raster_channels.py \
        --checkpoint ../runs/loop_blackswan/render/trained_dynamic.npz
    python scripts/bench_raster_channels.py --dims 0 13 16 20 28

Why this exists: the identity feature's width is a cost choice, and gsplat 1.5.3 only has
kernels for particular channel counts (1-5, 8, 9, 16, 17, 24, 32, 33, 64, ...). Anything else
is zero-padded up to the next power of two inside ``rasterize_to_pixels``. Our call renders
``RGB+ED``, so the channel count is ``3 + D + 1``: D=13 -> 17 and D=20 -> 24 run unpadded,
while D=16 -> 20 pads to 32. The table shows which is which.

Timing is CUDA events around one ``rasterize`` + ``backward`` through ``GsplatRasterizer``,
the same path training takes, after warm-up; median over ``--iters``. The first
measurement in the identity scope (§8) was taken ad hoc and never committed; this replaces it.

Needs CUDA. Check the GPU power cap first (it resets on reboot).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SUPPORTED = (1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513)


def _kernel_channels(c: int) -> int:
    return c if c in SUPPORTED else 1 << (c - 1).bit_length()


def _synthetic(n: int, width: int, height: int):
    import torch

    from reifyscene.render.gaussian4d.rasterizer import RenderCamera

    g = torch.Generator(device="cuda").manual_seed(0)
    means = torch.randn((n, 3), device="cuda", generator=g) * torch.tensor(
        [1.0, 1.0, 0.5], device="cuda"
    )
    quats = torch.nn.functional.normalize(torch.randn((n, 4), device="cuda", generator=g), dim=-1)
    scales = torch.full((n, 3), 0.02, device="cuda")
    opac = torch.full((n,), 0.7, device="cuda")
    colors = torch.rand((n, 3), device="cuda", generator=g)
    cam = RenderCamera.from_look_at([4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                    width, height)
    return means, quats, scales, opac, colors, cam


def _from_checkpoint(path: Path):
    import torch

    from reifyscene.ingest.streams import RunStreams
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
    from reifyscene.render.gaussian4d.rasterizer import RenderCamera

    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    scene, _ = bundle_to_scene(data, device="cuda")
    i = len(data["train_time_ids"]) // 2
    K = np.asarray(data["train_Ks"][i], np.float32)
    w, h = RunStreams.from_run(Path(path).parent.parent).only().render_raster  # <run>/<sub>/
    cam = RenderCamera(viewmat=np.asarray(data["train_w2cs"][i], np.float32), K=K,
                       width=w, height=h)
    with torch.no_grad():
        c = scene.cloud_at(int(data["train_time_ids"][i]))
    return c.means, c.quats, c.scales, c.opacities, c.colors, cam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="trained_dynamic.npz; default synthetic")
    ap.add_argument("--gaussians", type=int, default=80_000, help="synthetic cloud size")
    ap.add_argument("--size", type=int, nargs=2, default=(518, 294), metavar=("W", "H"))
    ap.add_argument("--dims", type=int, nargs="+", default=[0, 13, 16, 19, 20, 28])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    import torch

    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer, require_gsplat

    require_gsplat()
    if args.checkpoint:
        means, quats, scales, opac, colors, cam = _from_checkpoint(args.checkpoint)
        source = args.checkpoint.parent.parent.name
    else:
        means, quats, scales, opac, colors, cam = _synthetic(args.gaussians, *args.size)
        source = "synthetic"
    n = int(means.shape[0])
    r = GsplatRasterizer("cuda")
    print(f"{source}: {n} Gaussians, {cam.width}x{cam.height}, gsplat "
          f"{__import__('gsplat').__version__}, {torch.cuda.get_device_name()}")
    print(f"{'D':>3} {'channels':>8} {'kernel':>6} {'fwd+bwd ms':>10} {'vs first':>8}")

    base = None
    for d in args.dims:
        colors_p = colors.detach().clone().requires_grad_(True)
        feats = torch.zeros((n, d), device="cuda", requires_grad=True) if d else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for it in range(args.warmup + args.iters):
            start.record()
            out = r.rasterize(means, quats, scales, opac, colors_p, cam, features=feats)
            loss = out.rgb.sum() + (out.features.sum() if feats is not None else 0.0)
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            if it >= args.warmup:
                times.append(start.elapsed_time(end))
        ms = statistics.median(times)
        base = ms if base is None else base
        c = 3 + d + 1
        print(f"{d:>3} {c:>8} {_kernel_channels(c):>6} {ms:>10.2f} {ms / base:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""How well does a trained scene reproduce its own footage, and how sharp is it?

    python scripts/eval_train_views.py ../runs/loop_lab-coat --sub render_control

Renders every training frame of ``<run>/<sub>/trained_dynamic.npz`` and compares it with the
photo the scene stage trained on. Reports, averaged over all frames:

- **PSNR** and **SSIM** -- fidelity. (The render stage's own ``train_view_psnr`` averages only
  the first 8 frames.)
- **sharpness** -- the render's high-frequency energy relative to the photo's: variance of the
  Laplacian, render over photo. 1.0 means as much fine detail as the footage; a soft,
  blobby reconstruction sits well below it. This is the quantity "too blurry to judge" is about,
  and PSNR does not capture it well (a blur can score decent PSNR).

Writes ``<run>/<sub>/train_eval.json`` and full-size stills ``train_eval_<frame>.png`` (photo
above render) for ``--stills`` evenly spaced frames. Needs CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _laplacian_var(img: np.ndarray) -> float:
    g = img.astype(np.float64).mean(-1)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def evaluate(run: Path, sub: str, stills: int) -> dict:
    import torch
    from PIL import Image

    from reifyscene.render.gaussian4d.gaussian_trainer import _to_bchw, image_psnr, ssim
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
    from reifyscene.render.gaussian4d.rasterizer import build_rasterizer
    from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend
    from reifyscene.scene.scene_artifact import SceneArtifact

    with np.load(run / sub / "trained_dynamic.npz", allow_pickle=True) as z:
        scene, _ = bundle_to_scene({k: z[k] for k in z.files}, device="cuda")
    backend = GaussianRenderBackend({"device": "cuda"})
    views = backend._dynamic_train_views(SceneArtifact.load(run / "scene"))
    keys = backend._train_frame_keys
    rast = build_rasterizer({"device": "cuda"})
    psnrs, ssims, sharp = [], [], []
    pick = set(np.linspace(0, len(views) - 1, stills).round().astype(int).tolist())
    for i, (cam, gt, t) in enumerate(views):
        with torch.no_grad():
            c = scene.cloud_at(t)
            rgb = rast.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, cam).rgb
            rgb = rgb.clamp(0, 1)
            psnrs.append(float(image_psnr(rgb, gt)))
            ssims.append(float(ssim(_to_bchw(rgb), _to_bchw(gt))))
        r8 = (rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8)
        g8 = (gt.cpu().numpy() * 255 + 0.5).astype(np.uint8)
        sharp.append(_laplacian_var(r8) / max(_laplacian_var(g8), 1e-9))
        if i in pick:
            frame = str(keys[i]).rsplit(":", 1)[-1]
            Image.fromarray(np.concatenate([g8, r8], axis=0)).save(
                run / sub / f"train_eval_{int(frame):05d}.png")
    res = {"run": run.name, "sub": sub, "frames": len(views),
           "psnr": round(float(np.mean(psnrs)), 3), "ssim": round(float(np.mean(ssims)), 4),
           "sharpness": round(float(np.mean(sharp)), 4),
           "num_gaussians": int(scene.num_gaussians)}
    (run / sub / "train_eval.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--sub", default="render")
    ap.add_argument("--stills", type=int, default=3)
    args = ap.parse_args()
    print(f"{'run':<22} {'sub':<28} {'frames':>6} {'PSNR':>6} {'SSIM':>6} {'sharp':>6} "
          f"{'Gaussians':>10}")
    for run in args.run_dirs:
        r = evaluate(run.resolve(), args.sub, args.stills)
        print(f"{r['run']:<22} {r['sub']:<28} {r['frames']:>6} {r['psnr']:>6.2f} {r['ssim']:>6.3f} "
              f"{r['sharpness']:>6.3f} {r['num_gaussians']:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

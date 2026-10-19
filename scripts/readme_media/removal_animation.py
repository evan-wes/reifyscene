#!/usr/bin/env python
"""The README's removal animations: before | after | extracted entity, side by side.

    python scripts/readme_media/removal_animation.py RUN NODE_ID OUT_DIR --name NAME

Writes ``OUT_DIR/NAME_clip.webp`` -- the clip's frames along its own camera path, the three
panels in step -- and ``OUT_DIR/NAME_orbit.webp`` -- one moment (the middle frame) seen while the
camera swings +-20 degrees in yaw about the scene's pivot, a sine so the loop is seamless. The
entity is removed by the checkpoint's default method (by owner on an owned checkpoint), exactly
as the app removes it; "extracted entity" is what the removal took, rendered alone on black.

Needs CUDA. Animated WebP, which GitHub shows like a GIF at about a sixth of the size.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LABELS = ("before", "after", "extracted entity")


def _font(size: int):
    from matplotlib import font_manager

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)


def _panel(img: np.ndarray, label: str, width: int, font) -> Image.Image:
    im = Image.fromarray(img)
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    draw.text((11, 7), label, font=font, fill=(0, 0, 0))
    draw.text((10, 6), label, font=font, fill=(255, 255, 255))
    return im


def side_by_side(images, width: int, font, gap: int = 4) -> Image.Image:
    """The three renders as labelled panels, ``width`` each, left to right."""
    panels = [_panel(i, lab, width, font) for i, lab in zip(images, LABELS, strict=True)]
    out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1),
                            panels[0].height), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def write_webp(frames: list[Image.Image], path: Path, fps: float) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                   loop=0, quality=80, method=6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run, by name in runs/")
    ap.add_argument("node", help="the entity's node id")
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", required=True, help="file name stem")
    ap.add_argument("--width", type=int, default=360, help="each panel's width (px)")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--orbit-steps", type=int, default=40)
    ap.add_argument("--orbit-fps", type=float, default=15)
    ap.add_argument("--yaw", type=float, default=20.0, help="orbit amplitude (degrees)")
    args = ap.parse_args()

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.graph.renderable_handle import drop_gaussians
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    args.out.mkdir(parents=True, exist_ok=True)
    editor = RemoveEditor(RunOutputs.open(data_dir.path("runs", args.run)).checkpoint(),
                          device="cuda")
    r = editor.renderer
    fg, bg, _ = editor.select(args.node, method=editor.default_method)
    after, _ = bundle_to_scene(drop_gaussians(editor.bundle, fg, bg_indices=bg), device="cuda")
    n_fg = int(np.asarray(editor.bundle["fg_means"]).shape[0])
    n_bg = int(np.asarray(editor.bundle["bg_means"]).shape[0])
    alone, _ = bundle_to_scene(drop_gaussians(editor.bundle, np.setdiff1d(np.arange(n_fg), fg),
                                              bg_indices=np.setdiff1d(np.arange(n_bg), bg)),
                               device="cuda")
    font = _font(max(14, args.width // 28))

    def frame(f: int, camera) -> Image.Image:
        return side_by_side([r.render_view(s, f, camera) for s in (r.scene, after, alone)],
                            args.width, font)

    clip = [frame(f, r.orbit_camera(f)) for f in r.frames]          # yaw 0: the recorded camera
    write_webp(clip, args.out / f"{args.name}_clip.webp", args.fps)
    mid = r.frames[len(r.frames) // 2]
    yaws = [args.yaw * math.sin(2 * math.pi * k / args.orbit_steps)
            for k in range(args.orbit_steps)]
    orbit = [frame(mid, r.orbit_camera(mid, yaw=y)) for y in yaws]
    write_webp(orbit, args.out / f"{args.name}_orbit.webp", args.orbit_fps)
    for kind in ("clip", "orbit"):
        p = args.out / f"{args.name}_{kind}.webp"
        print(f"{p}  {p.stat().st_size / 1e6:.2f} MB  {clip[0].width}x{clip[0].height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""The README's viewer animation, step 2: the scene-panel screenshots of ``capture_viewer.mjs``,
cropped to their content -- what differs from the panel's own background on any frame, in either
theme -- and written as one animated WebP.

    python scripts/readme_media/viewer_animation.py FRAMES_DIR OUT.webp [--width 900] [--fps 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="the screenshots, in name order")
    ap.add_argument("out", type=Path)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--fps", type=float, default=4)
    args = ap.parse_args()

    shots = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(args.frames.glob("*.png"))]
    if not shots:
        raise SystemExit(f"no screenshots in {args.frames}")
    # the panel's background: the commonest colour just inside its 1 px border
    inner = shots[0][3:-3, 3:-3]
    edge = np.concatenate([inner[0], inner[-1], inner[:, 0], inner[:, -1]]).astype(np.int16)
    colours, counts = np.unique(edge, axis=0, return_counts=True)
    background = colours[np.argmax(counts)]
    content = np.zeros(shots[0].shape[:2], bool)
    for shot in shots:
        content |= (np.abs(shot.astype(np.int16) - background) > 10).any(-1)
    content[:, :3] = content[:, -3:] = content[:3, :] = content[-3:, :] = False   # the border
    ys, xs = np.nonzero(content)
    y0, y1 = max(int(ys.min()) - 8, 0), min(int(ys.max()) + 10, shots[0].shape[0])
    x0, x1 = max(int(xs.min()) - 8, 0), min(int(xs.max()) + 10, shots[0].shape[1])
    frames = []
    for shot in shots:
        im = Image.fromarray(shot[y0:y1, x0:x1])
        frames.append(im.resize((args.width, round(im.height * args.width / im.width)),
                                Image.LANCZOS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=round(1000 / args.fps), loop=0, quality=85, method=6)
    print(f"{args.out}  {args.out.stat().st_size / 1e6:.2f} MB  {frames[0].width}x"
          f"{frames[0].height}, {len(frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

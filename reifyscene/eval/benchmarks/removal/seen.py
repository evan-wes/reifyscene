"""Which background the video saw behind an entity, and what it looked like (tests T3, T4).

For a frame *t*, every other training frame's background pixels -- id map background, valid
geometry depth, at least ``erode_px`` from anything that is not background, because a mask edge
is where segmentation is least sure -- are lifted with that frame's depth and camera and
projected into *t*. Each lands on every pixel whose centre is less than a pixel from where it
projects in both directions (up to 2x2; so a surface seen slightly closer elsewhere does not
leave holes), z-buffered:

- a point **in front of** *t*'s own surface at the pixel (geometry depth, ``tol`` relative) is
  dropped: nothing can be in front of what the photo at *t* shows there, so it is a geometry
  disagreement, not background;
- of the rest, the nearest surface wins (``zmin``), and its colour is taken from the source
  frame nearest in time among those whose point lies within ``tol`` of it.

A pixel of the entity's mask that received a colour is **seen** (T3 scores the after-render
there against that colour); one that did not is **unseen** (T4: its share). The same projection
on background pixels around the entity (:func:`band`), where nothing occludes, is the **geometry
floor**: how well the projection predicts what is visible, which bounds how well T3 can do on
that clip.

Limit, by construction: only background is projected. Where another entity stands behind the
removed one, the expected colour is the background behind both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reifyscene.eval.benchmarks.edit_damage.score import dilate
from reifyscene.eval.benchmarks.removal.views import RunViews
from reifyscene.graph.id_maps import BACKGROUND


@dataclass
class Projected:
    """One frame's background as the other frames saw it."""

    colour: np.ndarray        # float32 (H, W, 3) in [0, 1]; NaN where no frame saw it
    source: np.ndarray        # int32 (H, W): the source frame, -1 where unseen

    @property
    def seen(self) -> np.ndarray:
        return self.source >= 0


def background_points(views: RunViews, frame: int, *, erode_px: int = 2):
    """``(xyz (P, 3) world, rgb (P, 3) float [0, 1])`` of ``frame``'s eroded background."""
    bg = views.id_maps[frame] == BACKGROUND
    keep = bg & ~dilate(~bg, erode_px) & (views.depth[frame] > 0)
    v, u = np.nonzero(keep)
    z = views.depth[frame][v, u].astype(np.float64)
    K = views.K[frame]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    c2w = np.linalg.inv(views.w2c[frame])
    xyz = cam @ c2w[:3, :3].T + c2w[:3, 3]
    rgb = views.photos[frame][v, u].astype(np.float32) / 255.0
    return xyz.astype(np.float32), rgb


class BackgroundProjector:
    """Projects every frame's background into any frame (:class:`Projected`), on the GPU."""

    def __init__(self, views: RunViews, *, device: str = "cuda", erode_px: int = 2,
                 tol: float = 0.05) -> None:
        import torch

        self.views, self.device, self.tol = views, device, float(tol)
        self._points: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for f in views.frames:
            xyz, rgb = background_points(views, f, erode_px=erode_px)
            self._points[f] = (torch.as_tensor(xyz, device=device),
                               torch.as_tensor(rgb, device=device))

    def _splat(self, frame: int, source: int):
        """``(pixel index, z, rgb)`` of ``source``'s points on ``frame``'s pixels round each,
        those in front of ``frame``'s own surface dropped."""
        import torch

        v = self.views
        h, w = v.shape
        xyz, rgb = self._points[source]
        w2c = torch.as_tensor(v.w2c[frame], dtype=torch.float32, device=self.device)
        K = torch.as_tensor(v.K[frame], dtype=torch.float32, device=self.device)
        cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        ok = z > 1e-6
        cam, z, rgb = cam[ok], z[ok], rgb[ok]
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        vv = K[1, 1] * cam[:, 1] / z + K[1, 2]
        front = torch.as_tensor(v.depth[frame], device=self.device).reshape(-1)
        idx, zs, cols = [], [], []
        for uu in (torch.floor(u).long(), torch.ceil(u).long()):
            for vvv in (torch.floor(vv).long(), torch.ceil(vv).long()):
                inb = (uu >= 0) & (uu < w) & (vvv >= 0) & (vvv < h)
                p = vvv[inb] * w + uu[inb]
                zz = z[inb]
                surface = front[p]
                behind = (surface <= 0) | (zz >= surface * (1.0 - self.tol))
                idx.append(p[behind])
                zs.append(zz[behind])
                cols.append(rgb[inb][behind])
        return torch.cat(idx), torch.cat(zs), torch.cat(cols)

    def project(self, frame: int) -> Projected:
        import torch

        v = self.views
        h, w = v.shape
        sources = sorted((f for f in v.frames if f != frame), key=lambda f: (abs(f - frame), f))
        zmin = torch.full((h * w,), float("inf"), device=self.device)
        for s in sources:
            p, z, _ = self._splat(frame, s)
            zmin.scatter_reduce_(0, p, z, reduce="amin")
        colour = torch.full((h * w, 3), float("nan"), device=self.device)
        src = torch.full((h * w,), -1, dtype=torch.int32, device=self.device)
        for s in sources:                                   # nearest in time first
            p, z, c = self._splat(frame, s)
            take = (z <= zmin[p] * (1.0 + self.tol)) & (src[p] < 0)
            colour[p[take]] = c[take]
            src[p[take]] = s
        return Projected(colour.reshape(h, w, 3).cpu().numpy(),
                         src.reshape(h, w).cpu().numpy())


def band(views: RunViews, node_id: str, frame: int, px: int, gap: int) -> np.ndarray:
    """Where the geometry floor is measured: background pixels within ``gap + px`` of the
    entity but at least ``gap`` from anything not background. Right at a mask's edge the photo
    still shows the entity (hair, a sleeve the mask missed) while the projection shows the
    background; on lab-coat a 2 px gap lifts the floor by 3.5 dB, and wider gaps change little."""
    bg = views.id_maps[frame] == BACKGROUND
    return dilate(views.mask(node_id, frame), gap + px) & ~dilate(~bg, gap) & bg


@dataclass
class SquaredError:
    """A running sum of squared error, for a PSNR pooled over pixels of many frames."""

    sse: float = 0.0
    n: int = 0

    def add(self, image: np.ndarray, expected: np.ndarray, where: np.ndarray) -> None:
        """``image`` uint8 against ``expected`` float [0, 1], over ``where``."""
        if where.any():
            d = image[where].astype(np.float64) / 255.0 - expected[where].astype(np.float64)
            self.sse += float((d * d).sum())
            self.n += int(d.size)

    def psnr(self) -> float | None:
        if not self.n:
            return None
        mse = self.sse / self.n
        return round(60.0 if mse <= 1e-6 else float(10.0 * np.log10(1.0 / mse)), 3)

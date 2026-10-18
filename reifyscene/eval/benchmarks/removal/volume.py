"""The space an entity occupies on each frame, and how much opacity is left in it (test T2).

- **People** with a body on that frame: the Sapiens body volume
  (:mod:`reifyscene.edit.body_volume`), built from the geometry's depth rather than a render's,
  so it does not move with the removal being judged.
- **Everything else**, and people on frames without a torso: a **slab** -- the entity's own
  pixels, from ``front`` (a margin in front of the geometry's surface there) to the surface plus
  the entity's own width on that frame. A slab, not a hull: a hull reaches into neighbours.

The opacity mass inside a volume is the summed opacity of the Gaussians, of both fields,
whose centres lie in it at that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reifyscene.eval.benchmarks.removal.views import RunViews

FRONT_MARGIN = 0.25            # of the entity's width: how far in front of its surface counts


def entity_width(views: RunViews, node_id: str, frame: int) -> float | None:
    """The entity's horizontal extent on ``frame`` in scene units (5th-95th percentile of its
    pixels' camera x), or None without enough valid depth."""
    m = views.mask(node_id, frame) & (views.depth[frame] > 0)
    if int(m.sum()) < 10:
        return None
    v, u = np.nonzero(m)
    z = views.depth[frame][v, u]
    K = views.K[frame]
    x = (u - K[0, 2]) / K[0, 0] * z
    lo, hi = np.percentile(x, [5, 95])
    return float(hi - lo) if hi > lo else None


@dataclass
class EntityVolume:
    """One entity's volume per frame; :meth:`inside` tests points against it."""

    views: RunViews
    node_id: str
    body: Any = None                          # BodyVolume, for a person
    widths: dict[int, float] = field(default_factory=dict)

    def kind(self, frame: int) -> str | None:
        if self.body is not None and frame in self.body.frames:
            return "body"
        return "slab" if frame in self.widths else None

    def inside(self, points: np.ndarray, frame: int) -> np.ndarray:
        """bool (N,): which ``points`` (world, (N, 3)) lie in the volume on ``frame``."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.zeros(len(pts), bool)
        kind = self.kind(frame)
        if kind is None:
            return out
        v = self.views
        h, w = v.shape
        cam = pts @ v.w2c[frame][:3, :3].T + v.w2c[frame][:3, 3]
        z = cam[:, 2]
        K = v.K[frame]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2])
            vv = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2])
        mask = v.mask(self.node_id, frame)
        if kind == "body":
            # Only points near the silhouette can be in the body: test those exactly.
            ys, xs = np.nonzero(mask)
            pad = 0.2 * max(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1) if len(xs) else 0
            near = ((z > 0) & (uu >= xs.min() - pad) & (uu <= xs.max() + pad)
                    & (vv >= ys.min() - pad) & (vv <= ys.max() + pad)) if len(xs) else z > 0
            idx = np.flatnonzero(near)
            out[idx] = self.body.distance(pts[idx], frame) <= 0.0
            return out
        inb = (z > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        idx = np.flatnonzero(inb)
        pu, pv = uu[idx].astype(int), vv[idx].astype(int)
        surface = v.depth[frame][pv, pu]
        width = self.widths[frame]
        hit = (mask[pv, pu] & (surface > 0) & (z[idx] >= surface - FRONT_MARGIN * width)
               & (z[idx] <= surface + width))
        out[idx[hit]] = True
        return out


def entity_volume(views: RunViews, node_id: str, *, body=None) -> EntityVolume:
    """``node_id``'s volume on every frame it has pixels; ``body`` its BodyVolume, if a person."""
    vol = EntityVolume(views, node_id, body=body)
    for f in views.present(node_id):
        width = entity_width(views, node_id, f)
        if width is not None:
            vol.widths[f] = width
    return vol


def person_body(run, renderer, views: RunViews, node_id: str):
    """The Sapiens body of a human node, from the geometry's depth; None for anything else, or
    when the run has no keypoints."""
    from reifyscene.edit.body_volume import sapiens_body
    from reifyscene.graph.renderable_handle import render_pixel_masks

    node = next((n for n in run.artifact("graph").nodes if n.node_id == node_id), None)
    if node is None or node.node_type != "human" or not run.has("humans"):
        return None
    stream = run.streams.resolve(renderer.stream_id)
    placed = render_pixel_masks(run, stream_id=stream.id)
    body = sapiens_body(renderer, run.artifact("humans"), node.attrs["track_id"],
                        placed.by_frame, node_id, stream, depth_by_frame=views.depth)
    return body if body.frames else None


def opacity_mass(volume: EntityVolume, positions: np.ndarray, opacities: np.ndarray,
                 frame: int, removed: np.ndarray) -> tuple[float, float]:
    """``(before, after)``: opacity inside the volume on ``frame``, of every Gaussian and of
    those the removal kept. ``removed`` bool (N,) over the combined (fg then bg) cloud."""
    inside = volume.inside(positions, frame)
    before = float(opacities[inside].sum())
    return before, float(opacities[inside & ~removed].sum())

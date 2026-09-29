"""Assign each Gaussian to an entity by the rendered evidence for it -- no training.

FlashSplat (Shen, Yang & Wang, ECCV 2024, arXiv:2409.08270) observes that once a Gaussian
scene is trained, rendering a mask is linear in the per-Gaussian labels, so the best labels
can be read off directly instead of learned. Each Gaussian accumulates, over every view, its
blending weight (``alpha * T``, its actual share of the pixel) into the class of every pixel
it touches; it then takes class *k* only if its evidence for *k* outweighs its evidence for
everything else combined, with a *slackness* bias toward "not *k*" -- each class decided on
its own. A Gaussian no view sees has no evidence and belongs to nothing.

That rule is what learned identity lacks. Its junk is mostly *supervised* Gaussians -- large
ones grazing an entity in some views while mostly covering something else (§4.6 of the
identity scope): the cross-entropy loss nudges each toward the pixels it touches and nothing
asks where most of its evidence lies.

Implemented here from the paper and a reading of their released code (whose rasterizer is
Inria-licensed and is not used): the evidence comes from our own gsplat wrapper. Rendering a
C-channel feature is linear in each Gaussian's feature, so for ``L = sum_pixels
onehot(label)[pixel] . F[pixel]`` the gradient with respect to Gaussian *i*'s feature is
exactly its accumulated blending weight per class. One backward per view. The scene is
dynamic, so each view renders the cloud at its own time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch

from reifyscene.graph.id_maps import IGNORE


def evidence_matrix(
    scene,
    rasterizer,
    views: Iterable[tuple],
    num_classes: int,
) -> torch.Tensor:
    """(N, C): each Gaussian's summed blending weight inside each class's pixels.

    ``views`` are ``(camera, time_id, id_map)`` with ``id_map`` an (H, W) integer tensor in
    the render raster; ignore pixels contribute to no class. Geometry is detached: this reads
    the scene, it never changes it.
    """
    n = scene.num_gaussians
    device = scene.fg.means.device
    counts = torch.zeros((n, num_classes), device=device)
    for camera, time_id, id_map in views:
        with torch.no_grad():
            cloud = scene.cloud_at(int(time_id))
        feats = torch.zeros((n, num_classes), device=device, requires_grad=True)
        out = rasterizer.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities,
                                   cloud.colors, camera, features=feats)
        valid = id_map != IGNORE
        onehot = torch.nn.functional.one_hot(
            torch.where(valid, id_map, torch.zeros_like(id_map)).long(), num_classes
        ).to(out.features.dtype) * valid[..., None]
        (grad,) = torch.autograd.grad((out.features * onehot).sum(), feats)
        counts += grad.detach()
    return counts


def assign(counts: torch.Tensor, *, slackness: float = 0.0) -> torch.Tensor:
    """(N, C) bool: does Gaussian *i* belong to class *k*? FlashSplat's per-class rule.

    Each class is decided on its own, as their solver does: Gaussian *i*'s evidence is split
    into [everything else, *k*], normalised, and it belongs to *k* if ``k > else + slackness``.
    Positive slackness asks for a clearer majority (precision). Negative admits Gaussians that
    are only partly *k* (completeness; their removal scripts use -0.8 to -0.98), so a
    Gaussian may then belong to several classes -- removing any of them takes it. At
    slackness >= 0 a Gaussian belongs to at most one class. No evidence, no class.
    """
    counts = counts.float()
    other = counts.sum(dim=1, keepdim=True) - counts
    pair = torch.nn.functional.normalize(torch.stack([other, counts], dim=-1), dim=-1)
    return (pair[..., 1] > pair[..., 0] + slackness) & (counts > 0)


def assignment_handles(
    member: torch.Tensor | np.ndarray,
    num_fg: int,
    node_ids: Sequence[str | None],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{node_id: (fg_indices, bg_indices)}`` from (N, C) membership over fg-then-bg."""
    m = member.cpu().numpy() if isinstance(member, torch.Tensor) else np.asarray(member)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, node in enumerate(node_ids):
        if node is None:
            continue
        idx = np.flatnonzero(m[:, k])
        out[node] = (idx[idx < num_fg], idx[idx >= num_fg] - num_fg)
    return out


def id_map_views(
    id_maps: Mapping[int, np.ndarray],
    cams_by_frame: Mapping[int, tuple],
    device: str,
) -> list[tuple]:
    """``(camera, time_id, id_map tensor)`` for every frame that has both a map and a camera."""
    return [
        (cam, t, torch.as_tensor(id_maps[f].astype(np.int64), device=device))
        for f, (cam, t) in sorted(cams_by_frame.items())
        if f in id_maps
    ]


def filter_members(
    scene,
    fg_idx: np.ndarray,
    bg_idx: np.ndarray,
    *,
    sor_k: int = 8,
    sor_std: float = 2.0,
    size_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Drop the floaters from a selection: isolated Gaussians, and oversized ones.

    The junk left in an evidence-majority selection on sharp scenes is needle-shaped and
    oversized Gaussians straddling the object's edge, and isolated specks. Two standard
    point-cloud clean-ups, applied per field in that field's own space (the foreground in its
    canonical pose, the background in the world -- they are never each other's neighbours):

    - **outlier** (statistical outlier removal): a Gaussian whose mean distance to its
      ``sor_k`` nearest selected neighbours exceeds the selection's mean by ``sor_std``
      standard deviations;
    - **oversized**: a Gaussian whose largest axis exceeds ``size_factor`` times the
      selection's median largest axis.

    Returns the kept ``(fg_idx, bg_idx)`` and how many each rule dropped.
    """
    from scipy.spatial import cKDTree

    stats = {"outlier": 0, "oversized": 0}
    kept = []
    fields = [(scene.fg, np.asarray(fg_idx, dtype=int))]
    fields.append((scene.bg, np.asarray(bg_idx, dtype=int)) if scene.bg is not None
                  else (None, np.asarray(bg_idx, dtype=int)))
    for field, idx in fields:
        if field is None or idx.size == 0:
            kept.append(idx)
            continue
        with torch.no_grad():
            pts = field.means.detach()[torch.as_tensor(idx)].cpu().numpy()
            size = field.get_scales().detach()[torch.as_tensor(idx)].amax(-1).cpu().numpy()
        keep = np.ones(idx.size, dtype=bool)
        big = size > size_factor * np.median(size)
        keep &= ~big
        stats["oversized"] += int(big.sum())
        if idx.size > sor_k:
            d, _ = cKDTree(pts).query(pts, k=sor_k + 1)
            mean_d = d[:, 1:].mean(1)
            far = mean_d > mean_d.mean() + sor_std * mean_d.std()
            stats["outlier"] += int((far & keep).sum())
            keep &= ~far
        kept.append(idx[keep])
    return kept[0], kept[1], stats

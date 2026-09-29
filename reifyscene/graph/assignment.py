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

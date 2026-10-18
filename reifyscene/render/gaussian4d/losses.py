# Portions adapted from Shape of Motion (https://github.com/vye16/shape-of-motion),
# Copyright (c) 2024 Vickie Ye, under the MIT License (LICENSES/MIT-shape-of-motion.txt).
"""Loss functions for the dynamic renderer: robust losses for the geometry terms of training.

Ported from Shape-of-Motion (``flow3d/loss_utils.py``, MIT, Copyright (c) 2024 Vickie Ye) so
the depth and mask terms behave exactly as they do there: an L1 that ignores the worst
``1 - quantile`` of pixels (a monocular depth prior is wrong somewhere in every frame, and a
trimmed loss lets it be), and the same on spatial gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trimmed_l1_loss(pred: torch.Tensor, gt: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    return loss[loss < loss_at_quantile].mean()


def masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None,
                   normalize: bool = True, quantile: float = 1.0) -> torch.Tensor:
    """Mean L1 over ``mask`` (weights), leaving out values above the ``quantile`` of the error.
    ``pred``/``gt`` (..., D); ``mask`` (..., 1)."""
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
    if quantile < 1:
        num = sum_loss.numel()
        if num < 16_000_000:
            threshold = torch.quantile(sum_loss, quantile)
        else:                                  # torch.quantile refuses very large inputs
            ordered, _ = torch.sort(sum_loss.reshape(-1))
            idxf = quantile * num
            idxi = int(idxf)
            threshold = ordered[idxi] + (ordered[idxi + 1] - ordered[idxi]) * (idxf - idxi)
        quantile_mask = (sum_loss < threshold).squeeze(-1)
    else:
        quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
    ndim = sum_loss.shape[-1]
    if normalize:
        return torch.sum((sum_loss * mask)[quantile_mask]) / (
            ndim * torch.sum(mask[quantile_mask]) + 1e-8)
    return torch.mean((sum_loss * mask)[quantile_mask])


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                  quantile: float = 0.98) -> torch.Tensor:
    """Trimmed L1 between the spatial gradients of ``pred`` and ``gt`` (B, H, W), over pixel
    pairs both inside ``mask`` (B, H, W)."""
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_x, pred_y = pred[:, :, 1:] - pred[:, :, :-1], pred[:, 1:, :] - pred[:, :-1, :]
    gt_x, gt_y = gt[:, :, 1:] - gt[:, :, :-1], gt[:, 1:, :] - gt[:, :-1, :]
    return (masked_l1_loss(pred_x[mask_x][..., None], gt_x[mask_x][..., None], quantile=quantile)
            + masked_l1_loss(pred_y[mask_y][..., None], gt_y[mask_y][..., None],
                             quantile=quantile))


def scale_ratio_loss(scales: torch.Tensor, max_ratio: float = 10.0) -> torch.Tensor:
    """Splatfacto's cap on "huge spikey gaussians" (after PhysGaussian): the mean, over the
    activated ``scales`` (N, 3), of how far each Gaussian's longest-to-shortest axis ratio
    exceeds ``max_ratio``. 0 for a Gaussian within it; a disc pays as a needle does."""
    if scales.shape[0] == 0:
        return scales.sum()
    ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-12)
    return (ratio.clamp_min(max_ratio) - max_ratio).mean()


def capsule_distance(points: torch.Tensor, capsules: torch.Tensor) -> torch.Tensor:
    """(N,): how far each of ``points`` (N, 3) lies outside the nearest of ``capsules`` (M, 7)
    -- segment ends ``a``, ``b`` and radius -- negative inside; +inf when there are none.
    The torch twin of :meth:`reifyscene.edit.body_volume.BodyVolume.distance`."""
    if capsules.shape[0] == 0:
        return torch.full(points.shape[:1], float("inf"), device=points.device)
    a, b, r = capsules[:, :3], capsules[:, 3:6], capsules[:, 6]
    ab = b - a
    rel = points[:, None, :] - a                                   # (N, M, 3)
    t = ((rel * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-18)).clamp(0.0, 1.0)
    off = rel - t[..., None] * ab
    return (((off * off).sum(-1) + 1e-12).sqrt() - r).amin(dim=1)


def owner_loss(shares: torch.Tensor, ids: torch.Tensor, known: torch.Tensor,
               quantile: float = 0.98) -> torch.Tensor | None:
    """The per-entity silhouette loss (``docs/design/2026-10-17-removal-by-ownership.md``
    §4.3): each owner's rendered share against the id map's one-hot, over labelled pixels.

    ``shares`` (H, W, C): per pixel, the composited weight of each owner channel's Gaussians;
    ``ids`` (H, W) the class of each labelled pixel (``< C``); ``known`` (H, W) bool, the
    labelled pixels. A pixel's error is the summed absolute difference over channels (0 when
    its own owner alone covers it fully, 2 when another owner does), halved to [0, 1]. As in
    Shape-of-Motion's mask term, the worst ``1 - quantile`` of pixels are left out: a mask is
    wrong somewhere in every frame. None when nothing is labelled.
    """
    if not bool(known.any()):
        return None
    pred = shares[known]
    target = F.one_hot(ids[known].long(), pred.shape[-1]).to(pred.dtype)
    err = (pred - target).abs().sum(dim=-1) / 2.0
    if quantile < 1 and err.numel() > 1:
        err = err[err <= torch.quantile(err, quantile)]
    return err.mean()

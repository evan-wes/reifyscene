"""Render-stage evaluator: co-visibility-masked novel-view synthesis.

Renders a Gaussian scene from held-out cameras and scores it against ground
truth over the co-visible pixels, using the shared masked-metric primitives.
This is the reifyscene-side producer for the
``covisibility_masked_novel_view_synthesis`` benchmark.
"""

from __future__ import annotations

from typing import Iterable

import torch

from reifyscene.eval.metrics.image import MaskedLPIPS, MaskedPSNR, MaskedSSIM
from reifyscene.render.gaussian4d.rasterizer import GaussianCloud, RenderCamera, Rasterizer


@torch.no_grad()
def score_masked_novel_views(
    rasterizer: Rasterizer,
    cloud: GaussianCloud,
    views: Iterable[tuple[RenderCamera, torch.Tensor, torch.Tensor]],
    *,
    device: str = "cuda",
    lpips_net: str = "alex",
) -> dict[str, float]:
    """Accumulate DyCheck masked mPSNR/mSSIM/mLPIPS over held-out views.

    ``views`` yields (camera, gt[H,W,3] in [0,1], covisible_mask[H,W]) — pass a
    generator to keep only one frame resident at a time. Returns the three
    masked metrics plus the view count.
    """
    mpsnr = MaskedPSNR().to(device)
    mssim = MaskedSSIM().to(device)
    mlpips = MaskedLPIPS(net_type=lpips_net).to(device)

    count = 0
    for camera, gt, mask in views:
        out = rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
        )
        pred = out.rgb.clamp(0.0, 1.0)[None]          # (1, H, W, 3)
        gt_b = gt.to(device)[None]                    # (1, H, W, 3)
        mask_b = mask.to(device).float()[None]        # (1, H, W)
        mpsnr.update(pred, gt_b, mask_b)
        mssim.update(pred, gt_b, mask_b)
        mlpips.update(pred, gt_b, mask_b)
        count += 1

    if count == 0:
        return {"masked_mpsnr": None, "masked_mssim": None, "masked_mlpips": None, "num_views": 0}

    return {
        "masked_mpsnr": float(mpsnr.compute()),
        "masked_mssim": float(mssim.compute()),
        "masked_mlpips": float(mlpips.compute()),
        "num_views": count,
    }


@torch.no_grad()
def score_masked_dynamic_views(
    rasterizer: Rasterizer,
    scene,
    views,
    *,
    device: str = "cuda",
    lpips_net: str = "alex",
) -> dict[str, float]:
    """Masked NVS over held-out (camera, gt, mask, time_id) views of a DynamicScene.

    Each view re-materializes the scene at its frame index via ``scene.cloud_at``.
    """
    mpsnr = MaskedPSNR().to(device)
    mssim = MaskedSSIM().to(device)
    mlpips = MaskedLPIPS(net_type=lpips_net).to(device)

    count = 0
    for camera, gt, mask, time_id in views:
        cloud = scene.cloud_at(int(time_id))
        out = rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
        )
        pred = out.rgb.clamp(0.0, 1.0)[None]
        gt_b = gt.to(device)[None]
        mask_b = mask.to(device).float()[None]
        mpsnr.update(pred, gt_b, mask_b)
        mssim.update(pred, gt_b, mask_b)
        mlpips.update(pred, gt_b, mask_b)
        count += 1

    if count == 0:
        return {"masked_mpsnr": None, "masked_mssim": None, "masked_mlpips": None, "num_views": 0}
    return {
        "masked_mpsnr": float(mpsnr.compute()),
        "masked_mssim": float(mssim.compute()),
        "masked_mlpips": float(mlpips.compute()),
        "num_views": count,
    }

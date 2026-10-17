# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from VGGT (https://github.com/facebookresearch/vggt). This file is licensed under the
# VGGT License, a copy of which is in LICENSES/VGGT-LICENSE.txt, and its use is subject to that
# licence's Acceptable Use Policy. It is not covered by reifyscene's own licence.
"""The two pieces of VGGT that reifyscene runs in an adapted form.

They are kept in this file, apart from the rest of the code, because the VGGT License requires
anything derived from VGGT to be shared under that licence (THIRD_PARTY_NOTICES.md).

- :func:`load_and_preprocess_images_without_upscale` is VGGT's ``load_and_preprocess_images``
  (``vggt/utils/load_fn.py``), changed never to scale a frame up: the width goes to the target
  or stays, in whole 14-pixel patches; the height follows in proportion and is centre-cropped to
  the target. VGGT-Long calls it in place of the upstream loader when its inputs are smaller
  than VGGT's 518 pixels (``VGGTLongInitializer._patched_vggt_long_preprocess``).
- :func:`forward_mixed_precision` is ``VGGT.forward`` (``vggt/models/vggt.py``, as VGGT-Long
  ships it), changed to run the prediction heads under the caller's autocast, where upstream
  turns autocast off around them, and to return the camera encoding in float32
  (``VGGTLongInitializer._patched_vggt_true_mixed_precision``).
"""

from __future__ import annotations

import numpy as np
import torch


def load_and_preprocess_images_without_upscale(
    image_path_list: list[str],
    *,
    mode: str = "crop",
    target_size: int,
) -> torch.Tensor:
    from PIL import Image

    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in {"crop", "pad"}:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    target_size = max(14, int(target_size))
    images: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()

    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        if mode == "crop":
            # Width to the target (never up), in whole 14-pixel patches, height kept in
            # proportion and cropped below. It used to scale by a float ratio (720 at 252
            # floored to 238) and clamp the height to the target, squashing portrait frames.
            new_width = max(14, (min(target_size, width) // 14) * 14)
            new_height = max(14, int(np.floor(height * new_width / width / 14.0)) * 14)
        else:
            scale = min(1.0, float(target_size) / float(max(width, height)))
            new_width = min(target_size, max(14, int(np.floor((width * scale) / 14.0)) * 14))
            new_height = min(target_size, max(14, int(np.floor((height * scale) / 14.0)) * 14))

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        if mode == "crop" and tensor.shape[1] > target_size:
            start_y = (tensor.shape[1] - target_size) // 2
            tensor = tensor[:, start_y : start_y + target_size, :]

        shapes.add((tensor.shape[1], tensor.shape[2]))
        images.append(tensor)

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images: list[torch.Tensor] = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    return torch.stack(images)


def forward_mixed_precision(model_self, images: torch.Tensor, query_points: torch.Tensor = None):
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    if query_points is not None and len(query_points.shape) == 2:
        query_points = query_points.unsqueeze(0)

    aggregated_tokens_list, patch_start_idx = model_self.aggregator(images)

    predictions = {}
    if model_self.camera_head is not None:
        pose_enc_list = model_self.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]
        if isinstance(pose_enc, torch.Tensor) and pose_enc.dtype in (torch.float16, torch.bfloat16):
            pose_enc = pose_enc.float()
        predictions["pose_enc"] = pose_enc
    if model_self.depth_head is not None:
        depth, depth_conf = model_self.depth_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        predictions["depth"] = depth
        predictions["depth_conf"] = depth_conf
    if model_self.point_head is not None:
        pts3d, pts3d_conf = model_self.point_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        predictions["world_points"] = pts3d
        predictions["world_points_conf"] = pts3d_conf
    if model_self.track_head is not None and query_points is not None:
        track_list, vis, conf = model_self.track_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            query_points=query_points,
        )
        predictions["track"] = track_list[-1]
        predictions["vis"] = vis
        predictions["conf"] = conf

    predictions["images"] = images
    return predictions

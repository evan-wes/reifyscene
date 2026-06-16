"""Load a motion-init bundle into a DynamicScene.

This is the swappable ``MotionInitializer`` seam. Today the only backend loads a
bundle dumped by ``scripts/borrow_som_motion_init.py`` (SoM's track-based init —
a stopgap). A native room4d motion initializer would implement the same
load-shape and is tracked follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from room4d.render.gaussian4d.dynamic_field import DynamicGaussianField, DynamicScene
from room4d.render.gaussian4d.gaussian_field import GaussianField
from room4d.render.gaussian4d.motion import MotionBases
from room4d.render.gaussian4d.rasterizer import RenderCamera


@dataclass
class BorrowedCamera:
    """A normalized camera from the bundle (w2c is gsplat-ready world->cam)."""

    frame_name: str
    time_id: int
    w2c: np.ndarray  # (4, 4)
    K: np.ndarray  # (3, 3)

    def render_camera(self, width: int, height: int, k_scale: float = 1.0) -> RenderCamera:
        """RenderCamera at (width, height); k_scale rescales K for a coarser factor."""
        K = self.K.astype(np.float32).copy()
        if k_scale != 1.0:
            K[:2] *= float(k_scale)
        return RenderCamera(viewmat=self.w2c.astype(np.float32), K=K, width=int(width), height=int(height))


def _cameras(data, split: str) -> list[BorrowedCamera]:
    w2cs, ks = data[f"{split}_w2cs"], data[f"{split}_Ks"]
    tids, names = data[f"{split}_time_ids"], data[f"{split}_frame_names"]
    return [
        BorrowedCamera(str(names[i]), int(tids[i]), w2cs[i], ks[i])
        for i in range(len(names))
    ]


def load_borrowed_init(npz_path: str, device: str = "cuda") -> tuple[DynamicScene, dict[str, list[BorrowedCamera]]]:
    """Load a borrowed init bundle -> (DynamicScene, {"train"/"val": cameras})."""
    data = np.load(npz_path, allow_pickle=True)

    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(data[key], dtype=torch.float32, device=device)

    fg = DynamicGaussianField(
        means=t("fg_means"),
        quats=t("fg_quats"),
        log_scales=t("fg_scales"),
        logit_opacities=t("fg_opacities").reshape(-1),
        colors_logit=t("fg_colors"),
        motion_coefs=t("fg_motion_coefs"),
    )
    bases = MotionBases(rots=t("motion_rots"), transls=t("motion_transls"))
    bg = None
    if "bg_means" in data.files:
        bg = GaussianField(
            means=t("bg_means"),
            quats=t("bg_quats"),
            log_scales=t("bg_scales"),
            logit_opacities=t("bg_opacities").reshape(-1),
            colors_logit=t("bg_colors"),
        )
    scene = DynamicScene(fg, bases, bg).to(device)
    cameras = {split: _cameras(data, split) for split in ("train", "val")}
    return scene, cameras

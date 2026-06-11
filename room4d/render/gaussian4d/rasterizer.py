"""gsplat-backed rasterization for room4d Gaussian scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None

try:
    import gsplat
except Exception:  # pragma: no cover - optional dependency guard
    gsplat = None


GSPLAT_INSTALL_HINT = (
    "The room4d renderer requires gsplat built for this GPU. "
    "Run scripts/setup_room4d.sh in the active environment (gsplat is installed "
    "by default; do not pass --no-gsplat)."
)


def gsplat_available() -> bool:
    """True when gsplat is importable and a CUDA device is present."""
    return torch is not None and gsplat is not None and torch.cuda.is_available()


def require_gsplat() -> None:
    if torch is None or gsplat is None:
        raise RuntimeError(GSPLAT_INSTALL_HINT)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "room4d rendering needs a CUDA device: gsplat rasterization kernels are GPU-only."
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def look_at_viewmat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera 4x4 in OpenCV convention (+x right, +y down, +z forward)."""
    eye = np.asarray(eye, dtype=np.float32)
    forward = _normalize(np.asarray(target, dtype=np.float32) - eye)
    right = _normalize(np.cross(forward, np.asarray(up, dtype=np.float32)))
    if not np.any(right):
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ eye
    return viewmat


def default_intrinsics(width: int, height: int) -> np.ndarray:
    # Matches the framing the camera paths were authored against:
    # fx = fy = 0.75 * max(w, h), principal point at the image center.
    fx = 0.75 * float(max(width, height))
    return np.asarray(
        [[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_to_pixels(
    points: np.ndarray, camera: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project world points for overlays; returns (xy, valid)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    viewmat = look_at_viewmat(
        np.asarray(camera["eye"], dtype=np.float32),
        np.asarray(camera["target"], dtype=np.float32),
        np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float32),
    )
    cam = pts @ viewmat[:3, :3].T + viewmat[:3, 3][None, :]
    z = cam[:, 2]
    valid = z > 1e-3
    K = default_intrinsics(width, height)
    xy = np.zeros((pts.shape[0], 2), dtype=np.float32)
    xy[valid, 0] = K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]
    xy[valid, 1] = K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]
    valid &= (
        (xy[:, 0] >= 0.0) & (xy[:, 0] < width) & (xy[:, 1] >= 0.0) & (xy[:, 1] < height)
    )
    return xy, valid


@dataclass
class GaussianCloud:
    """Device-resident Gaussian set in the renderer's z-up world frame."""

    means: Any  # (N, 3) float32
    quats: Any  # (N, 4) float32 wxyz, normalized
    scales: Any  # (N, 3) float32, world units
    opacities: Any  # (N,) float32 in [0, 1]
    colors: Any  # (N, 3) float32 in [0, 1]

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def empty(cls, device: str) -> "GaussianCloud":
        require_gsplat()
        return cls(
            means=torch.zeros((0, 3), dtype=torch.float32, device=device),
            quats=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scales=torch.zeros((0, 3), dtype=torch.float32, device=device),
            opacities=torch.zeros((0,), dtype=torch.float32, device=device),
            colors=torch.zeros((0, 3), dtype=torch.float32, device=device),
        )

    @classmethod
    def from_numpy(
        cls,
        means: np.ndarray,
        scales: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        device: str,
        quats: np.ndarray | None = None,
    ) -> "GaussianCloud":
        require_gsplat()
        means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
        n = means.shape[0]
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0
        colors = np.clip(colors.astype(np.float32).reshape(-1, 3), 0.0, 1.0)
        if quats is None:
            quats = np.zeros((n, 4), dtype=np.float32)
            quats[:, 0] = 1.0
        return cls(
            means=torch.as_tensor(means, device=device),
            quats=torch.as_tensor(np.asarray(quats, dtype=np.float32), device=device),
            scales=torch.as_tensor(np.asarray(scales, dtype=np.float32).reshape(-1, 3), device=device),
            opacities=torch.as_tensor(
                np.clip(np.asarray(opacities, dtype=np.float32).reshape(-1), 0.0, 1.0), device=device
            ),
            colors=torch.as_tensor(colors, device=device),
        )

    @classmethod
    def concat(cls, clouds: list["GaussianCloud"], device: str) -> "GaussianCloud":
        clouds = [c for c in clouds if len(c) > 0]
        if not clouds:
            return cls.empty(device)
        if len(clouds) == 1:
            return clouds[0]
        return cls(
            means=torch.cat([c.means for c in clouds], dim=0),
            quats=torch.cat([c.quats for c in clouds], dim=0),
            scales=torch.cat([c.scales for c in clouds], dim=0),
            opacities=torch.cat([c.opacities for c in clouds], dim=0),
            colors=torch.cat([c.colors for c in clouds], dim=0),
        )


class GaussianRasterizer:
    """Rasterize GaussianClouds with gsplat's CUDA EWA splatting."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def render(
        self, cloud: GaussianCloud, camera: dict[str, Any], width: int, height: int
    ) -> dict[str, np.ndarray]:
        """Render one eye/target/up camera to float32 canvases.

        Returns rgb (H, W, 3) in [0, 1], alpha (H, W) in [0, 1], and depth
        (H, W) with +inf where nothing splatted.
        """
        require_gsplat()
        if len(cloud) == 0:
            return {
                "rgb": np.zeros((height, width, 3), dtype=np.float32),
                "alpha": np.zeros((height, width), dtype=np.float32),
                "depth": np.full((height, width), np.inf, dtype=np.float32),
            }
        viewmat = torch.as_tensor(
            look_at_viewmat(
                np.asarray(camera["eye"], dtype=np.float32),
                np.asarray(camera["target"], dtype=np.float32),
                np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float32),
            ),
            device=self.device,
        )[None]
        K = torch.as_tensor(default_intrinsics(width, height), device=self.device)[None]
        with torch.no_grad():
            rendered, alphas, _ = gsplat.rasterization(
                cloud.means,
                cloud.quats,
                cloud.scales,
                cloud.opacities,
                cloud.colors,
                viewmat,
                K,
                width,
                height,
                render_mode="RGB+ED",
            )
        out = rendered[0]
        rgb = out[..., :3].clamp(0.0, 1.0).float().cpu().numpy()
        depth = out[..., 3].float().cpu().numpy()
        alpha = alphas[0, ..., 0].clamp(0.0, 1.0).float().cpu().numpy()
        depth = np.where(alpha > 1e-4, depth, np.inf).astype(np.float32)
        return {"rgb": rgb, "alpha": alpha, "depth": depth}

"""gsplat-backed rasterization for room4d Gaussian scenes."""

from __future__ import annotations

import abc
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


def _intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    """Build a 3x3 pinhole K from a CameraIntrinsics-like object or a 3x3 array."""
    if hasattr(intrinsics, "fx"):
        return np.asarray(
            [
                [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(
        f"intrinsics must be a CameraIntrinsics or a 3x3 matrix, got shape {arr.shape}"
    )


@dataclass
class RenderCamera:
    """Resolved pinhole camera for rasterization: world-to-camera pose + K.

    The backend-agnostic camera contract a Rasterizer renders from. Build it via
    ``from_look_at`` (procedural path cameras, optionally synthesizing K) or
    ``from_world_cam`` (a real dataset camera with known extrinsics/intrinsics).
    """

    viewmat: np.ndarray  # (4, 4) world-to-camera, OpenCV convention (+x right, +y down, +z fwd)
    K: np.ndarray  # (3, 3) pinhole intrinsics in pixels
    width: int
    height: int

    @classmethod
    def from_look_at(
        cls,
        eye: Any,
        target: Any,
        up: Any,
        width: int,
        height: int,
        intrinsics: Any | None = None,
    ) -> "RenderCamera":
        """Path-camera constructor; synthesizes intrinsics when none supplied."""
        viewmat = look_at_viewmat(
            np.asarray(eye, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
            np.asarray(up, dtype=np.float32),
        )
        K = default_intrinsics(width, height) if intrinsics is None else _intrinsics_to_matrix(intrinsics)
        return cls(viewmat=viewmat, K=K, width=int(width), height=int(height))

    @classmethod
    def from_world_cam(
        cls,
        T_world_cam: Any,
        intrinsics: Any,
        width: int,
        height: int,
    ) -> "RenderCamera":
        """Real-camera constructor: viewmat = inv(T_world_cam), K from intrinsics."""
        Twc = np.asarray(T_world_cam, dtype=np.float32).reshape(4, 4)
        viewmat = np.linalg.inv(Twc).astype(np.float32)
        return cls(
            viewmat=viewmat,
            K=_intrinsics_to_matrix(intrinsics),
            width=int(width),
            height=int(height),
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


class Rasterizer(abc.ABC):
    """Pluggable contract for rasterizing GaussianClouds to image canvases.

    Backends are selected via ``build_rasterizer``; gsplat is the first one.
    The trainer and render-proof depend on this interface, not on gsplat
    directly, so the splatting kernel stays swappable. PR 2 extends the contract
    with a differentiable training path + densification telemetry.
    """

    backend_name = "base"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @abc.abstractmethod
    def render(self, cloud: GaussianCloud, camera: RenderCamera) -> dict[str, np.ndarray]:
        """Render a cloud from a resolved camera to float32 canvases.

        Returns rgb (H, W, 3) in [0, 1], alpha (H, W) in [0, 1], and depth
        (H, W) with +inf where nothing splatted.
        """
        raise NotImplementedError


class GsplatRasterizer(Rasterizer):
    """Rasterize GaussianClouds with gsplat's CUDA EWA splatting (first backend)."""

    backend_name = "gsplat"

    def render(self, cloud: GaussianCloud, camera: RenderCamera) -> dict[str, np.ndarray]:
        require_gsplat()
        width, height = int(camera.width), int(camera.height)
        if len(cloud) == 0:
            return {
                "rgb": np.zeros((height, width, 3), dtype=np.float32),
                "alpha": np.zeros((height, width), dtype=np.float32),
                "depth": np.full((height, width), np.inf, dtype=np.float32),
            }
        viewmat = torch.as_tensor(camera.viewmat, dtype=torch.float32, device=self.device)[None]
        K = torch.as_tensor(camera.K, dtype=torch.float32, device=self.device)[None]
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


_RASTERIZERS: dict[str, type[Rasterizer]] = {
    "gsplat": GsplatRasterizer,
}


def build_rasterizer(cfg: dict | None = None) -> Rasterizer:
    """Construct the configured rasterizer backend (default: gsplat)."""
    cfg = cfg or {}
    name = str(cfg.get("rasterizer_backend", "gsplat")).lower()
    device = str(cfg.get("device", "cuda"))
    backend_cls = _RASTERIZERS.get(name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown rasterizer backend '{name}'. Supported backends: {sorted(_RASTERIZERS)}"
        )
    return backend_cls(device=device)

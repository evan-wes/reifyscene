"""gsplat-backed rasterization for reifyscene Gaussian scenes."""

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
    "The reifyscene renderer requires gsplat built for this GPU. "
    "Run scripts/setup_reifyscene.sh in the active environment (gsplat is installed "
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
            "reifyscene rendering needs a CUDA device: gsplat rasterization kernels are GPU-only."
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
    # (N, D) float32, or None. Composited exactly like colour -- this is how per-Gaussian
    # identity reaches the image. Unbounded: whatever the caller stores is what blends.
    features: Any = None

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
        features: np.ndarray | None = None,
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
            features=(
                None
                if features is None
                else torch.as_tensor(
                    np.asarray(features, dtype=np.float32).reshape(n, -1), device=device
                )
            ),
        )

    @classmethod
    def concat(cls, clouds: list["GaussianCloud"], device: str) -> "GaussianCloud":
        clouds = [c for c in clouds if len(c) > 0]
        if not clouds:
            return cls.empty(device)
        if len(clouds) == 1:
            return clouds[0]
        widths = {None if c.features is None else int(c.features.shape[-1]) for c in clouds}
        if len(widths) > 1:
            # Filling the gap with zeros would assert "no identity" for Gaussians that were
            # simply never given one -- the silent-wrong case, so refuse instead.
            raise ValueError(
                f"cannot concat clouds with differing feature widths {sorted(widths, key=str)}"
            )
        return cls(
            means=torch.cat([c.means for c in clouds], dim=0),
            quats=torch.cat([c.quats for c in clouds], dim=0),
            scales=torch.cat([c.scales for c in clouds], dim=0),
            opacities=torch.cat([c.opacities for c in clouds], dim=0),
            colors=torch.cat([c.colors for c in clouds], dim=0),
            features=(
                None
                if clouds[0].features is None
                else torch.cat([c.features for c in clouds], dim=0)
            ),
        )


@dataclass
class RasterOutput:
    """Differentiable rasterization result + densification telemetry.

    ``rgb``/``alpha``/``depth`` are on-device torch tensors that carry grad when
    rasterized with ``with_telemetry=True``. ``means2d`` is the screen-space
    projected means tensor whose ``.grad`` (populated after ``backward()`` when
    ``retain_grad`` was set) drives adaptive density control; ``radii`` is the
    per-Gaussian screen radius. These two are the telemetry any rasterizer
    backend must surface so the trainer's densification never touches gsplat
    internals. Shapes follow the unpacked single-camera convention: ``means2d``
    is ``(1, N, 2)`` (gradient-bearing), ``radii`` is ``(1, N)`` or ``(1, N, 2)``.
    """

    rgb: Any  # (H, W, 3)
    alpha: Any  # (H, W)
    depth: Any  # (H, W)
    means2d: Any = None  # (1, N, 2) screen-space means in [-1, 1] NDC (grad source)
    radii: Any = None  # (1, N) or (1, N, 2) screen radii
    width: int = 0  # render width (needed to rescale the NDC means2d gradient)
    height: int = 0
    features: Any = None  # (H, W, D) when features were rasterized, else None


class Rasterizer(abc.ABC):
    """Pluggable contract for rasterizing Gaussians to image canvases.

    Backends are selected via ``build_rasterizer``; gsplat is the first one.
    The trainer and render-proof depend on this interface, not on gsplat
    directly, so the splatting kernel stays swappable. ``rasterize`` is the
    differentiable core (training + telemetry); ``render`` is a no-grad numpy
    convenience built on top of it (flythrough / render-proof preview).
    """

    backend_name = "base"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @abc.abstractmethod
    def rasterize(
        self,
        means: Any,
        quats: Any,
        scales: Any,
        opacities: Any,
        colors: Any,
        camera: RenderCamera,
        *,
        with_telemetry: bool = False,
        features: Any = None,
    ) -> RasterOutput:
        """Differentiably rasterize activated Gaussian params from a camera.

        ``means`` (N,3), ``quats`` (N,4), ``scales`` (N,3, world units),
        ``opacities`` (N,) in [0,1], ``colors`` (N,3) in [0,1]. When
        ``with_telemetry`` is True the result carries grad-ready ``means2d``
        (with ``retain_grad``) and ``radii`` for densification.

        ``features`` (N,D), when given, is alpha-composited in the same pass as
        ``colors`` and returned as ``RasterOutput.features`` (H,W,D). Without it
        the output is exactly what it was before features existed.
        """
        raise NotImplementedError

    def render(self, cloud: GaussianCloud, camera: RenderCamera) -> dict[str, np.ndarray]:
        """No-grad render of a GaussianCloud to float32 canvases.

        Returns rgb (H, W, 3) in [0, 1], alpha (H, W) in [0, 1], and depth
        (H, W) with +inf where nothing splatted.
        """
        width, height = int(camera.width), int(camera.height)
        if len(cloud) == 0:
            empty = {
                "rgb": np.zeros((height, width, 3), dtype=np.float32),
                "alpha": np.zeros((height, width), dtype=np.float32),
                "depth": np.full((height, width), np.inf, dtype=np.float32),
            }
            if cloud.features is not None:
                d = int(cloud.features.shape[-1])
                empty["features"] = np.zeros((height, width, d), dtype=np.float32)
            return empty
        with torch.no_grad():
            out = self.rasterize(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera,
                features=cloud.features,
            )
        rgb = out.rgb.clamp(0.0, 1.0).float().cpu().numpy()
        depth = out.depth.float().cpu().numpy()
        alpha = out.alpha.clamp(0.0, 1.0).float().cpu().numpy()
        depth = np.where(alpha > 1e-4, depth, np.inf).astype(np.float32)
        result = {"rgb": rgb, "alpha": alpha, "depth": depth}
        if out.features is not None:
            result["features"] = out.features.float().cpu().numpy()
        return result


class GsplatRasterizer(Rasterizer):
    """Rasterize Gaussians with gsplat's CUDA EWA splatting (first backend)."""

    backend_name = "gsplat"

    def __init__(self, device: str = "cuda", absgrad: bool = False) -> None:
        super().__init__(device)
        # absgrad=False matches the ported gsplat DefaultStrategy thresholds.
        self.absgrad = bool(absgrad)

    def rasterize(
        self,
        means: Any,
        quats: Any,
        scales: Any,
        opacities: Any,
        colors: Any,
        camera: RenderCamera,
        *,
        with_telemetry: bool = False,
        features: Any = None,
    ) -> RasterOutput:
        require_gsplat()
        # Features ride in the colour channels of the same call: one sort, one composite,
        # so they blend with exactly the weights RGB does. gsplat appends ED depth last.
        n_feat = 0
        if features is not None:
            if features.ndim != 2 or features.shape[0] != colors.shape[0]:
                raise ValueError(
                    f"features must be (N, D) with N={colors.shape[0]}, got {tuple(features.shape)}"
                )
            n_feat = int(features.shape[1])
            colors = torch.cat([colors, features.to(colors.dtype)], dim=-1)
        viewmats = torch.as_tensor(camera.viewmat, dtype=torch.float32, device=self.device).reshape(1, 4, 4)
        Ks = torch.as_tensor(camera.K, dtype=torch.float32, device=self.device).reshape(1, 3, 3)
        width, height = int(camera.width), int(camera.height)
        render_colors, render_alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            render_mode="RGB+ED",
            packed=False,
            absgrad=self.absgrad,
        )
        means2d = info.get("means2d")
        if with_telemetry and means2d is not None and getattr(means2d, "requires_grad", False):
            means2d.retain_grad()
        img = render_colors[0]
        return RasterOutput(
            rgb=img[..., :3],
            alpha=render_alphas[0, ..., 0],
            depth=img[..., 3 + n_feat],
            means2d=means2d,
            radii=info.get("radii"),
            width=width,
            height=height,
            features=img[..., 3:3 + n_feat] if n_feat else None,
        )


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
    rasterizer = backend_cls(device=device)
    if "absgrad" in cfg and hasattr(rasterizer, "absgrad"):
        rasterizer.absgrad = bool(cfg["absgrad"])
    return rasterizer

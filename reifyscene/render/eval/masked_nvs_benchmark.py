"""In-pipeline masked novel-view-synthesis benchmark (DyCheck protocol).

Scores a trained render checkpoint against a benchmark dataset's held-out val
cameras + covisibility masks -- the real measuring instrument, replacing the
weak train-view PSNR. It is **dataset-coupled**: held-out GT views + covisibility
masks only exist for benchmark datasets (DyCheck / NVIDIA), so this runs only
when an eval dataset source is configured, never on arbitrary input video.

Reuses the exact scoring path the standalone proof harness uses
(``score_masked_novel_views`` + ``DatasetGeometryInitializer``), so an
in-pipeline number is directly comparable to ``run_render_proof.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.render.eval.novel_view import score_masked_dynamic_views, score_masked_novel_views
from reifyscene.render.gaussian4d.rasterizer import GaussianCloud, RenderCamera, build_rasterizer


def _load_trained_cloud(npz_path: Path, device: str) -> GaussianCloud:
    data = np.load(npz_path)

    def _t(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(data[key], dtype=np.float32), device=device)

    return GaussianCloud(
        means=_t("means"),
        quats=_t("quats"),
        scales=_t("scales"),
        opacities=_t("opacities"),
        colors=_t("colors"),
    )


def score_render_artifact_masked_nvs(
    render_artifact,
    eval_cfg: dict[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Score a trained RenderArtifact's masked NVS against a dataset's val split.

    ``eval_cfg`` keys: ``dataset_root`` (required to run), ``factor``, ``split``
    (the train split the renderer was fit on), ``eval_split`` (held-out, default
    ``val``), ``max_eval_frames``, ``lpips_net``. Returns the masked
    mPSNR/mSSIM/mLPIPS + ``num_views`` (or ``{}`` when no dataset source / no
    checkpoint is available).
    """
    dataset_root = eval_cfg.get("dataset_root")
    if not dataset_root:
        return {}
    checkpoint = (getattr(render_artifact, "scene_state", {}) or {}).get("checkpoint")
    if not checkpoint:
        return {}
    npz_path = Path(render_artifact.save_dir) / checkpoint
    if not npz_path.exists():
        return {}

    eval_split = str(eval_cfg.get("eval_split", "val"))
    backend = DatasetGeometryInitializer({
        "dataset_root": str(dataset_root),
        "factor": int(eval_cfg.get("factor", 2)),
        "split": str(eval_cfg.get("split", "train")),
    })
    cloud = _load_trained_cloud(npz_path, device)
    rasterizer = build_rasterizer({"device": device})

    val_specs = backend.split_frames(eval_split)
    max_eval = eval_cfg.get("max_eval_frames")
    if max_eval:
        val_specs = val_specs[: int(max_eval)]

    def _views():
        for cam, t in val_specs:
            name = backend.frame_name(cam, int(t))
            intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
            camera = RenderCamera.from_world_cam(t_world_cam, intrinsics, width, height)
            gt = torch.as_tensor(backend.load_rgb(name), dtype=torch.float32, device=device) / 255.0
            mask = torch.as_tensor(backend.load_covisible(name, eval_split), device=device)
            yield camera, gt, mask

    metrics = score_masked_novel_views(
        rasterizer, cloud, _views(), device=device, lpips_net=str(eval_cfg.get("lpips_net", "alex"))
    )
    metrics["eval_split"] = eval_split
    metrics["dataset"] = Path(str(dataset_root)).name
    return metrics


def _normalize_w2c(w2c: np.ndarray, transfm: np.ndarray, scale: float) -> np.ndarray:
    """Place a raw world->cam in the scene-normalized frame (SoM convention)."""
    out = (np.asarray(w2c, np.float32) @ np.linalg.inv(np.asarray(transfm, np.float32))).astype(np.float32)
    out[:3, 3] /= float(scale)
    return out


def eval_dataset(eval_cfg: dict[str, Any]) -> DatasetGeometryInitializer:
    """The benchmark dataset reader the held-out views are loaded from."""
    return DatasetGeometryInitializer({
        "dataset_root": str(eval_cfg["dataset_root"]),
        "factor": int(eval_cfg.get("factor", 2)),
        "split": str(eval_cfg.get("split", "train")),
    })


@dataclass(frozen=True)
class HeldoutPlacement:
    """One held-out view placed in a trained scene's frame, before any image is loaded.

    ``camera`` maps the ground-truth image's (width, height) to the RenderCamera:
    bundle-carried cameras are sized from the image, dataset-placed ones carry
    their own size and ignore it.
    """

    frame_name: str
    time_id: int
    camera: Callable[[int, int], RenderCamera]


def heldout_placements(
    data, cams, eval_cfg: dict[str, Any], backend
) -> list[HeldoutPlacement] | None:
    """Place a dynamic bundle's held-out cameras, or ``None`` when they cannot be placed.

    Held-out cameras come from the bundle when present (the gate path: normalized
    cams carried from a DyCheck-sourced build, rendered at
    ``k_scale = bundle_factor / eval_factor``); otherwise they are built from the
    dataset and placed in the scene frame via the bundle's scene-norm (the
    fully-in-pipeline path, where the provider has no val split).

    Scoring and view export both place cameras here, so an exported view is
    exactly a view that was scored.
    """
    eval_split = str(eval_cfg.get("eval_split", "val"))
    eval_factor = int(eval_cfg.get("factor", 2))
    max_eval = eval_cfg.get("max_eval_frames")

    bundle_val = cams.get(eval_split) or cams.get("val") or []
    if bundle_val:
        # The bundle is built full-res relative to the eval factor, so K scales by the ratio.
        has_factor = "factor" in getattr(data, "files", data)
        bundle_factor = int(np.asarray(data["factor"])) if has_factor else 1
        k_scale = float(bundle_factor) / float(eval_factor)
        val_cams = bundle_val[: int(max_eval)] if max_eval else bundle_val
        return [
            HeldoutPlacement(
                cam.frame_name,
                int(cam.time_id),
                lambda width, height, cam=cam: cam.render_camera(width, height, k_scale=k_scale),
            )
            for cam in val_cams
        ]

    if "scene_norm_transfm" not in getattr(data, "files", data):
        return None
    transfm = np.asarray(data["scene_norm_transfm"], np.float32)
    scale = float(np.asarray(data["scene_norm_scale"]))
    specs = backend.split_frames(eval_split)
    if max_eval:
        specs = specs[: int(max_eval)]
    placements = []
    for cam, time_id in specs:
        name = backend.frame_name(cam, int(time_id))
        intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
        w2c = _normalize_w2c(np.linalg.inv(np.asarray(t_world_cam, np.float32)), transfm, scale)
        K = np.array(
            [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
            np.float32,
        )
        camera = RenderCamera(viewmat=w2c, K=K, width=int(width), height=int(height))
        placements.append(
            HeldoutPlacement(name, int(time_id), lambda _w, _h, camera=camera: camera)
        )
    return placements


def load_heldout_view(
    placement: HeldoutPlacement, backend, eval_split: str, device: str
) -> tuple[RenderCamera, torch.Tensor, torch.Tensor]:
    """(camera, gt[H,W,3] in [0,1], covisible mask[H,W]) for one placed held-out view."""
    gt_u8 = backend.load_rgb(placement.frame_name)
    height, width = gt_u8.shape[:2]
    gt = torch.as_tensor(np.asarray(gt_u8, dtype=np.float32), device=device) / 255.0
    covisible = backend.load_covisible(placement.frame_name, eval_split)
    mask = torch.as_tensor(np.asarray(covisible), device=device)
    return placement.camera(width, height), gt, mask


def score_render_artifact_masked_nvs_dynamic(
    render_artifact,
    eval_cfg: dict[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Masked-NVS for a trained DYNAMIC render artifact (DyCheck protocol).

    Mirrors :func:`score_render_artifact_masked_nvs` for a moving scene: reloads
    the trained ``DynamicScene`` from the ``trained_dynamic.npz`` bundle, scores
    held-out val views with ``score_masked_dynamic_views`` (re-materializing the
    scene at each view's ``time_id``). Held-out cameras are placed by
    :func:`heldout_placements`.
    """
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    dataset_root = eval_cfg.get("dataset_root")
    if not dataset_root:
        return {}
    checkpoint = (getattr(render_artifact, "scene_state", {}) or {}).get("checkpoint")
    if not checkpoint:
        return {}
    npz_path = Path(render_artifact.save_dir) / checkpoint
    if not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=True)
    scene, cams = bundle_to_scene(data, device=device)
    backend = eval_dataset(eval_cfg)
    placements = heldout_placements(data, cams, eval_cfg, backend)
    if placements is None:
        return {}  # cannot place dataset cameras in the scene frame without a scene-norm
    rasterizer = build_rasterizer({"device": device})
    eval_split = str(eval_cfg.get("eval_split", "val"))
    lpips_net = str(eval_cfg.get("lpips_net", "alex"))

    views = ((*load_heldout_view(p, backend, eval_split, device), p.time_id) for p in placements)
    metrics = score_masked_dynamic_views(
        rasterizer, scene, views, device=device, lpips_net=lpips_net
    )
    metrics["eval_split"] = eval_split
    metrics["dataset"] = Path(str(dataset_root)).name
    metrics["kind"] = "dynamic"
    return metrics

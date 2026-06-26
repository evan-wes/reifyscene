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

from pathlib import Path
from typing import Any

import numpy as np
import torch

from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.render.eval.novel_view import score_masked_novel_views
from room4d.render.gaussian4d.rasterizer import GaussianCloud, RenderCamera, build_rasterizer


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

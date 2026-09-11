"""In-pipeline masked novel-view-synthesis benchmark (DyCheck protocol).

Scores a trained render checkpoint against a benchmark dataset's held-out val
cameras + covisibility masks -- the real measuring instrument, replacing the
weak train-view PSNR. It is **dataset-coupled**: held-out GT views + covisibility
masks only exist for benchmark datasets (DyCheck / NVIDIA), so this runs only
when an eval dataset source is configured, never on arbitrary input video.

Reuses the exact scoring path the standalone proof harness uses
(``score_masked_novel_views`` + ``DatasetGeometryInitializer``), so an
in-pipeline number is directly comparable to ``run_render_proof.py``.

A dynamic scene built on self-reconstructed geometry (VGGT / VGGT-Long) lives in
its own frame, not the dataset's, so its held-out cameras are placed through a
similarity fitted on the training cameras (``align: sim3``); every score records
how its cameras were placed. See :func:`place_heldout_views`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.geometry.similarity import (
    apply_similarity_to_c2w,
    fit_similarity,
    similarity_residuals,
)
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


# How held-out cameras reach a dynamic scene's frame; recorded as ``alignment["method"]``
# with every score:
#   bundle_val  the bundle carries its own held-out cameras (DyCheck-sourced builds)
#   none        dataset cameras through the bundle's scene-norm: right only when the scene
#               was built on the dataset's own geometry (the GT oracle)
#   sim3        a similarity fitted from the dataset's training cameras onto the bundle's,
#               for self-reconstructed geometry, whose frame differs from the dataset's by an
#               unknown rotation, translation and scale (without it the renders miss the scene)
ALIGN_MAX_ROTATION_DEG = 5.0
ALIGN_MAX_CENTRE_FRACTION = 0.2


class AlignmentFailed(ValueError):
    """The training cameras do not support a trustworthy alignment, so no score is given."""

    def __init__(self, reason: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.record = record or {}


def default_alignment(geometry_backend: str | None) -> str:
    """``none`` for a scene built on the benchmark's own geometry, ``sim3`` otherwise."""
    return "none" if str(geometry_backend or "").lower() == "dataset" else "sim3"


@dataclass(frozen=True)
class HeldoutPlacement:
    """One held-out view placed in a trained scene's frame, before any image is loaded.

    ``camera`` maps the ground-truth image's (width, height) to the RenderCamera:
    bundle-carried cameras are sized from the image, dataset-placed ones carry
    their own size and ignore it. ``time_id`` is the dataset's frame; ``scene_time``
    is the scene time the view is rendered at, the nearest training keyframe's, which
    differs from ``time_id`` when the scene was trained on sparse keyframes.
    """

    frame_name: str
    time_id: int
    camera: Callable[[int, int], RenderCamera]
    scene_time: int


@dataclass(frozen=True)
class HeldoutViews:
    """The placed held-out views and how they were placed (recorded with the score)."""

    placements: list[HeldoutPlacement]
    alignment: dict[str, Any]


def _keyframe_frames(train_cams) -> dict[int, tuple[str, int]]:
    """Scene time -> (stream, video frame) of each training camera.

    Bundle frame names are ``<stream>:<frame>``. Sparse-keyframe scenes number their
    time by keyframe, so the frame cannot be read off the time id.
    """
    frames = {}
    for cam in train_cams:
        stream, _, frame = str(cam.frame_name).rpartition(":")
        try:
            frames[int(cam.time_id)] = (stream or "0", int(frame))
        except ValueError:
            frames[int(cam.time_id)] = ("0", int(cam.time_id))
    return frames


def _nearest_scene_time(frame: int, keyframes: dict[int, tuple[str, int]]) -> int:
    if not keyframes:
        return int(frame)
    return min(keyframes, key=lambda t: (abs(keyframes[t][1] - frame), t))


def _fit_training_alignment(
    keyframes: dict[int, tuple[str, int]], train_cams, backend, eval_cfg: dict[str, Any]
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit dataset world -> scene frame on the training cameras; refuse an untrustworthy fit."""
    src, dst = {}, {}
    for cam in train_cams:
        stream, frame = keyframes[int(cam.time_id)]
        _, t_world_cam, _ = backend.load_camera(backend.frame_name(stream, frame))
        src[str(cam.time_id)] = np.linalg.inv(np.asarray(t_world_cam, np.float64))
        dst[str(cam.time_id)] = np.asarray(cam.w2c, np.float64)
    keys = sorted(src, key=int)
    if len(keys) < 3:
        raise AlignmentFailed(f"need at least 3 training cameras to align, have {len(keys)}")
    s, R, t = fit_similarity(src, dst, keys)
    record = {
        "method": "sim3",
        "fit_on": "training cameras",
        "num_train_cameras": len(keys),
        "scale": s,
        **similarity_residuals(src, dst, keys, s, R, t),
    }
    max_rot = float(eval_cfg.get("align_max_rotation_deg", ALIGN_MAX_ROTATION_DEG))
    max_ctr = float(eval_cfg.get("align_max_centre_fraction", ALIGN_MAX_CENTRE_FRACTION))
    problems = []
    if not s > 0:
        problems.append(f"scale {s:.4g} is not positive (camera correspondences disagree)")
    if record["rotation_residual_deg_median"] > max_rot:
        problems.append(
            f"median rotation residual {record['rotation_residual_deg_median']:.2f} deg > {max_rot}"
        )
    if record["centre_residual_median"] > max_ctr:
        problems.append(
            f"median centre residual {record['centre_residual_median']:.3f} of the trajectory "
            f"> {max_ctr}"
        )
    if problems:
        raise AlignmentFailed("; ".join(problems), record)
    return s, R, t, record


def place_heldout_views(data, cams, eval_cfg: dict[str, Any], backend) -> HeldoutViews | None:
    """Place a dynamic bundle's held-out cameras in its scene frame.

    Held-out cameras come from the bundle when it carries them (``bundle_val``: the
    gate path, rendered at ``k_scale = bundle_factor / eval_factor``); otherwise
    they are the dataset's, placed according to ``eval_cfg["align"]`` (default
    ``none``; see :func:`default_alignment`). Each view renders at the nearest
    training keyframe's scene time.

    Returns ``None`` when they cannot be placed at all (``none`` without a
    scene-norm); raises :class:`AlignmentFailed` when ``sim3`` is asked for but the
    training cameras do not support a trustworthy fit. Scoring and view export both
    place cameras here, so an exported view is exactly a view that was scored.
    """
    eval_split = str(eval_cfg.get("eval_split", "val"))
    eval_factor = int(eval_cfg.get("factor", 2))
    max_eval = eval_cfg.get("max_eval_frames")
    train_cams = cams.get("train") or []
    keyframes = _keyframe_frames(train_cams)

    bundle_val = cams.get(eval_split) or cams.get("val") or []
    if bundle_val:
        # The bundle is built full-res relative to the eval factor, so K scales by the ratio.
        has_factor = "factor" in getattr(data, "files", data)
        bundle_factor = int(np.asarray(data["factor"])) if has_factor else 1
        k_scale = float(bundle_factor) / float(eval_factor)
        val_cams = bundle_val[: int(max_eval)] if max_eval else bundle_val
        placements = [
            HeldoutPlacement(
                cam.frame_name,
                int(cam.time_id),
                lambda width, height, cam=cam: cam.render_camera(width, height, k_scale=k_scale),
                int(cam.time_id),
            )
            for cam in val_cams
        ]
        return HeldoutViews(placements, {"method": "bundle_val"})

    align = str(eval_cfg.get("align", "none")).lower()
    if align == "sim3":
        s, R, t, record = _fit_training_alignment(keyframes, train_cams, backend, eval_cfg)

        def to_scene_w2c(t_world_cam: np.ndarray) -> np.ndarray:
            c2w = apply_similarity_to_c2w(np.asarray(t_world_cam, np.float64), s, R, t)
            return np.linalg.inv(c2w).astype(np.float32)

    elif align == "none":
        if "scene_norm_transfm" not in getattr(data, "files", data):
            return None
        transfm = np.asarray(data["scene_norm_transfm"], np.float32)
        scale = float(np.asarray(data["scene_norm_scale"]))
        record = {"method": "none"}

        def to_scene_w2c(t_world_cam: np.ndarray) -> np.ndarray:
            w2c = np.linalg.inv(np.asarray(t_world_cam, np.float32))
            return _normalize_w2c(w2c, transfm, scale)

    else:
        raise ValueError(f"unknown alignment {align!r}: expected 'none' or 'sim3'")
    record["scene_time"] = "nearest training keyframe"

    specs = backend.split_frames(eval_split)
    if max_eval:
        specs = specs[: int(max_eval)]
    placements = []
    for cam, time_id in specs:
        name = backend.frame_name(cam, int(time_id))
        intrinsics, t_world_cam, (width, height) = backend.load_camera(name)
        K = np.array(
            [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
            np.float32,
        )
        camera = RenderCamera(
            viewmat=to_scene_w2c(t_world_cam), K=K, width=int(width), height=int(height)
        )
        placements.append(
            HeldoutPlacement(
                name,
                int(time_id),
                lambda _w, _h, camera=camera: camera,
                _nearest_scene_time(int(time_id), keyframes),
            )
        )
    return HeldoutViews(placements, record)


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


def score_dynamic_bundle(
    data,
    scene,
    cams,
    eval_cfg: dict[str, Any],
    *,
    backend,
    device: str = "cuda",
    rasterizer=None,
) -> dict[str, Any]:
    """Masked NVS of a loaded dynamic bundle against the benchmark's held-out views.

    Returns ``{}`` when the held-out cameras cannot be placed. When an alignment is
    refused the result carries the refused ``alignment`` record and no score: a
    number from misplaced cameras would measure the placement, not the scene.
    """
    eval_split = str(eval_cfg.get("eval_split", "val"))
    context = {
        "eval_split": eval_split,
        "dataset": Path(str(eval_cfg["dataset_root"])).name,
        "kind": "dynamic",
    }
    try:
        views = place_heldout_views(data, cams, eval_cfg, backend)
    except AlignmentFailed as exc:
        return {**context, "alignment": {**exc.record, "status": "refused", "reason": str(exc)}}
    if views is None:
        return {}  # cannot place dataset cameras in the scene frame without a scene-norm
    rasterizer = rasterizer or build_rasterizer({"device": device})
    stream = (
        (*load_heldout_view(p, backend, eval_split, device), p.scene_time) for p in views.placements
    )
    metrics = score_masked_dynamic_views(
        rasterizer, scene, stream, device=device, lpips_net=str(eval_cfg.get("lpips_net", "alex"))
    )
    return {**metrics, **context, "alignment": views.alignment}


def score_render_artifact_masked_nvs_dynamic(
    render_artifact,
    eval_cfg: dict[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Masked-NVS for a trained DYNAMIC render artifact (DyCheck protocol).

    Mirrors :func:`score_render_artifact_masked_nvs` for a moving scene: reloads
    the trained ``DynamicScene`` from the ``trained_dynamic.npz`` bundle and scores
    it with :func:`score_dynamic_bundle`. Held-out cameras are placed by
    :func:`place_heldout_views`, per ``eval_cfg["align"]``.
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
    return score_dynamic_bundle(
        data, scene, cams, eval_cfg, backend=eval_dataset(eval_cfg), device=device
    )

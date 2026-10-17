"""Export what a trained scene looks like: held-out views and a replay.

The masked-NVS score says how good held-out renders are; this writes some of them
out so a person can see it. Nothing is retrained -- it renders straight from a
run's ``render/trained_dynamic.npz`` (or, for held-out views of a static scene,
``render/trained_gaussians.npz``).

- **Held-out views.** For evenly spaced views of the benchmark's held-out split:
  the render, the real photo, the covisibility mask, and that view's masked PSNR.
  Cameras are placed by the scorer's own :func:`place_heldout_views` (or
  :func:`place_static_heldout_views`), so an exported view is exactly a view that
  was scored.
- **Replay.** A dynamic scene re-rendered from every training camera at its own
  time: the path the camera actually took. It shows reconstruction fidelity, not
  novel views, and should be labelled that way.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from reifyscene.eval.metrics.image import MaskedPSNR
from reifyscene.render.eval.masked_nvs_benchmark import (
    HeldoutViews,
    load_heldout_view,
    place_heldout_views,
    place_static_heldout_views,
)
from reifyscene.render.export import write_video


def pick_evenly(n: int, count: int) -> list[int]:
    """``count`` indices spread evenly over ``range(n)``, first and last included."""
    if n <= 0 or count <= 0:
        return []
    if count >= n:
        return list(range(n))
    if count == 1:
        return [n // 2]
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _to_u8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _render_cloud(rasterizer, cloud, camera) -> torch.Tensor:
    out = rasterizer.rasterize(
        cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
    )
    return out.rgb.clamp(0.0, 1.0)


def _render(rasterizer, scene, camera, time_id: int) -> torch.Tensor:
    return _render_cloud(rasterizer, scene.cloud_at(int(time_id)), camera)


@torch.no_grad()
def export_heldout_views(
    scene,
    data,
    cams: dict[str, list],
    eval_cfg: dict[str, Any],
    out_dir: str | Path,
    *,
    backend,
    rasterizer,
    count: int = 8,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Write render / photo / mask PNGs for ``count`` evenly spaced held-out views.

    Also writes ``index.json`` with each view's frame, time and masked PSNR (the
    same per-frame formula the benchmark averages) and how the cameras were placed
    (``eval_cfg["align"]``). Returns those records. Raises ``AlignmentFailed`` when
    an alignment is asked for but refused.
    """
    views = place_heldout_views(data, cams, eval_cfg, backend)
    if views is None:
        raise ValueError(
            "cannot place held-out cameras: the bundle has no val cameras and no scene-norm"
        )
    return _write_heldout_views(
        views,
        lambda camera, time_id: _render(rasterizer, scene, camera, time_id),
        eval_cfg,
        out_dir,
        backend=backend,
        count=count,
        device=device,
    )


@torch.no_grad()
def export_static_heldout_views(
    cloud,
    train_views: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
    out_dir: str | Path,
    *,
    backend,
    rasterizer,
    count: int = 8,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """:func:`export_heldout_views` for a trained static cloud.

    Cameras are placed through the scene's ``train_views``, as the static scorer
    places them; every view renders the one cloud.
    """
    views = place_static_heldout_views(train_views, eval_cfg, backend)
    return _write_heldout_views(
        views,
        lambda camera, _time_id: _render_cloud(rasterizer, cloud, camera),
        eval_cfg,
        out_dir,
        backend=backend,
        count=count,
        device=device,
    )


def _write_heldout_views(
    views: HeldoutViews,
    render_at: Callable[[Any, int], torch.Tensor],
    eval_cfg: dict[str, Any],
    out_dir: str | Path,
    *,
    backend,
    count: int,
    device: str,
) -> list[dict[str, Any]]:
    """Render, compare and write ``count`` evenly spaced placed views, plus ``index.json``."""
    placements = views.placements
    eval_split = str(eval_cfg.get("eval_split", "val"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in pick_evenly(len(placements), count):
        placement = placements[index]
        camera, gt, mask = load_heldout_view(placement, backend, eval_split, device)
        pred = render_at(camera, placement.scene_time)
        psnr = MaskedPSNR().to(device)
        psnr.update(pred[None], gt.to(device)[None], mask.to(device).float()[None])

        stem = f"{index:04d}_{placement.frame_name.replace(':', '_')}"
        files = {
            "render": f"{stem}_render.png",
            "photo": f"{stem}_photo.png",
            "mask": f"{stem}_mask.png",
        }
        Image.fromarray(_to_u8(pred)).save(out_dir / files["render"])
        Image.fromarray(_to_u8(gt)).save(out_dir / files["photo"])
        Image.fromarray(mask.detach().cpu().numpy().astype(np.uint8) * 255).save(
            out_dir / files["mask"]
        )
        records.append(
            {
                "index": index,
                "frame_name": placement.frame_name,
                "time_id": placement.time_id,
                "masked_psnr": float(psnr.compute()),
                "files": files,
            }
        )

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "eval_split": eval_split,
                "num_heldout_views": len(placements),
                "alignment": views.alignment,
                "views": records,
            },
            indent=2,
        )
        + "\n"
    )
    return records


@torch.no_grad()
def export_replay(
    scene,
    cams: dict[str, list],
    out_path: str | Path,
    *,
    rasterizer,
    raster: tuple[int, int],
    fps: int = 30,
    stride: int = 1,
) -> dict[str, Any]:
    """Render every ``stride``-th training camera at its own time and encode a video.

    ``raster`` is the (width, height) the scene was trained at -- the run's stream says
    (``FrameStream.render_raster``); it is not inferred from the intrinsics. ``fps`` is the
    output video's playback rate, not a claim about the capture. Writes the video (mp4v, like
    the pipeline's flythrough) and a sidecar ``<name>.json`` with the size, fps and the time id
    of every frame.
    """
    train = sorted(cams.get("train") or [], key=lambda cam: cam.time_id)[:: max(1, int(stride))]
    if not train:
        raise ValueError("the bundle has no training cameras to replay")
    width, height = int(raster[0]), int(raster[1])
    frames = [
        _to_u8(_render(rasterizer, scene, cam.render_camera(width, height), cam.time_id))
        for cam in train
    ]
    out_path = Path(out_path)
    write_video(frames, out_path, fps=fps)
    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "time_ids": [int(cam.time_id) for cam in train],
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta

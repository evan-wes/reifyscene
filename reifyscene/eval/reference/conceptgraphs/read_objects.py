"""Read ConceptGraphs' fused objects back as per-frame instance masks.

ConceptGraphs saves a ``MapObjectList`` in which each fused object keeps parallel lists of
the detections that formed it -- ``image_idx``, ``mask``, ``class_name`` -- alongside its
accumulated point cloud. That is already a per-frame instance segmentation, so scoring it
needs no reprojection: its masks go straight into the same scorer our own masks do.

Two mappings have to be undone, and both are recorded by the export rather than guessed:

- **Frame index.** ConceptGraphs numbers frames by position in a sorted glob. DAVIS names
  its annotations by source frame number. ``source_frame_for_index`` in the export
  manifest maps one to the other.
- **Resolution.** The export writes colour in the geometry's pixels (a pure resize of the
  source frame on every DAVIS sequence), so masks come back in that raster while the
  ground truth is in the source's. They are resized back with nearest-neighbour, which is
  exact for the resize the transform applied.
"""

from __future__ import annotations

import gzip
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.io import read_json


def _to_mask(raw: Any) -> np.ndarray | None:
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(bool) if arr.ndim == 2 else None


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``size`` (w, h). Exact inverse of a pure resize."""
    from PIL import Image

    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(pil.resize(size, Image.Resampling.NEAREST)) > 127


def masks_by_frame(
    objects_path: Path,
    manifest_path: Path,
    *,
    to_source_pixels: bool = True,
    min_detections: int = 1,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Return ``({source_frame: {object_id: mask}}, info)`` from a saved ConceptGraphs run.

    ``min_detections`` drops objects fused from fewer than that many views. ConceptGraphs'
    own ``obj_min_detections`` default is 3; the default here is 1 so that filtering is a
    decision made once, visibly, by the caller rather than twice in different places.
    """
    objects_path, manifest_path = Path(objects_path), Path(manifest_path)
    manifest = read_json(manifest_path)
    source_for_index = [int(v) for v in manifest["source_frame_for_index"]]
    source_size = tuple(int(v) for v in manifest["image_transform"]["source_size"])

    with gzip.open(objects_path, "rb") as fh:
        payload = pickle.load(fh)
    objects = payload.get("objects", payload) if isinstance(payload, Mapping) else payload

    by_frame: dict[int, dict[str, np.ndarray]] = {}
    labels: dict[str, str] = {}
    kept = skipped = unmapped = 0
    for obj_id, obj in enumerate(objects):
        image_idx = list(obj.get("image_idx") or [])
        masks = list(obj.get("mask") or [])
        if len(image_idx) < min_detections or len(masks) != len(image_idx):
            skipped += 1
            continue
        kept += 1
        names = list(obj.get("class_name") or [])
        if names:  # the object's own most frequent caption, for the report
            labels[str(obj_id)] = max(set(map(str, names)), key=lambda n: names.count(n))
        for idx, raw in zip(image_idx, masks, strict=True):
            if not 0 <= int(idx) < len(source_for_index):
                unmapped += 1
                continue
            mask = _to_mask(raw)
            if mask is None or not mask.any():
                continue
            if to_source_pixels:
                mask = _resize_mask(mask, source_size)
            by_frame.setdefault(source_for_index[int(idx)], {})[str(obj_id)] = mask

    info = {
        "objects_total": len(objects),
        "objects_kept": kept,
        "objects_skipped": skipped,
        "detections_unmapped": unmapped,
        "frames_with_masks": len(by_frame),
        "labels": labels,
        "min_detections": int(min_detections),
        "scale_factor": manifest.get("scale_factor"),
        "pinhole_residual_pct": manifest.get("pinhole_residual_pct"),
    }
    return by_frame, info

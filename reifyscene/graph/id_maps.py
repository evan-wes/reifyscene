"""Per-frame id maps: which graph entity every render pixel belongs to, as integers on disk.

This is the input contract for per-Gaussian identity (I0 of
``docs/design/2026-09-29-per-gaussian-identity-scope.md``). Identity training supervises a
rasterized per-Gaussian feature with cross-entropy against one integer class per pixel, so the
entity masks have to become exactly that: one uint16 PNG per frame, in the geometry's raster,
plus a manifest that says what each integer means.

The masks themselves come from :func:`reifyscene.graph.renderable_handle.render_pixel_masks`,
the same mapping the renderable handle and the step-1 gate use (segment tracklet -> reid
canonical -> graph node, through the geometry's recorded ``ImageTransform``). Nothing here
re-derives it, so a round trip reproduces the masks the handle uses today.

Encoding -- chosen so the loss can never be taught something the masks do not say:

- ``0`` is **background**: no segment covers the pixel.
- ``1..K`` are the graph's entity nodes, numbered in sorted ``node_id`` order so the numbering
  is deterministic for a given graph.
- ``65535`` is **ignore**, for three kinds of pixel:
    * **overlap** -- two or more entities claim it (SAM 2 does this at occlusion boundaries).
      Picking a winner would train a guess.
    * **unmapped** -- a tracklet the graph dropped covers it. It is a real thing with no id;
      calling it background would teach the field that, say, a person is background.
    * **padding** -- the transform put no source pixel there.

``num_classes`` in the manifest is ``K + 1`` (background included, ignore not). It is
scene-dependent, and so is the loss's ``log(num_classes)`` normalisation, which is why it lives
in the manifest rather than a config.

Identity is time-invariant, so an entity that enters or leaves mid-clip needs no special case:
it is simply absent from the frames it is not in. The manifest records each class's span.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

BACKGROUND = 0
IGNORE = 65535
MANIFEST = "manifest.json"
FORMAT_VERSION = 1


def compose_id_map(
    entity_masks: Mapping[str, np.ndarray],
    class_of: Mapping[str, int],
    *,
    shape: tuple[int, int],
    unmapped: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """One frame's ``(H, W)`` uint16 id map, and how many pixels each ignore reason took.

    A pixel an unmapped tracklet shares with an entity counts as ``unmapped``, not
    ``overlap``: the entity may be right there, but the pixel is contested either way.
    """
    coverage = np.zeros(shape, dtype=np.int32)
    out = np.full(shape, BACKGROUND, dtype=np.uint16)
    for node_id, mask in entity_masks.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != shape:
            raise ValueError(f"mask for {node_id} is {m.shape}, raster is {shape}")
        coverage += m
        out[m] = class_of[node_id]
    overlap = coverage >= 2
    unm = np.zeros(shape, bool) if unmapped is None else np.asarray(unmapped, dtype=bool)
    pad = np.zeros(shape, bool) if valid is None else ~np.asarray(valid, dtype=bool)
    out[overlap | unm | pad] = IGNORE
    counts = {
        "overlap": int((overlap & ~unm & ~pad).sum()),
        "unmapped": int((unm & ~pad).sum()),
        "padding": int(pad.sum()),
    }
    return out, counts


def _frame_file(frame: int) -> str:
    return f"frame_{frame:05d}.png"


def _stream_num_frames(run_dir: Path, stream_id: str) -> int:
    ingest = json.loads((run_dir / "ingest" / "artifact.json").read_text())
    for raw in ingest.get("streams") or []:
        stream = raw.get("value", raw)
        if stream.get("stream_id") == stream_id:
            return int(stream["num_frames"])
    raise ValueError(f"{run_dir.name}: ingest has no stream {stream_id!r}")


def export_id_maps(
    run_dir: Path, *, stream_id: str = "cam_00", out_dir: Path | None = None
) -> dict[str, Any]:
    """Write ``run_dir/identity/frame_NNNNN.png`` + ``manifest.json``; return the manifest.

    Every ingest frame gets a map, including frames with no segment at all (all background):
    whichever frames training samples, there is a target for it.
    """
    from PIL import Image

    from reifyscene.graph.renderable_handle import render_pixel_masks

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "identity"
    placed = render_pixel_masks(run_dir, stream_id=stream_id)
    transform = placed.transform
    width, height = transform.output_size
    shape = (int(height), int(width))
    src_w, src_h = transform.source_size
    valid = transform.apply_to_image(np.ones((src_h, src_w), np.uint8), kind="mask").astype(bool)

    node_ids = sorted(placed.labels)
    if len(node_ids) >= IGNORE:
        raise ValueError(f"{run_dir.name}: {len(node_ids)} entities do not fit in uint16")
    class_of = {nid: i + 1 for i, nid in enumerate(node_ids)}

    num_frames = _stream_num_frames(run_dir, stream_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class: dict[str, dict[str, Any]] = {
        nid: {"frames": [], "pixels": 0} for nid in node_ids
    }
    totals = {"overlap": 0, "unmapped": 0, "padding": 0, "background": 0, "entity": 0}
    frames_out: list[dict[str, Any]] = []
    for frame in range(num_frames):
        masks = placed.by_frame.get(frame, {})
        id_map, counts = compose_id_map(
            masks, class_of, shape=shape, unmapped=placed.unmapped.get(frame), valid=valid
        )
        Image.fromarray(id_map).save(out_dir / _frame_file(frame))
        for key, n in counts.items():
            totals[key] += n
        totals["background"] += int((id_map == BACKGROUND).sum())
        for nid in masks:
            px = int((id_map == class_of[nid]).sum())
            per_class[nid]["pixels"] += px
            if px:
                per_class[nid]["frames"].append(frame)
        totals["entity"] += int(((id_map != BACKGROUND) & (id_map != IGNORE)).sum())
        frames_out.append({"frame_idx": frame, "file": _frame_file(frame)})

    total_px = num_frames * shape[0] * shape[1]
    classes = [{"index": BACKGROUND, "node_id": None, "label": "background"}]
    for nid in node_ids:
        fr = per_class[nid]["frames"]
        classes.append({
            "index": class_of[nid],
            "node_id": nid,
            "label": placed.labels[nid],
            "frames_present": len(fr),
            "first_frame": fr[0] if fr else None,
            "last_frame": fr[-1] if fr else None,
            "pixels": per_class[nid]["pixels"],
        })
    notes = list(placed.notes)
    for c in classes[1:]:
        if not c["frames_present"]:
            notes.append(f"{c['node_id']} has a class but no unambiguous pixel on any frame")
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "run": run_dir.name,
        "stream_id": stream_id,
        "raster": {"width": int(width), "height": int(height)},
        "image_transform": transform.to_dict(),
        "encoding": {"dtype": "uint16", "background": BACKGROUND, "ignore": IGNORE},
        "num_classes": len(classes),
        "classes": classes,
        "frames": frames_out,
        "fractions": {
            k: round(v / total_px, 6) if total_px else 0.0 for k, v in totals.items()
        } | {
            "ignore": round(
                (totals["overlap"] + totals["unmapped"] + totals["padding"]) / total_px, 6
            ) if total_px else 0.0
        },
        "notes": notes,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_id_maps(id_dir: Path) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """``(manifest, {frame_idx: (H, W) uint16})`` from a directory :func:`export_id_maps` wrote."""
    from PIL import Image

    id_dir = Path(id_dir)
    manifest = json.loads((id_dir / MANIFEST).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{id_dir}: id-map format {manifest.get('format_version')!r} unknown")
    h, w = manifest["raster"]["height"], manifest["raster"]["width"]
    maps: dict[int, np.ndarray] = {}
    for entry in manifest["frames"]:
        arr = np.asarray(Image.open(id_dir / entry["file"]))
        if arr.dtype != np.uint16 or arr.shape != (h, w):
            raise ValueError(f"{entry['file']}: {arr.dtype} {arr.shape}, expected uint16 {(h, w)}")
        maps[int(entry["frame_idx"])] = arr
    return manifest, maps


def masks_from_id_maps(
    manifest: Mapping[str, Any], maps: Mapping[int, np.ndarray], frames: Iterable[int] | None = None
) -> dict[int, dict[str, np.ndarray]]:
    """Invert the encoding: ``{frame: {node_id: mask}}`` for every entity present on a frame."""
    index_to_node = {c["index"]: c["node_id"] for c in manifest["classes"] if c["node_id"]}
    out: dict[int, dict[str, np.ndarray]] = {}
    for frame in frames if frames is not None else sorted(maps):
        arr = maps[frame]
        slot = {
            index_to_node[int(i)]: arr == i
            for i in np.unique(arr)
            if int(i) in index_to_node
        }
        if slot:
            out[int(frame)] = slot
    return out


def round_trip_mismatches(
    manifest: Mapping[str, Any],
    maps: Mapping[int, np.ndarray],
    by_frame: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, int]:
    """Pixels where the decoded maps disagree with the source masks, per node id.

    Agreement means: on every non-ignore pixel, an entity's decoded mask equals its source
    mask. Ignore pixels are excluded by construction -- that is where the encoding chose not
    to assert anything. An empty dict is a perfect round trip.
    """
    decoded = masks_from_id_maps(manifest, maps)
    known = {c["node_id"] for c in manifest["classes"] if c["node_id"]}
    bad: dict[str, int] = {}
    for frame, arr in maps.items():
        keep = arr != IGNORE
        src = by_frame.get(frame, {})
        dec = decoded.get(frame, {})
        for nid in set(src) | set(dec):
            if nid not in known:
                bad[nid] = bad.get(nid, 0) + int(np.asarray(src[nid], bool).sum())
                continue
            a = np.asarray(src.get(nid, np.zeros_like(keep)), bool) & keep
            b = np.asarray(dec.get(nid, np.zeros_like(keep)), bool)
            n = int((a ^ b).sum())
            if n:
                bad[nid] = bad.get(nid, 0) + n
    return bad

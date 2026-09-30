"""Score removals against DAVIS ground truth: is what went the object, and is the rest intact?

For every annotated object in a run: match it to a graph node (IoU pooled over the clip,
one-to-one, greedy), remove that node with the selection the graph's ``gaussians`` layer holds
(:mod:`reifyscene.graph.handles`), render before / after on training frames where the object is
annotated, and score against the **ground-truth** mask -- never our own masks, which the
selection was computed from.

Per object, averaged over its frames:

- **taken recall / precision**: a pixel of the unedited render is *taken* when the removed
  Gaussians contribute over half its composited weight. Recall: how much of the object went.
  Precision: how much of what went is the object. Independent of what is behind it.
- **alone precision**: the removed Gaussians rendered by themselves, which also shows what
  they hid behind other things -- junk a new viewpoint would reveal.
- **halo ratio**: changed pixels outside the object, per pixel of the object.
- **others damaged**: share of the other annotated objects' pixels that changed.
- **rest PSNR**: before vs after, over every pixel farther than ``ring`` px from the object.
  The direct answer to "did the edit touch the rest of the scene": identical is 60 dB (capped).

The frames are the training views: the reconstruction was fitted to them and they never see
behind the object. Damage these cannot show -- what a moved camera reveals -- needs held-out
views (DyCheck), which is why this is the optimistic measure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PSNR_CAP = 60.0


def gt_masks(ann_dir: Path, frame: int, transform) -> dict[int, np.ndarray]:
    """``{object id: bool mask}`` of one DAVIS annotation frame, placed in the render raster."""
    from PIL import Image

    path = ann_dir / f"{frame:05d}.png"
    if not path.is_file():
        return {}
    arr = np.asarray(Image.open(path))
    return {int(i): transform.apply_to_image((arr == i).astype(np.uint8), kind="mask") > 0
            for i in np.unique(arr) if int(i) != 0}


def match_objects(gt_by_frame, node_by_frame, *, match_iou: float):
    """Greedy one-to-one GT object -> node on IoU pooled over frames.

    Returns ``(matched {gid: (node, iou)}, best {gid: (node, iou)} regardless of the threshold,
    area_gt {gid: pixels})``.
    """
    inter: dict[tuple[int, str], int] = {}
    area_gt: dict[int, int] = {}
    area_node: dict[str, int] = {}
    for f, gts in gt_by_frame.items():
        nodes = node_by_frame.get(f, {})
        for gid, g in gts.items():
            area_gt[gid] = area_gt.get(gid, 0) + int(g.sum())
            for nid, m in nodes.items():
                inter[(gid, nid)] = inter.get((gid, nid), 0) + int((g & m).sum())
    for nodes in node_by_frame.values():
        for nid, m in nodes.items():
            area_node[nid] = area_node.get(nid, 0) + int(m.sum())
    pairs = []
    for (gid, nid), i in inter.items():
        union = area_gt[gid] + area_node[nid] - i
        if union:
            pairs.append((i / union, gid, nid))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    taken_g, taken_n, out = set(), set(), {}
    for iou, gid, nid in pairs:
        if iou < match_iou or gid in taken_g or nid in taken_n:
            continue
        taken_g.add(gid)
        taken_n.add(nid)
        out[gid] = (nid, round(float(iou), 4))
    best = {gid: max((p for p in pairs if p[1] == gid), default=(0.0, gid, None))
            for gid in area_gt}
    return out, {gid: (b[2], round(float(b[0]), 4)) for gid, b in best.items()}, area_gt


def changed(a: np.ndarray, b: np.ndarray, thr: int) -> np.ndarray:
    """Pixels whose summed absolute RGB difference exceeds ``thr``."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(-1) > thr


def fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if mask.shape == shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                      .resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 127


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Square dilation by ``px`` pixels (no scipy: shifts of a padded mask)."""
    if px <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(px):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def change_score(ch: np.ndarray, mask: np.ndarray, others: np.ndarray | None = None) -> dict:
    """What changed inside the object, outside it, and on the other annotated objects.

    ``rest_intact`` divides by the whole frame outside the object, so tearing a few thousand
    pixels off a neighbour still reads ~97%. ``halo_ratio`` (changed pixels outside, per pixel
    of the object) and ``others_damaged`` (share of the *other annotated objects'* pixels that
    changed) are the measures that cannot hide that.
    """
    inside = int((ch & mask).sum())
    outside = int((ch & ~mask).sum())
    out = {
        "entity_removed": round(inside / max(1, int(mask.sum())), 4),
        "rest_intact": round(1.0 - outside / max(1, int((~mask).sum())), 6),
        "halo_px": outside,
        "halo_ratio": round(outside / max(1, int(mask.sum())), 4),
    }
    if others is not None:
        o = others & ~mask
        hit = int((ch & o).sum())
        out["others_damaged"] = round(hit / int(o.sum()), 4) if o.any() else None
    return out


def taken_score(alpha: np.ndarray, gt: np.ndarray, prefix: str = "taken") -> dict:
    took = alpha > 0.5
    hit = int((took & gt).sum())
    return {f"{prefix}_recall": round(hit / max(1, int(gt.sum())), 4),
            f"{prefix}_precision": round(hit / int(took.sum()), 4) if took.any() else None,
            f"{prefix}_px": int(took.sum())}


def rest_psnr(before: np.ndarray, after: np.ndarray, keep: np.ndarray) -> float | None:
    """PSNR of ``after`` against ``before`` over ``keep`` pixels, capped at ``PSNR_CAP``."""
    if not keep.any():
        return None
    d = before[keep].astype(np.float64) - after[keep].astype(np.float64)
    mse = float((d * d).mean())
    if mse == 0.0:
        return PSNR_CAP
    return round(min(PSNR_CAP, 10.0 * np.log10(255.0 ** 2 / mse)), 3)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


METRICS = ("taken_recall", "taken_precision", "alone_precision", "halo_ratio",
           "others_damaged", "rest_psnr_db")


def score_run(run: Path, *, checkpoint_sub: str, davis: Path, frames: int = 4,
              match_iou: float = 0.25, changed_threshold: int = 12, ring: int = 8,
              device: str = "cuda") -> dict[str, Any]:
    """Every annotated object of one DAVIS run: matched, removed via the layer, scored."""
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.graph.renderable_handle import drop_gaussians, render_pixel_masks

    run = Path(run)
    seq = run.name.split("_", 1)[-1]
    ann_dir = davis / "Annotations" / "480p" / seq
    placed = render_pixel_masks(run)
    ingest = json.loads((run / "ingest" / "artifact.json").read_text())
    n_frames = int(next(s.get("value", s) for s in ingest["streams"])["num_frames"])
    gt_by_frame = {f: gt_masks(ann_dir, f, placed.transform) for f in range(n_frames)}
    matched, best, _ = match_objects(gt_by_frame, placed.by_frame, match_iou=match_iou)

    ed = RemoveEditor(run, checkpoint_sub=checkpoint_sub, device=device)
    r = ed.renderer
    graph = json.loads((run / "graph" / "artifact.json").read_text())
    layer_status = {n["value"]["node_id"]: n["value"]["attrs"].get("gaussians", {}).get("status")
                    for n in graph["nodes"]}

    objects = []
    for gid in sorted({g for gts in gt_by_frame.values() for g in gts}):
        row: dict[str, Any] = {"gt_id": gid, "best_node": best[gid][0],
                               "best_iou": best[gid][1]}
        objects.append(row)
        if gid not in matched:
            row["status"] = "not_proposed"
            continue
        node, iou = matched[gid]
        row.update(node_id=node, label=placed.labels.get(node, ""), match_iou=iou)
        fg, bg, stats = ed.select(node)
        row.update(source=stats.get("source"), removed_fg=int(len(fg)), removed_bg=int(len(bg)))
        if len(fg) + len(bg) == 0:
            row["status"] = "unassignable"
            row["layer_status"] = layer_status.get(node)
            continue
        row["status"] = "scored"
        present = [f for f in range(n_frames) if gid in gt_by_frame[f] and f in r.time_of]
        step = max(1, len(present) // max(1, frames))
        use = present[::step][:frames]
        after = r.after(drop_gaussians(ed.bundle, fg, bg_indices=bg), use)
        alone = r.taken(fg, bg, use)
        share = r.share(fg, bg, use)
        per_frame = []
        for f in use:
            before = r.before(f)
            ch = changed(before, after[f], changed_threshold)
            gt = fit(gt_by_frame[f][gid], ch.shape)
            others = np.zeros_like(gt)
            for og, om in gt_by_frame[f].items():
                if og != gid:
                    others |= fit(om, ch.shape)
            per_frame.append({
                "frame": f, **change_score(ch, gt, others), **taken_score(share[f], gt),
                **taken_score(alone[f][1], gt, prefix="alone"),
                "rest_psnr_db": rest_psnr(before, after[f], ~dilate(gt, ring))})
        row["frames"] = per_frame
        row.update({k: _mean(per_frame, k) for k in METRICS})
    return {"run": run.name, "sequence": seq, "checkpoint": checkpoint_sub,
            "layer": (graph.get("metadata") or {}).get("layers", {}).get("gaussians"),
            "objects": objects}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by status, and the median of each metric over scored objects."""
    objs = [o for r in rows for o in r["objects"]]
    scored = [o for o in objs if o.get("status") == "scored"]
    out: dict[str, Any] = {"objects": len(objs)}
    for s in ("scored", "unassignable", "not_proposed"):
        out[s] = sum(o.get("status") == s for o in objs)
    for k in METRICS:
        vals = [o[k] for o in scored if o.get(k) is not None]
        out[f"median_{k}"] = round(float(np.median(vals)), 4) if vals else None
    return out

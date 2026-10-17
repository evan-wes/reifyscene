#!/usr/bin/env python
"""Remove an entity from a trained scene and render before/after -- the spike's step-1 gate.

    python scripts/render_entity_edit.py slice_blackswan --node auto

The gate the spike sets is *"is the removed region the entity, and is what remains intact?"*
This answers both, and measures rather than only shows them:

- **before / after / difference** stills, rendered from the training cameras, so the render is
  fidelity rather than a novel view whose own artifacts would confound the question.
- **changed-vs-mask IoU.** The pixels the edit changed are compared against the entity's own
  mask in the render's raster. High IoU means the hole is the entity; pixels changing far from
  the mask mean the handle claimed Gaussians belonging to something else.
- **untouched fraction** outside the mask: what remains intact, as a number.

`remove` is executed as a checkpoint transform -- the entity's Gaussians are dropped and the
scene is rebuilt -- so what renders afterwards is a real scene rather than a masked image, and
nothing in the render path needs an exclusion argument threaded through it.

Needs CUDA: the rasterizer is gsplat-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run, by name in the data directory's runs/")
    ap.add_argument("--node", default="auto",
                    help="node id, a label, or 'auto' for the largest claim")
    ap.add_argument("--frames", type=int, default=3, help="how many training frames to render")
    ap.add_argument("--out", default="entity_edit", help="output folder name inside the run")
    ap.add_argument("--changed-threshold", type=int, default=12,
                    help="per-pixel L1 over 0-255 channels above which a pixel counts as changed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.common.frame_keys import parse_key
    from reifyscene.graph.renderable_handle import (
        compute_renderable_handles,
        drop_gaussians,
        entity_masks_in_render_pixels,
    )
    from reifyscene.render.eval.checkpoint_views import _render, _to_u8
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
    from reifyscene.render.gaussian4d.rasterizer import build_rasterizer

    run = RunOutputs.open(data_dir.path(data_dir.RUNS, args.run))
    checkpoint = run.checkpoint()
    out_dir = run.output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = compute_renderable_handles(checkpoint)
    if args.node == "auto":
        pick = max(handles.entities.values(), key=lambda e: e.assigned)
    else:
        matches = [e for e in handles.entities.values()
                   if e.node_id == args.node or e.label == args.node]
        if not matches:
            raise SystemExit(f"no entity matching {args.node!r}; have: "
                             + ", ".join(f"{e.node_id} ({e.label})"
                                         for e in handles.entities.values()))
        pick = max(matches, key=lambda e: e.assigned)
    if pick.unassignable:
        raise SystemExit(f"{pick.node_id} is unassignable: no Gaussians to remove")
    print(f"removing {pick.node_id} ({pick.label}): {pick.assigned} of "
          f"{handles.num_foreground} foreground Gaussians ({pick.assigned_fraction:.1%})")

    bundle = checkpoint.load()
    edited = drop_gaussians(bundle, pick.gaussian_indices)

    masks, _labels, _notes = entity_masks_in_render_pixels(run)
    frame_names = [str(n) for n in np.asarray(bundle["train_frame_names"])]
    time_ids = [int(t) for t in np.asarray(bundle["train_time_ids"])]
    stream = run.streams.only()
    source = [parse_key(n)[1] for n in frame_names]
    # Render frames where the entity actually has a mask -- elsewhere there is nothing to see.
    candidates = [i for i, f in enumerate(source) if pick.node_id in masks.get(f, {})]
    if not candidates:
        raise SystemExit(f"{pick.node_id} has no mask on any training frame")
    step = max(1, len(candidates) // max(1, args.frames))
    chosen = candidates[::step][: args.frames]

    scene_a, cams = bundle_to_scene(bundle, device=args.device)
    scene_b, _ = bundle_to_scene(edited, device=args.device)
    rasterizer = build_rasterizer({"device": args.device})
    train = sorted(cams.get("train") or [], key=lambda c: c.time_id)
    by_time = {int(c.time_id): c for c in train}
    width, height = stream.render_raster

    rows = []
    for i in chosen:
        cam = by_time.get(time_ids[i])
        if cam is None:
            continue
        rc = cam.render_camera(width, height)
        with torch.no_grad():
            before = _to_u8(_render(rasterizer, scene_a, rc, time_ids[i]))
            after = _to_u8(_render(rasterizer, scene_b, rc, time_ids[i]))
        diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).sum(-1)
        changed = diff > int(args.changed_threshold)
        mask = masks[source[i]][pick.node_id]
        if mask.shape != changed.shape:   # both are the recorded raster; a mismatch is a bug
            raise SystemExit(f"mask {mask.shape} and render {changed.shape} rasters differ")
        inter = int((changed & mask).sum())
        union = int((changed | mask).sum())
        outside = int((changed & ~mask).sum())
        rows.append({
            "frame": source[i], "time_id": time_ids[i],
            "changed_px": int(changed.sum()), "mask_px": int(mask.sum()),
            # The gate asks two questions, so they are reported as two numbers. A single IoU
            # conflates them and reads as failure when the entity was removed cleanly but its
            # Gaussians' footprints reach past its silhouette.
            "entity_removed": round(inter / max(1, int(mask.sum())), 4),      # of the entity
            "untouched_outside_mask": round(1.0 - outside / max(1, int((~mask).sum())), 6),
            "halo_px": outside,
            "changed_vs_mask_iou": round(inter / union, 4) if union else 0.0,
            "changed_inside_mask": round(inter / max(1, int(changed.sum())), 4),
        })
        tag = f"{source[i]:05d}"
        Image.fromarray(before).save(out_dir / f"{tag}_before.png")
        Image.fromarray(after).save(out_dir / f"{tag}_after.png")
        Image.fromarray(changed.astype(np.uint8) * 255).save(out_dir / f"{tag}_changed.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(out_dir / f"{tag}_mask.png")
        print(f"  frame {source[i]:4d}  entity removed {rows[-1]['entity_removed']:6.1%}  "
              f"rest intact {rows[-1]['untouched_outside_mask']:7.2%}  "
              f"halo {rows[-1]['halo_px']:6d}px  (IoU {rows[-1]['changed_vs_mask_iou']:.3f})")

    summary = {
        "run": run.name, "node_id": pick.node_id, "label": pick.label,
        "removed_gaussians": pick.assigned, "foreground_total": handles.num_foreground,
        "removed_fraction": round(pick.assigned_fraction, 4),
        "changed_threshold": int(args.changed_threshold), "frames": rows,
        "mean_entity_removed": round(
            float(np.mean([r["entity_removed"] for r in rows])), 4) if rows else 0.0,
        "mean_untouched_outside_mask": round(
            float(np.mean([r["untouched_outside_mask"] for r in rows])), 4) if rows else 0.0,
        "mean_changed_vs_mask_iou": round(
            float(np.mean([r["changed_vs_mask_iou"] for r in rows])), 4) if rows else 0.0,
    }
    (out_dir / "index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGATE  entity removed {summary['mean_entity_removed']:.1%}   "
          f"rest intact {summary['mean_untouched_outside_mask']:.2%}   -> {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Judge a removal against photos, from the training cameras and from cameras it never saw.

    python scripts/heldout_edit_damage.py slice_spin [--entity 1] [--views 60]

G4 part 3 of docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md. The DAVIS benchmark
(``benchmarks/davis_edit_damage.json``) scores training views only. A DyCheck run also has
held-out cameras at other positions, with real photos and covisibility masks.

What makes this a measurement rather than a tautology: where the entity *is* in each view comes
from the photo, not from the removal. Every person in the photo is segmented
(:mod:`reifyscene.eval.benchmarks.edit_damage.silhouettes`: person detector + SAM 2 box prompt,
pseudo ground truth, saved for checking), and the one overlapping where the entity renders is
its silhouette ``S``. Then, per view:

- **taken recall**: of ``S``, the share of pixels where the removed Gaussians carry over half
  the composited weight in the unedited render. How much of the person the removal takes.
- **alone precision**: the removed Gaussians rendered by themselves; the share of their pixels
  inside ``S`` dilated by ``--ring`` px. What they take that is *not* the person lands outside
  -- the "shadow" behind an entity one camera never saw around.
- **rest**: covisible pixels outside dilated ``S``. ``rest_change_db`` compares after with
  before (60 = untouched); ``rest_psnr_before`` / ``_after`` score each against the photo, and
  their difference is the damage the edit did to the rest of the scene.

The same measures on training views, against silhouettes from the training photos, give the
held-out-minus-training difference: what the extra viewpoints reveal.

Writes ``<run>/edits/heldout_damage/entity_<n>.json``, per-view masks (``masks/``) and a few
check strips (photo + silhouette in green and the removed footprint in red | before | after |
removed alone). Needs CUDA (gsplat).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _u8(t) -> np.ndarray:
    return t.clamp(0, 1).mul(255).round().byte().cpu().numpy()


def _med(rows, key):
    vals = [x[key] for x in rows if x.get(key) is not None]
    return round(float(np.median(vals)), 4) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run, by name in the data directory's runs/")
    ap.add_argument("--entity", default=None,
                    help="node id or tracklet number (default: the grounded person with the "
                         "most Gaussians)")
    ap.add_argument("--checkpoint", default="render",
                    help="the checkpoint scored: the render stage's own, or a named retrain")
    ap.add_argument("--views", type=int, default=60, help="held-out views scored (evenly spaced)")
    ap.add_argument("--train-views", type=int, default=20)
    ap.add_argument("--ring", type=int, default=8, help="px of tolerance around the silhouette")
    ap.add_argument("--min-iou", type=float, default=0.2,
                    help="a person counts as the entity only above this IoU with its render")
    ap.add_argument("--strips", type=int, default=4)
    ap.add_argument("--out-name", default="heldout_damage",
                    help="the run's output dir to write into")
    ap.add_argument("--body-bound", action="store_true",
                    help="bound the person's selection by their body volume (experimental)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.edit.renderer import outline
    from reifyscene.eval.benchmarks.edit_damage import dilate, rest_psnr
    from reifyscene.eval.benchmarks.edit_damage.silhouettes import PersonSegmenter, pick_entity
    from reifyscene.graph.node_ids import tracklet_number
    from reifyscene.graph.renderable_handle import drop_gaussians
    from reifyscene.query.scene_query import SceneQuery
    from reifyscene.render.eval.masked_nvs_benchmark import (
        default_alignment,
        eval_dataset,
        load_heldout_view,
        place_heldout_views,
    )
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    run = RunOutputs.open(data_dir.path(data_dir.RUNS, args.run))
    cfg = run.config
    eval_cfg = dict((cfg.get("evaluation") or {}).get("masked_nvs") or {})
    if not eval_cfg.get("dataset_root"):
        raise SystemExit(f"{run.name}: no evaluation.masked_nvs.dataset_root -- not a held-out run")
    eval_cfg.setdefault("align", default_alignment(cfg.get("geometry", {}).get("backend_name")))
    eval_split = str(eval_cfg.get("eval_split", "val"))

    nodes = [n["value"] for n in run.record("graph")["nodes"]]
    if args.entity:
        # refuses an ambiguous reference
        nid = SceneQuery(run.artifact("graph")).entity(str(args.entity)).node_id
        node = next(n for n in nodes if n["node_id"] == nid)
    else:
        people = [n for n in nodes if n["node_type"] == "human"
                  and n["attrs"].get("gaussians", {}).get("status") == "grounded"]
        if not people:
            raise SystemExit(f"{run.name}: no grounded person (run the handles stage first)")
        node = max(people, key=lambda n: n["attrs"]["gaussians"]["fg"]
                   + n["attrs"]["gaussians"]["bg"])
    nid = node["node_id"]

    stream = run.streams.only()
    ed = RemoveEditor(run.checkpoint(args.checkpoint), device=args.device,
                      body_bound=args.body_bound)
    fg, bg, stats = ed.select(nid)
    if len(fg) + len(bg) == 0:
        raise SystemExit(f"{nid} has no Gaussians to remove")
    r = ed.renderer
    n_fg = int(np.asarray(ed.bundle["fg_means"]).shape[0])
    n_bg = int(np.asarray(ed.bundle["bg_means"]).shape[0]) if "bg_means" in ed.bundle else 0
    after_scene, _ = bundle_to_scene(drop_gaussians(ed.bundle, fg, bg_indices=bg),
                                     device=args.device)
    alone_scene, _ = bundle_to_scene(
        drop_gaussians(ed.bundle, np.setdiff1d(np.arange(n_fg), fg),
                       bg_indices=np.setdiff1d(np.arange(n_bg), bg)), device=args.device)
    ind = torch.zeros((r.scene.num_gaussians, 1), device=args.device)
    ind[torch.as_tensor(fg, device=args.device, dtype=torch.long)] = 1.0
    ind[n_fg + torch.as_tensor(bg, device=args.device, dtype=torch.long)] = 1.0

    def draw(scene, cam, t, features=None):
        with torch.no_grad():
            c = scene.cloud_at(int(t))
            return r.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, cam,
                                          features=features)

    seg = PersonSegmenter(device=args.device)
    out_dir = run.output_dir(args.out_name)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    tag = tracklet_number(nid)

    def score_view(name, photo, cam, t, covis, strip: bool) -> dict:
        h, w = photo.shape[:2]
        before, after = draw(r.scene, cam, t), draw(after_scene, cam, t)
        alone = draw(alone_scene, cam, t)
        share = draw(r.scene, cam, t, features=ind).features[..., 0].reshape(h, w).cpu().numpy()
        taken = share > 0.5
        alone_px = alone.alpha.reshape(h, w).cpu().numpy() > 0.5
        person, iou = pick_entity(seg.people(photo), taken)
        row = {"view": name, "scene_time": int(t), "silhouette_iou_with_render": round(iou, 4),
               "silhouette_source": seg.backend}
        if person is None or iou < args.min_iou:
            row["note"] = "no person in the photo overlaps where the entity renders"
            return row
        s = person.mask
        s_ring = dilate(s, args.ring)
        rest = covis & ~s_ring
        b, a = _u8(before.rgb), _u8(after.rgb)
        row.update({
            "silhouette_px": int(s.sum()),
            "taken_recall": round(int((taken & s).sum()) / max(1, int(s.sum())), 4),
            "alone_precision": (round(int((alone_px & s_ring).sum()) / int(alone_px.sum()), 4)
                                if alone_px.any() else None),
            "alone_outside_px": int((alone_px & ~s_ring).sum()),
            "rest_change_db": rest_psnr(b, a, rest),
            "rest_psnr_before": rest_psnr(photo, b, rest),
            "rest_psnr_after": rest_psnr(photo, a, rest),
        })
        if None not in (row["rest_psnr_before"], row["rest_psnr_after"]):
            row["rest_damage_db"] = round(row["rest_psnr_before"] - row["rest_psnr_after"], 4)
        Image.fromarray(s.astype(np.uint8) * 255).save(out_dir / "masks" / f"{name}.png")
        if strip:
            check = photo.copy()
            for m, colour in ((s, (40, 220, 90)), (alone_px, (255, 64, 64))):
                o = outline(m, width=2, color=colour)
                check[o[..., 3] > 0] = o[o[..., 3] > 0][:, :3]
            Image.fromarray(np.concatenate([check, b, a, _u8(alone.rgb)], axis=1)).save(
                out_dir / f"entity_{tag}_{name}.jpg", quality=88)
        return row

    # held-out views
    backend = eval_dataset(eval_cfg)
    _, cams = bundle_to_scene(ed.bundle, device=args.device)
    views = place_heldout_views(ed.bundle, cams, eval_cfg, backend)
    if views is None:
        raise SystemExit("held-out cameras cannot be placed in this scene")
    chosen = views.placements[:: max(1, len(views.placements) // max(1, args.views))][: args.views]
    every = max(1, len(chosen) // max(1, args.strips))
    held = []
    for i, p in enumerate(chosen):
        cam, gt, covis = load_heldout_view(p, backend, eval_split, args.device)
        h, w = gt.shape[:2]
        held.append(score_view(f"heldout_{p.frame_name}", _u8(gt), cam, p.scene_time,
                               covis.reshape(h, w).cpu().numpy() > 0, strip=i % every == 0))

    # training views, against the training photos in the render raster
    train = []
    frames = r.frames[:: max(1, len(r.frames) // max(1, args.train_views))][: args.train_views]
    for i, f in enumerate(frames):
        t = r.time_of[f]
        photo = stream.frame_in_render(f)
        train.append(score_view(f"train_{stream.key(f).replace(':', '_')}", photo,
                                r.cams[t].render_camera(*r.size), t,
                                np.ones(photo.shape[:2], bool),
                                strip=i % max(1, len(frames) // 2) == 0))

    keys = ("taken_recall", "alone_precision", "rest_change_db", "rest_damage_db",
            "rest_psnr_before", "rest_psnr_after")
    summary = {split: {"views": len(rows), "with_silhouette": sum("taken_recall" in x
                                                                  for x in rows),
                       **{f"median_{k}": _med(rows, k) for k in keys}}
               for split, rows in (("heldout", held), ("train", train))}
    for k in ("taken_recall", "alone_precision"):
        a, b = summary["heldout"][f"median_{k}"], summary["train"][f"median_{k}"]
        summary[f"heldout_minus_train_{k}"] = round(a - b, 4) if None not in (a, b) else None
    result = {
        "run": run.name, "entity": nid, "label": node.get("label"),
        "checkpoint": args.checkpoint, "selection": stats,
        "removed_fg": int(len(fg)), "removed_bg": int(len(bg)),
        "silhouettes": f"pseudo ground truth: {seg.backend}; masks saved under masks/",
        "alignment": views.alignment, "ring_px": args.ring, "summary": summary,
        "heldout_views": held, "train_views": train,
    }
    (out_dir / f"entity_{tag}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("entity", "label", "removed_fg", "removed_bg",
                                             "silhouettes", "summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

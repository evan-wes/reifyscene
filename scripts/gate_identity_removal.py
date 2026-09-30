#!/usr/bin/env python
"""The step-1 gate, re-run with learned identity -- scored against DAVIS ground truth.

    python scripts/gate_identity_removal.py ../runs/loop_*

I4 of docs/design/2026-09-29-per-gaussian-identity-scope.md. For every hand-annotated DAVIS
object in each run:

1. **Match it to a graph node** by mask IoU pooled over the clip (sum of intersections over
   sum of unions, both in the render raster), one-to-one and greedy, reusing the inventory
   benchmark's matcher. An object no node matches is recorded as *not proposed*: an upstream
   miss identity cannot fix, kept out of identity's numbers.
2. **Remove it two ways**, each on its own checkpoint:
   - *baseline* -- the projection-vote handle on the pipeline's ``render/`` checkpoint
     (foreground only; what the recorded gate used);
   - *identity* -- every foreground AND background Gaussian whose trained identity
     probability for the node is >= tau, on ``<run>/<identity-sub>/``, for each ``--tau``.
3. **Render before/after** on up to ``--frames`` training frames where the object is
   annotated, and score the changed pixels against the **ground-truth** mask -- not our own
   masks, which identity was trained on, so scoring against them would be circular. Our
   node's mask is scored alongside for reference.

Per object, method and tau, two questions answered separately:

- **Is the removed region the entity?** Each pixel of the *unedited* render is split by where
  its composited weight came from: a one-channel feature (1 on the removed Gaussians, 0
  elsewhere) is rasterized with the scene, so it carries occlusion exactly. A pixel counts as
  taken when the removed Gaussians contribute over half of it. Scored against the GT mask as
  *taken recall* (how much of the object was taken) and *taken precision* (how much of what
  was taken is the object). This does not depend on what, if anything, was reconstructed
  behind the object -- a still person against unseen background leaves a smear on removal
  whatever the selection, which "changed pixels" cannot tell apart from a bad selection.
  The strip's "took" panels render the removed Gaussians *alone*, which also shows the
  parts hidden behind other things in the training views; *alone recall / precision* score
  that view too. Both are reported because they disagree: a selection can be clean where it
  is visible and still carry hidden junk that removal takes and a new viewpoint would show.
- **Is the rest intact?** *rest intact* is the fraction of pixels outside the GT mask that did
  not change; *halo* counts those that did. *entity removed* (changed pixels inside the mask)
  is kept for comparison with the first gate, which reported it.

A third method, *flashsplat@s*, needs no training: FlashSplat's evidence-majority assignment
(:mod:`reifyscene.graph.assignment`) on the pipeline's own ``render/`` checkpoint, from every
training frame's id map, at each ``--slackness``.

``--base-sub`` picks the checkpoint the old vote and FlashSplat run on (default the
pipeline's ``render/``); ``--identity-sub none`` leaves learned identity out.

Writes ``gate/index.json`` under the identity checkpoint (or the base one without identity)
and, per object-frame, one strip ``obj<id>_<frame>_strip.jpg`` whose panels are listed in
each object's ``strip_panels`` (before | old after | old took | flashsplat after | flashsplat
took [| identity before | identity after | identity took]; identity at tau=0.5, flashsplat at
the first slackness) and ``obj<id>_<frame>_gt.png``, the GT outline.

Needs CUDA (gsplat). Renders only; a few minutes for all ten loop runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reifyscene.edit.renderer import CheckpointRenderer as _Renderer  # noqa: E402
from reifyscene.edit.renderer import outline as _outline  # noqa: E402
from reifyscene.eval.benchmarks.edit_damage.score import (  # noqa: E402
    change_score as _score,
)
from reifyscene.eval.benchmarks.edit_damage.score import (  # noqa: E402
    changed as _changed,
)
from reifyscene.eval.benchmarks.edit_damage.score import fit as _fit  # noqa: E402
from reifyscene.eval.benchmarks.edit_damage.score import (  # noqa: E402
    gt_masks as _gt_masks,
)
from reifyscene.eval.benchmarks.edit_damage.score import match_objects  # noqa: E402
from reifyscene.eval.benchmarks.edit_damage.score import (  # noqa: E402
    taken_score as _taken_score,
)


def _mean(rows: list[dict], ref: str, key: str) -> float | None:
    vals = [r[ref][key] for r in rows if r[ref].get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None




def filter_members(scene, fg_idx, bg_idx):
    from reifyscene.graph.assignment import filter_members as _filter

    return _filter(scene, fg_idx, bg_idx)


def _flashsplat_handles(run: Path, rend: _Renderer, slackness: list[float], device: str):
    """``{slackness: {node: (fg, bg)}}`` from evidence over every training frame's id map."""
    from reifyscene.graph.assignment import (
        assign,
        assignment_handles,
        evidence_matrix,
        id_map_views,
    )
    from reifyscene.graph.id_maps import export_id_maps, load_id_maps

    if not (run / "identity" / "manifest.json").is_file():
        export_id_maps(run)
    manifest, maps = load_id_maps(run / "identity")
    shape = (rend.size[1], rend.size[0])
    if next(iter(maps.values())).shape != shape:
        raise ValueError(f"{run.name}: id maps are {next(iter(maps.values())).shape}, "
                         f"renders are {shape}")
    node_ids = [c["node_id"] for c in sorted(manifest["classes"], key=lambda c: c["index"])]
    cams = {f: (rend.cams[t].render_camera(*rend.size), t) for f, t in rend.time_of.items()}
    counts = evidence_matrix(rend.scene, rend.rasterizer, id_map_views(maps, cams, device),
                             len(node_ids))
    n_fg = rend.scene.fg.num_gaussians
    return {s: assignment_handles(assign(counts, slackness=s), n_fg, node_ids)
            for s in slackness}


def gate_run(run: Path, args) -> dict:
    from reifyscene.graph.renderable_handle import (
        compute_renderable_handles,
        drop_gaussians,
        identity_handles,
        render_pixel_masks,
    )
    from reifyscene.ingest.streams import RunStreams

    stream = RunStreams.from_run(run).only()
    placed = render_pixel_masks(run, stream_id=stream.id)
    gt_by_frame = _gt_masks(stream)
    matched, best, _ = match_objects(gt_by_frame, placed.by_frame, match_iou=args.match_iou)

    def load(sub: str) -> dict:
        with np.load(run / sub / "trained_dynamic.npz", allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    use_identity = args.identity_sub.lower() != "none"
    base_bundle = load(args.base_sub)
    base_r = _Renderer.for_run(base_bundle, args.device, run, stream_id=stream.id)
    id_bundle = load(args.identity_sub) if use_identity else None
    id_r = (_Renderer.for_run(id_bundle, args.device, run, stream_id=stream.id)
            if use_identity else base_r)
    # compute_renderable_handles resolves its checkpoint under <run>/render/
    ckpt = "trained_dynamic.npz" if args.base_sub == "render" else \
        f"../{args.base_sub}/trained_dynamic.npz"
    base_handles = compute_renderable_handles(run, device="cpu", checkpoint=ckpt).entities
    id_handles = ({tau: identity_handles(id_bundle, threshold=tau) for tau in args.tau}
                  if use_identity else {})
    fs_handles = _flashsplat_handles(run, base_r, args.slackness, args.device)
    out_dir = run / (args.identity_sub if use_identity else args.base_sub) / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("obj*"):
        stale.unlink()

    objects = []
    for gid in sorted({g for gts in gt_by_frame.values() for g in gts}):
        row: dict = {"gt_id": gid, "best_node": best[gid][0], "best_iou": best[gid][1]}
        if gid not in matched:
            row["status"] = "not_proposed"
            objects.append(row)
            continue
        node, iou = matched[gid]
        row.update(status="matched", node_id=node, label=placed.labels.get(node, ""),
                   match_iou=iou)
        present = [f for f in stream.frames if gid in gt_by_frame.get(f, {})
                   and f in base_r.time_of and f in id_r.time_of]
        step = max(1, len(present) // max(1, args.frames))
        frames = present[::step][: args.frames]
        row["frames"] = frames

        methods: dict[str, tuple[dict, _Renderer, np.ndarray, np.ndarray]] = {}
        h = base_handles.get(node)
        base_idx = np.asarray([] if h is None else list(h.gaussian_indices), dtype=int)
        none = np.array([], dtype=int)
        methods["baseline"] = (drop_gaussians(base_bundle, base_idx), base_r, base_idx, none)
        for tau in (args.tau if use_identity else []):
            fg_i, bg_i = id_handles[tau].get(node, (none, none))
            methods[f"identity@{tau:g}"] = (
                drop_gaussians(id_bundle, fg_i, bg_indices=bg_i), id_r, fg_i, bg_i)
        for s in args.slackness:
            fg_i, bg_i = fs_handles[s].get(node, (none, none))
            methods[f"flashsplat@{s:g}"] = (
                drop_gaussians(base_bundle, fg_i, bg_indices=bg_i), base_r, fg_i, bg_i)
        # the first slackness again, with floaters (isolated, oversized) filtered out
        fg_i, bg_i = fs_handles[args.slackness[0]].get(node, (none, none))
        fg_f, bg_f, fstats = filter_members(base_r.scene, fg_i, bg_i)
        row["filter_dropped"] = fstats
        methods[f"flashsplat@{args.slackness[0]:g}+filter"] = (
            drop_gaussians(base_bundle, fg_f, bg_indices=bg_f), base_r, fg_f, bg_f)

        row["methods"] = {}
        changed_maps: dict[str, dict[int, np.ndarray]] = {}
        afters: dict[str, dict[int, np.ndarray]] = {}
        tooks: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for name, (edited, rend, fg_i, bg_i) in methods.items():
            after = afters[name] = rend.after(edited, frames)
            took = tooks[name] = rend.taken(fg_i, bg_i, frames)
            share = rend.share(fg_i, bg_i, frames)
            per_frame, changed_maps[name] = [], {}
            for f in frames:
                ch = _changed(rend.before(f), after[f], args.changed_threshold)
                changed_maps[name][f] = ch
                gt = _fit(gt_by_frame[f][gid], ch.shape)
                others = np.zeros_like(gt)
                for og, om in gt_by_frame[f].items():
                    if og != gid:
                        others |= _fit(om, ch.shape)
                ours = _fit(placed.by_frame.get(f, {}).get(node, np.zeros_like(gt)), ch.shape)
                alpha, alone = share[f], took[f][1]
                per_frame.append({"frame": f,
                                  "vs_gt": {**_score(ch, gt, others), **_taken_score(alpha, gt),
                                            **_taken_score(alone, gt, prefix="alone")},
                                  "vs_ours": {**_score(ch, ours), **_taken_score(alpha, ours),
                                              **_taken_score(alone, ours, prefix="alone")}})
            keys = ("entity_removed", "rest_intact", "halo_px", "halo_ratio", "others_damaged",
                    "taken_recall", "taken_precision", "alone_recall", "alone_precision")
            agg = {ref: {k: _mean(per_frame, ref, k) for k in keys}
                   for ref in ("vs_gt", "vs_ours")} if per_frame else {}
            row["methods"][name] = {"removed_fg": len(fg_i), "removed_bg": len(bg_i),
                                    "frames": per_frame, **agg}
        from PIL import Image

        fs = f"flashsplat@{args.slackness[0]:g}"
        extra = [("fsp", f"flashsplat@{x:g}") for x in args.slackness if x > 0][:1] + [
            ("fsf", f"{fs}+filter")]
        names = ["before", "old-after", "old-took", "fs-after", "fs-took"] + [
            f"{k}-{kind}" for k, _ in extra for kind in ("after", "took")] + (
            ["id-before", "id-after", "id-took"] if use_identity else [])
        row["strip_panels"] = names
        for f in frames:
            panels = [base_r.before(f), afters["baseline"][f], tooks["baseline"][f][0],
                      afters[fs][f], tooks[fs][f][0]]
            for _, key in extra:
                panels += [afters[key][f], tooks[key][f][0]]
            if use_identity:
                panels += [id_r.before(f), afters["identity@0.5"][f],
                           tooks["identity@0.5"][f][0]]
            strip = np.concatenate(panels, axis=1)
            Image.fromarray(strip).save(out_dir / f"obj{gid}_{f:05d}_strip.jpg", quality=90)
            gt = _fit(gt_by_frame[f][gid], changed_maps["baseline"][f].shape)
            Image.fromarray(_outline(gt), "RGBA").save(out_dir / f"obj{gid}_{f:05d}_gt.png")
        objects.append(row)

    result = {"run": run.name, "base_sub": args.base_sub,
              "identity_sub": args.identity_sub if use_identity else None, "tau": args.tau,
              "slackness": args.slackness,
              "match_iou": args.match_iou, "changed_threshold": args.changed_threshold,
              "objects": objects}
    (out_dir / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--identity-sub", default="render_identity_3d",
                    help="identity-trained checkpoint dir, or 'none' to leave identity out")
    ap.add_argument("--base-sub", default="render",
                    help="checkpoint the old vote and FlashSplat run on")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--slackness", type=float, nargs="+", default=[0.0, -0.8, 0.4],
                    help="FlashSplat slackness values; the first fills the image strips")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--match-iou", type=float, default=0.25,
                    help="pooled-over-clip IoU a node needs to count as the object")
    ap.add_argument("--changed-threshold", type=int, default=12,
                    help="per-pixel L1 over 0-255 channels above which a pixel counts as changed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if 0.5 not in args.tau:
        args.tau = sorted(set(args.tau) | {0.5})

    fs = f"flashsplat@{args.slackness[0]:g}"
    cols = [("old vote", "baseline")] + (
        [("identity", "identity@0.5")] if args.identity_sub.lower() != "none" else []) + [
        (fs, fs), (fs + "+filter", fs + "+filter")]
    print(f"{'run':<22} {'obj':>3} {'label':<11}  " + "  ".join(
        f"{name + ' took/alone-P/vis-P/intact':>34}" for name, _ in cols))
    for run in args.run_dirs:
        res = gate_run(run.resolve(), args)
        for o in res["objects"]:
            if o["status"] != "matched":
                print(f"{res['run']:<22} {o['gt_id']:>3} {'-':<11}  not proposed "
                      f"(best node {o['best_node']}, IoU {o['best_iou']:.2f})")
                continue

            def p(v):
                return "  -  " if v is None else f"{100 * v:5.1f}"

            cells = []
            for _, key in cols:
                m = o["methods"][key]["vs_gt"]
                cells.append(f"{p(m['taken_recall'])} / {p(m['alone_precision'])} / "
                             f"{p(m['taken_precision'])} / {p(m['rest_intact'])}")
            print(f"{res['run']:<22} {o['gt_id']:>3} {o['label'][:11]:<11}  " + "    ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

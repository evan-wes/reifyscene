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
  (The strip's "took" panels render the removed Gaussians alone, which shows parts hidden
  behind other things too; they are for looking, the metric uses the visible share.)
- **Is the rest intact?** *rest intact* is the fraction of pixels outside the GT mask that did
  not change; *halo* counts those that did. *entity removed* (changed pixels inside the mask)
  is kept for comparison with the first gate, which reported it.

Writes ``<run>/<identity-sub>/gate/index.json`` and, at tau=0.5, per object-frame one strip
``obj<id>_<frame>_strip.jpg`` (before | baseline after | identity after | baseline took |
identity took) and ``obj<id>_<frame>_gt.png``, the GT outline on transparency.

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
DAVIS = REPO.parent / "data/Davis2017/DAVIS-2017-trainval-480p/DAVIS"


def _gt_masks(ann_dir: Path, frame: int, transform) -> dict[int, np.ndarray]:
    from PIL import Image

    path = ann_dir / f"{frame:05d}.png"
    if not path.is_file():
        return {}
    arr = np.asarray(Image.open(path))
    return {int(i): transform.apply_to_image((arr == i).astype(np.uint8), kind="mask") > 0
            for i in np.unique(arr) if int(i) != 0}


def match_objects(gt_by_frame, node_by_frame, *, match_iou: float):
    """Greedy one-to-one GT object -> node on IoU pooled over frames."""
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


def _score(changed: np.ndarray, mask: np.ndarray) -> dict:
    inside = int((changed & mask).sum())
    outside = int((changed & ~mask).sum())
    return {
        "entity_removed": round(inside / max(1, int(mask.sum())), 4),
        "rest_intact": round(1.0 - outside / max(1, int((~mask).sum())), 6),
        "halo_px": outside,
    }


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if mask.shape == shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                      .resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 127


class _Renderer:
    """Renders a bundle (and edits of it) at training cameras, caching the unedited view."""

    def __init__(self, bundle: dict, device: str) -> None:
        from reifyscene.render.eval.checkpoint_views import replay_size
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
        from reifyscene.render.gaussian4d.rasterizer import build_rasterizer

        self.bundle, self.device = bundle, device
        self.rasterizer = build_rasterizer({"device": device})
        self.scene, cams = bundle_to_scene(bundle, device=device)
        self.cams = {int(c.time_id): c for c in cams.get("train") or []}
        self.size = replay_size(np.asarray(bundle["train_Ks"])[0])
        names = [str(n) for n in np.asarray(bundle["train_frame_names"])]
        self.time_of = {int(n.rsplit(":", 1)[-1]): int(t)
                        for n, t in zip(names, np.asarray(bundle["train_time_ids"]), strict=True)}
        self._before: dict[int, np.ndarray] = {}

    def _render(self, scene, frame: int) -> np.ndarray:
        import torch

        from reifyscene.render.eval.checkpoint_views import _render, _to_u8

        t = self.time_of[frame]
        rc = self.cams[t].render_camera(*self.size)
        with torch.no_grad():
            return _to_u8(_render(self.rasterizer, scene, rc, t))

    def render_alpha(self, scene, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8, alpha float) of ``scene`` at the training camera for ``frame``."""
        import torch

        from reifyscene.render.eval.checkpoint_views import _to_u8

        t = self.time_of[frame]
        rc = self.cams[t].render_camera(*self.size)
        with torch.no_grad():
            c = scene.cloud_at(t)
            out = self.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, rc)
        return _to_u8(out.rgb.clamp(0, 1)), out.alpha.clamp(0, 1).cpu().numpy()

    def taken(self, fg_idx, bg_idx, frames: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Render ONLY the given Gaussians: what a removal took, on black."""
        from reifyscene.graph.renderable_handle import drop_gaussians
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        h, w = self.size[1], self.size[0]
        blank = {f: (np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.float32)) for f in frames}
        if len(fg_idx) + len(bg_idx) == 0:
            return blank
        n_fg = int(np.asarray(self.bundle["fg_means"]).shape[0])
        n_bg = int(np.asarray(self.bundle["bg_means"]).shape[0]) if "bg_means" in self.bundle else 0
        only = drop_gaussians(self.bundle, np.setdiff1d(np.arange(n_fg), fg_idx),
                              bg_indices=np.setdiff1d(np.arange(n_bg), bg_idx))
        scene, _ = bundle_to_scene(only, device=self.device)
        return {f: self.render_alpha(scene, f) for f in frames}

    def share(self, fg_idx, bg_idx, frames: list[int]) -> dict[int, np.ndarray]:
        """Per pixel, the composited weight the given Gaussians contribute to the full render."""
        import torch

        n_fg = self.scene.fg.num_gaussians
        ind = torch.zeros((self.scene.num_gaussians, 1), device=self.device)
        ind[torch.as_tensor(np.asarray(fg_idx, dtype=np.int64), device=self.device)] = 1.0
        ind[n_fg + torch.as_tensor(np.asarray(bg_idx, dtype=np.int64), device=self.device)] = 1.0
        out: dict[int, np.ndarray] = {}
        for f in frames:
            t = self.time_of[f]
            rc = self.cams[t].render_camera(*self.size)
            with torch.no_grad():
                c = self.scene.cloud_at(t)
                r = self.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors,
                                              rc, features=ind)
            out[f] = r.features[..., 0].clamp(0, 1).cpu().numpy()
        return out

    def before(self, frame: int) -> np.ndarray:
        if frame not in self._before:
            self._before[frame] = self._render(self.scene, frame)
        return self._before[frame]

    def after(self, edited: dict, frames: list[int]) -> dict[int, np.ndarray]:
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        scene, _ = bundle_to_scene(edited, device=self.device)
        return {f: self._render(scene, f) for f in frames}


def _changed(a: np.ndarray, b: np.ndarray, thr: int) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(-1) > thr


def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """RGBA image: the mask's boundary in red, transparent elsewhere."""
    inner = mask.copy()
    for _ in range(width):
        e = inner.copy()
        e[1:] &= inner[:-1]
        e[:-1] &= inner[1:]
        e[:, 1:] &= inner[:, :-1]
        e[:, :-1] &= inner[:, 1:]
        inner = e
    edge = mask & ~inner
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[edge] = (255, 64, 64, 255)
    return rgba


def _mean(rows: list[dict], ref: str, key: str) -> float | None:
    vals = [r[ref][key] for r in rows if r[ref][key] is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def _taken_score(alpha: np.ndarray, gt: np.ndarray) -> dict:
    took = alpha > 0.5
    hit = int((took & gt).sum())
    return {"taken_recall": round(hit / max(1, int(gt.sum())), 4),
            "taken_precision": round(hit / int(took.sum()), 4) if took.any() else None,
            "taken_px": int(took.sum())}


def gate_run(run: Path, args) -> dict:
    from reifyscene.graph.renderable_handle import (
        compute_renderable_handles,
        drop_gaussians,
        identity_handles,
        render_pixel_masks,
    )

    seq = run.name.removeprefix("loop_")
    ann_dir = args.davis / "Annotations" / "480p" / seq
    placed = render_pixel_masks(run)
    ingest = json.loads((run / "ingest" / "artifact.json").read_text())
    n_frames = int(next(s.get("value", s) for s in ingest["streams"])["num_frames"])
    gt_by_frame = {f: _gt_masks(ann_dir, f, placed.transform) for f in range(n_frames)}
    matched, best, _ = match_objects(gt_by_frame, placed.by_frame, match_iou=args.match_iou)

    def load(sub: str) -> dict:
        with np.load(run / sub / "trained_dynamic.npz", allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    base_bundle, id_bundle = load("render"), load(args.identity_sub)
    base_r, id_r = _Renderer(base_bundle, args.device), _Renderer(id_bundle, args.device)
    base_handles = compute_renderable_handles(run, device="cpu").entities
    id_handles = {tau: identity_handles(id_bundle, threshold=tau) for tau in args.tau}
    out_dir = run / args.identity_sub / "gate"
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
        present = [f for f in range(n_frames) if gid in gt_by_frame[f]
                   and f in base_r.time_of and f in id_r.time_of]
        step = max(1, len(present) // max(1, args.frames))
        frames = present[::step][: args.frames]
        row["frames"] = frames

        methods: dict[str, tuple[dict, _Renderer, np.ndarray, np.ndarray]] = {}
        h = base_handles.get(node)
        base_idx = np.asarray([] if h is None else list(h.gaussian_indices), dtype=int)
        none = np.array([], dtype=int)
        methods["baseline"] = (drop_gaussians(base_bundle, base_idx), base_r, base_idx, none)
        for tau in args.tau:
            fg_i, bg_i = id_handles[tau].get(node, (none, none))
            methods[f"identity@{tau:g}"] = (
                drop_gaussians(id_bundle, fg_i, bg_indices=bg_i), id_r, fg_i, bg_i)

        row["methods"] = {}
        changed_maps: dict[str, dict[int, np.ndarray]] = {}
        afters: dict[str, dict[int, np.ndarray]] = {}
        tooks: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for name, (edited, rend, fg_i, bg_i) in methods.items():
            after = afters[name] = rend.after(edited, frames)
            tooks[name] = rend.taken(fg_i, bg_i, frames)
            share = rend.share(fg_i, bg_i, frames)
            per_frame, changed_maps[name] = [], {}
            for f in frames:
                ch = _changed(rend.before(f), after[f], args.changed_threshold)
                changed_maps[name][f] = ch
                gt = _fit(gt_by_frame[f][gid], ch.shape)
                ours = _fit(placed.by_frame.get(f, {}).get(node, np.zeros_like(gt)), ch.shape)
                alpha = share[f]
                per_frame.append({"frame": f,
                                  "vs_gt": {**_score(ch, gt), **_taken_score(alpha, gt)},
                                  "vs_ours": {**_score(ch, ours), **_taken_score(alpha, ours)}})
            keys = ("entity_removed", "rest_intact", "halo_px", "taken_recall", "taken_precision")
            agg = {ref: {k: _mean(per_frame, ref, k) for k in keys}
                   for ref in ("vs_gt", "vs_ours")} if per_frame else {}
            row["methods"][name] = {"removed_fg": len(fg_i), "removed_bg": len(bg_i),
                                    "frames": per_frame, **agg}
        from PIL import Image

        for f in frames:
            strip = np.concatenate([id_r.before(f), afters["baseline"][f],
                                    afters["identity@0.5"][f], tooks["baseline"][f][0],
                                    tooks["identity@0.5"][f][0]], axis=1)
            Image.fromarray(strip).save(out_dir / f"obj{gid}_{f:05d}_strip.jpg", quality=90)
            gt = _fit(gt_by_frame[f][gid], changed_maps["baseline"][f].shape)
            Image.fromarray(_outline(gt), "RGBA").save(out_dir / f"obj{gid}_{f:05d}_gt.png")
        objects.append(row)

    result = {"run": run.name, "identity_sub": args.identity_sub, "tau": args.tau,
              "match_iou": args.match_iou, "changed_threshold": args.changed_threshold,
              "objects": objects}
    (out_dir / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--identity-sub", default="render_identity_3d")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--match-iou", type=float, default=0.25,
                    help="pooled-over-clip IoU a node needs to count as the object")
    ap.add_argument("--changed-threshold", type=int, default=12,
                    help="per-pixel L1 over 0-255 channels above which a pixel counts as changed")
    ap.add_argument("--davis", type=Path, default=DAVIS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if 0.5 not in args.tau:
        args.tau = sorted(set(args.tau) | {0.5})

    print(f"{'run':<22} {'obj':>3} {'label':<11} {'iou':>5}  "
          f"{'baseline took rec/prec  intact':>31}  {'identity took rec/prec  intact':>31}  fg+bg")
    for run in args.run_dirs:
        res = gate_run(run.resolve(), args)
        for o in res["objects"]:
            if o["status"] != "matched":
                print(f"{res['run']:<22} {o['gt_id']:>3} {'-':<11} {o['best_iou']:>5.2f}  "
                      f"not proposed (best node {o['best_node']})")
                continue
            b, i = o["methods"]["baseline"]["vs_gt"], o["methods"]["identity@0.5"]["vs_gt"]
            n = o["methods"]["identity@0.5"]

            def p(v):
                return "   -  " if v is None else f"{v:6.1%}"

            print(f"{res['run']:<22} {o['gt_id']:>3} {o['label'][:11]:<11} {o['match_iou']:>5.2f}  "
                  f"{p(b['taken_recall'])} / {p(b['taken_precision'])}  {b['rest_intact']:>8.2%}   "
                  f"{p(i['taken_recall'])} / {p(i['taken_precision'])}  {i['rest_intact']:>8.2%}  "
                  f"{n['removed_fg']}+{n['removed_bg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

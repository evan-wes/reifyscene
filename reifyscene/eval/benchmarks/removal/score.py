"""A run's removals against the owner's four criteria (``docs/design/2026-10-17-removal-by-
ownership.md`` §1, §5).

Per entity, what does not depend on the removal method -- measured once:

- **coverage** (T4): the share of the entity's own pixels, over its frames, that no other frame
  saw the background behind (:mod:`.seen`); its complement ranks clips by how much there is to
  reveal;
- **the geometry floor**: PSNR of the unedited render (and, as a diagnostic, of the photo)
  against the projected background on the background pixels around the entity;
- its **volume** on each frame (:mod:`.volume`) and the Gaussians inside it.

Per entity and method (``flashsplat``: FlashSplat with the floater filter, as the app removes
today; ``flashsplat-unfiltered``; ``identity``: the learned identity, where the checkpoint has
it):

- **T1**: share of judged views without a trace (:mod:`.trace`), on training views and on
  views orbited ``orbit_yaws`` about the scene;
- **T2**: opacity left in the volume after removal, as a share of before (all frames pooled);
- **T3**: PSNR of the after-render against the projected background over the entity's seen
  pixels, and its gap below the floor;
- **residual**: the share of the before-render inside the entity's pixels drawn by Gaussians
  the removal kept (a diagnostic of where a trace comes from).

The views are the training frames and orbits of them, so nothing here needs held-out data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from reifyscene.eval.benchmarks.removal.seen import BackgroundProjector, SquaredError, band
from reifyscene.eval.benchmarks.removal.trace import TraceCount, entity_region, has_trace
from reifyscene.eval.benchmarks.removal.views import RunViews
from reifyscene.eval.benchmarks.removal.volume import entity_volume, person_body

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import Checkpoint

# The owner's pass marks (design doc §5, decided 2026-10-17).
PASS = {"t1_clean": 0.95, "t2_left": 0.05, "t3_gap_db": 1.0}
METHODS = ("flashsplat", "flashsplat-unfiltered", "identity")


@dataclass
class ScoreConfig:
    min_px: int = 200              # a frame counts for an entity with this many own pixels
    min_frames: int = 5            # an entity on fewer frames is not scored
    band_px: int = 16              # the floor's band of background around the entity ...
    band_gap_px: int = 4           # ... starting this far from any mask
    erode_px: int = 2              # background this close to a mask is not projected
    tol: float = 0.05              # relative depth tolerance of the projection
    orbit_yaws: tuple[float, ...] = (-20.0, 20.0)
    orbit_frames: int = 8          # frames orbited per yaw
    stills: int = 3                # before/after stills saved per entity and method


def select(editor, node_id: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    """``(fg, bg)`` indices ``method`` removes for ``node_id``."""
    if method == "flashsplat":
        fg, bg, _ = editor.select(node_id)
    elif method == "flashsplat-unfiltered":
        fg, bg, _ = editor.select(node_id, filtered=False)
    elif method == "identity":
        from reifyscene.graph.renderable_handle import identity_handles

        fg, bg = identity_handles(editor.bundle).get(node_id, (np.array([], int),) * 2)
    else:
        raise ValueError(f"unknown removal method {method!r}; methods: {', '.join(METHODS)}")
    return np.asarray(fg, np.int64), np.asarray(bg, np.int64)


def available_methods(bundle) -> list[str]:
    return [m for m in METHODS if m != "identity" or "identity_classifier_w" in bundle]


class _Scene:
    """The checkpoint's Gaussian centres per frame and their opacities, on the host."""

    def __init__(self, renderer) -> None:
        import torch

        self.r = renderer
        self.n_fg = renderer.scene.fg.num_gaussians
        self.n = renderer.scene.num_gaussians
        with torch.no_grad():
            self.opacities = renderer.scene.cloud_at(0).opacities.cpu().numpy().astype(np.float64)
        self._pos: dict[int, np.ndarray] = {}

    def positions(self, frame: int) -> np.ndarray:
        import torch

        if frame not in self._pos:
            with torch.no_grad():
                c = self.r.scene.cloud_at(self.r.time_of[frame])
                self._pos[frame] = c.means.cpu().numpy()
        return self._pos[frame]

    def combined(self, fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, bool)
        out[fg] = True
        out[self.n_fg + bg] = True
        return out

    def split(self, combined_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(combined_idx, np.int64)
        return idx[idx < self.n_fg], idx[idx >= self.n_fg] - self.n_fg


def _share(renderer, scene, frame: int, camera, members: np.ndarray) -> np.ndarray:
    """Per pixel, the composited weight ``members`` (bool over the combined cloud) contribute
    to ``scene``'s render at ``frame``'s moment from ``camera``."""
    import torch

    ind = torch.as_tensor(members.astype(np.float32)[:, None], device=renderer.device)
    with torch.no_grad():
        c = scene.cloud_at(renderer.time_of[frame])
        out = renderer.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors,
                                            camera, features=ind)
    return out.features[..., 0].clamp(0, 1).cpu().numpy()


def _rgb(renderer, scene, frame: int, camera) -> np.ndarray:
    return renderer.render_view(scene, frame, camera)


def _stills(path, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    from PIL import Image

    rows = [np.concatenate([b, np.full((b.shape[0], 4, 3), 255, np.uint8), a], axis=1)
            for b, a in pairs]
    gap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.concatenate([sheet, gap, row], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=90)


def _evenly(items: list[int], k: int) -> list[int]:
    if len(items) <= k:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def verdict(row: dict[str, Any]) -> dict[str, bool | None]:
    """Each pass mark for one method's row; None where the test could not be measured."""
    t1 = [row["t1_train"]["clean"], row["t1_orbit"]["clean"]]
    t1 = [v for v in t1 if v is not None]
    gap = row.get("t3_gap_db")
    out = {
        "t1": all(v >= PASS["t1_clean"] for v in t1) if t1 else None,
        "t2": None if row["t2_left"] is None else row["t2_left"] <= PASS["t2_left"],
        "t3": None if gap is None else gap <= PASS["t3_gap_db"],
    }
    measured = [v for v in out.values() if v is not None]
    out["all"] = all(measured) if measured else None
    return out


def score_run(checkpoint: Checkpoint, *, methods: list[str] | None = None,
              nodes: list[str] | None = None, cfg: ScoreConfig | None = None,
              device: str = "cuda", stills_dir=None, log=print) -> dict[str, Any]:
    """Every entity of ``checkpoint``'s run (or ``nodes``), every method, every test."""
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.eval.benchmarks.edit_damage.silhouettes import InstanceSegmenter
    from reifyscene.eval.benchmarks.removal.trace import AFTER_SCORE
    from reifyscene.graph.node_ids import tracklet_number
    from reifyscene.graph.renderable_handle import drop_gaussians
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    cfg = cfg or ScoreConfig()
    run = checkpoint.run
    ed = RemoveEditor(checkpoint, device=device)
    r = ed.renderer
    methods = list(methods or available_methods(ed.bundle))
    views = RunViews.build(run, r)
    cloud = _Scene(r)
    projector = BackgroundProjector(views, device=device, erode_px=cfg.erode_px, tol=cfg.tol)
    projected: dict[int, Any] = {}
    segmenter = InstanceSegmenter(device=device, score_threshold=AFTER_SCORE)
    found: dict[tuple[str, str], list] = {}            # (view, category) -> before silhouettes
    graph_nodes = {n.node_id: n for n in run.artifact("graph").nodes}

    def before_found(view: str, category: str, image) -> list:
        if (view, category) not in found:
            found[(view, category)] = segmenter.find(image(), category)
        return found[(view, category)]

    def project(f: int):
        if f not in projected:
            projected[f] = projector.project(f)
        return projected[f]

    entities = sorted(views.class_of) if nodes is None else list(nodes)
    rows: list[dict[str, Any]] = []
    for node_id in entities:
        node = graph_nodes.get(node_id)
        label = views.labels.get(node_id, "")
        row: dict[str, Any] = {"node_id": node_id, "label": label,
                               "node_type": node.node_type if node else None}
        rows.append(row)
        frames = views.present(node_id, min_px=cfg.min_px)
        row["frames"] = len(frames)
        if len(frames) < cfg.min_frames:
            row["status"] = "too_few_frames"
            continue
        row["status"] = "scored"
        log(f"  {node_id} ({label}): {len(frames)} frames")

        # -- what the method does not change ---------------------------------------------------
        px = seen_px = 0
        floor, floor_photo = SquaredError(), SquaredError()
        for f in frames:
            m, pr = views.mask(node_id, f), project(f)
            px += int(m.sum())
            seen_px += int((m & pr.seen).sum())
            ring = band(views, node_id, f, cfg.band_px, cfg.band_gap_px) & pr.seen
            floor.add(r.before(f), pr.colour, ring)
            floor_photo.add(views.photos[f], pr.colour, ring)
        row["coverage"] = {"pixels": px, "seen_share": round(seen_px / px, 4),
                           "unseen_share": round(1.0 - seen_px / px, 4),
                           "floor_db": floor.psnr(), "floor_photo_db": floor_photo.psnr()}

        body = person_body(run, r, views, node_id) if row["node_type"] == "human" else None
        volume = entity_volume(views, node_id, body=body)
        inside = {f: np.flatnonzero(volume.inside(cloud.positions(f), f))
                  for f in frames if volume.kind(f)}
        kinds = [volume.kind(f) for f in inside]
        row["volume"] = {"frames": len(inside), "body_frames": kinds.count("body"),
                         "slab_frames": kinds.count("slab")}

        orbit = [(f, y) for y in cfg.orbit_yaws for f in _evenly(frames, cfg.orbit_frames)]
        regions: dict[str, np.ndarray | None] = {}
        for f in frames:
            regions[f"train:{f}"] = entity_region(
                before_found(f"train:{f}", label, lambda f=f: r.before(f)),
                views.mask(node_id, f))
        for f, yaw in orbit:
            cam = r.orbit_camera(f, yaw=yaw)
            members = np.zeros(cloud.n, bool)
            if f in inside:
                members[inside[f]] = True
            hint = _share(r, r.scene, f, cam, members) > 0.5
            regions[f"orbit:{f}:{yaw:g}"] = entity_region(
                before_found(f"orbit:{f}:{yaw:g}", label,
                             lambda f=f, cam=cam: _rgb(r, r.scene, f, cam)), hint)

        # -- each method -------------------------------------------------------------------------
        row["methods"] = {}
        for method in methods:
            fg, bg = select(ed, node_id, method)
            res: dict[str, Any] = {"removed_fg": int(len(fg)), "removed_bg": int(len(bg))}
            row["methods"][method] = res
            removed = cloud.combined(fg, bg)
            after_scene, _ = bundle_to_scene(drop_gaussians(ed.bundle, fg, bg_indices=bg),
                                             device=device)
            t1 = {"train": TraceCount(), "orbit": TraceCount()}
            t3 = SquaredError()
            before_mass = after_mass = 0.0
            residual = []
            share = r.share(fg, bg, frames)
            after_imgs: dict[int, np.ndarray] = {}
            for f in frames:
                after_imgs[f] = r.render_alpha(after_scene, f)[0]
                m, pr = views.mask(node_id, f), project(f)
                t3.add(after_imgs[f], pr.colour, m & pr.seen)
                residual.append(float(1.0 - share[f][m].mean()))
                if f in inside:
                    idx = inside[f]
                    before_mass += float(cloud.opacities[idx].sum())
                    after_mass += float(cloud.opacities[idx][~removed[idx]].sum())
            for f in frames:
                view, region = f"train:{f}", regions[f"train:{f}"]
                if region is None:
                    t1["train"].unjudged += 1
                    continue
                t1["train"].judged.append(view)
                if has_trace(segmenter.find(after_imgs[f], label), region):
                    t1["train"].traces.append(view)
            for f, yaw in orbit:
                view = f"orbit:{f}:{yaw:g}"
                if regions[view] is None:
                    t1["orbit"].unjudged += 1
                    continue
                t1["orbit"].judged.append(view)
                after = _rgb(r, after_scene, f, r.orbit_camera(f, yaw=yaw))
                if has_trace(segmenter.find(after, label), regions[view]):
                    t1["orbit"].traces.append(view)
            t3_db = t3.psnr()
            res.update({
                "t1_train": t1["train"].to_dict(), "t1_orbit": t1["orbit"].to_dict(),
                "t2_left": round(after_mass / before_mass, 4) if before_mass > 0 else None,
                "t3_db": t3_db,
                "t3_gap_db": (round(row["coverage"]["floor_db"] - t3_db, 3)
                              if t3_db is not None and row["coverage"]["floor_db"] is not None
                              else None),
                "residual": round(float(np.mean(residual)), 4),
            })
            res["pass"] = verdict(res)
            log(f"    {method:<22} T1 {res['t1_train']['clean']}/{res['t1_orbit']['clean']}  "
                f"T2 {res['t2_left']}  T3 {t3_db} (gap {res['t3_gap_db']})  "
                f"residual {res['residual']}  pass {res['pass']['all']}")
            if stills_dir is not None and cfg.stills:
                pick = _evenly(frames, cfg.stills)
                _stills(stills_dir / f"{tracklet_number(node_id)}_{method}.jpg",
                        [(r.before(f), after_imgs[f]) for f in pick])
            del after_scene
    return {"run": run.name, "checkpoint": checkpoint.name, "config": asdict(cfg),
            "methods": methods, "entities": rows}


def ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scored entities of many runs, most background seen first: the clip ranking."""
    out = []
    for run in rows:
        for e in run["entities"]:
            if e.get("status") != "scored":
                continue
            out.append({"run": run["run"], "node_id": e["node_id"], "label": e["label"],
                        "frames": e["frames"], **e["coverage"]})
    return sorted(out, key=lambda x: -x["seen_share"])

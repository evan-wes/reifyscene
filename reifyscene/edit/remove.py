"""``remove(entity)`` on a finished run: select its Gaussians, drop them, render the result.

Selection is the removal path the identity scope settled on (§4.7-4.9 of
``docs/design/2026-09-29-per-gaussian-identity-scope.md``): FlashSplat's evidence-majority
assignment against the run's id maps, then the floater filter. The edit is a checkpoint transform
(the selected Gaussians dropped, the scene rebuilt), so what renders afterwards is a real scene.

An edit is a *view* on a run. Its results go to the run's output dir for the edit
(``RunOutputs.output_dir(<edit id>)``) and nothing in the pipeline's own stage directories
changes. The evidence matrix -- one backward per training frame -- is computed once per
checkpoint and cached (``output_dir("cache", <checkpoint>)``); after that an edit is an
assignment, a filter and a few renders.

What an edit reports, beside the images: how many Gaussians it took from each field, what share
of the scene that is (the renderable handle's *assigned fraction*), what the floater filter
dropped, and ``unassignable`` when no Gaussian's evidence favours the entity at all.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import Checkpoint

REMOVED_FILE = "removed.npz"     # the removed Gaussians, so the edit can be re-rendered


def edit_id(op: str, node_id: str) -> str:
    from reifyscene.graph.node_ids import tracklet_number

    return f"{op}-{tracklet_number(node_id)}-{time.strftime('%Y%m%d-%H%M%S')}"


class RemoveEditor:
    """Holds one run's checkpoint on the GPU and executes removals against it."""

    def __init__(self, checkpoint: Checkpoint, device: str = "cuda") -> None:
        """``checkpoint``: the run's trained scene to edit (``RunOutputs.checkpoint()``)."""
        from reifyscene.edit.renderer import CheckpointRenderer

        self.checkpoint = checkpoint
        self.run = checkpoint.run
        self.device = device
        self.use_layer = True     # the handles stage turns this off: it is what writes the layer
        self.bundle = checkpoint.load()
        self.renderer = CheckpointRenderer.for_run(self.bundle, device, self.run)
        self._counts = None
        self._node_ids: list[str | None] = []
        self._edited: dict[tuple[str, str], Any] = {}   # (edit dir, show) -> scene; last two

    # -- evidence -----------------------------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self.run.output_dir("cache", self.checkpoint.name)

    def evidence(self, *, refresh: bool = False):
        """(N, C) per-Gaussian blending weight per id-map class, cached per checkpoint.

        ``refresh`` re-exports the id maps from the graph and recomputes, ignoring both caches
        (the pipeline's ``handles`` stage: the graph may have changed since the cache).
        """
        import torch

        from reifyscene.graph.assignment import evidence_matrix, id_map_views
        from reifyscene.graph.id_maps import export_id_maps, has_id_maps, load_id_maps

        if self._counts is not None and not refresh:
            return self._counts
        cpath, npath = self.cache_dir / "evidence.npy", self.cache_dir / "classes.json"
        if cpath.is_file() and npath.is_file() and not refresh:
            counts = np.load(cpath)
            if counts.shape[0] == self.renderer.scene.num_gaussians:
                self._counts = torch.as_tensor(counts, device=self.device)
                self._node_ids = json.loads(npath.read_text())
                return self._counts
        if refresh or not has_id_maps(self.run):
            export_id_maps(self.run)
        manifest, maps = load_id_maps(self.run)
        self._node_ids = [c["node_id"]
                          for c in sorted(manifest["classes"], key=lambda c: c["index"])]
        r = self.renderer
        cams = {f: (r.cams[t].render_camera(*r.size), t) for f, t in r.time_of.items()}
        self._counts = evidence_matrix(r.scene, r.rasterizer, id_map_views(maps, cams, self.device),
                                       len(self._node_ids))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cpath, self._counts.cpu().numpy())
        npath.write_text(json.dumps(self._node_ids))
        return self._counts

    # -- selection ----------------------------------------------------------------------------

    def select(
        self, node_id: str, *, slackness: float = 0.0, filtered: bool = True
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """``(fg_idx, bg_idx, stats)`` for one entity.

        The default selection (slackness 0, filter on) comes from the graph's ``gaussians``
        layer when that was computed against this checkpoint; otherwise, and for any other
        parameters, it is computed here. ``stats["source"]`` says which.
        """
        from reifyscene.graph.assignment import assign, assignment_handles, filter_members
        from reifyscene.graph.handles import StaleHandles, load_handles

        if slackness == 0.0 and filtered and self.use_layer:
            try:
                fg, bg = load_handles(self.run, node_id, checkpoint=self.checkpoint)
                return fg, bg, {"selected_fg": int(len(fg)), "selected_bg": int(len(bg)),
                                "source": "graph layer"}
            except (KeyError, StaleHandles, FileNotFoundError):
                pass

        counts = self.evidence()
        if node_id not in self._node_ids:
            raise KeyError(f"{node_id} is not a class of this run's id maps")
        n_fg = self.renderer.scene.fg.num_gaussians
        handles = assignment_handles(assign(counts, slackness=slackness), n_fg, self._node_ids)
        fg, bg = handles.get(node_id, (np.array([], int), np.array([], int)))
        stats: dict[str, Any] = {"selected_fg": int(len(fg)), "selected_bg": int(len(bg)),
                                 "source": "computed"}
        if filtered and len(fg) + len(bg):
            fg, bg, dropped = filter_members(self.renderer.scene, fg, bg)
            stats["filter_dropped"] = dropped
        return fg, bg, stats

    # -- the edit -----------------------------------------------------------------------------

    def remove(
        self,
        node_id: str,
        *,
        slackness: float = 0.0,
        filtered: bool = True,
        frames: list[int] | None = None,
        out_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Remove one entity; write before / after / took renders and ``result.json``."""
        from PIL import Image

        from reifyscene.graph.renderable_handle import drop_gaussians

        t0 = time.time()
        fg, bg, stats = self.select(node_id, slackness=slackness, filtered=filtered)
        eid = edit_id("remove", node_id)
        out = Path(out_dir) if out_dir is not None else self.run.output_dir(eid)
        out.mkdir(parents=True, exist_ok=True)
        frames = self.renderer.frames if frames is None else [f for f in frames
                                                              if f in self.renderer.time_of]
        r = self.renderer
        after = r.after(drop_gaussians(self.bundle, fg, bg_indices=bg), frames)
        took = r.taken(fg, bg, frames)
        for f in frames:
            Image.fromarray(r.before(f)).save(out / f"before_{f:05d}.jpg", quality=90)
            Image.fromarray(after[f]).save(out / f"after_{f:05d}.jpg", quality=90)
            Image.fromarray(took[f][0]).save(out / f"took_{f:05d}.jpg", quality=90)
        n_total = r.scene.num_gaussians
        n_taken = int(len(fg) + len(bg))
        result = {
            "edit_id": eid, "op": "remove", "run": self.run.name, "node_id": node_id,
            "checkpoint": self.checkpoint.name, "slackness": slackness, "filtered": filtered,
            "removed_fg": int(len(fg)), "removed_bg": int(len(bg)),
            "assigned_fraction": round(n_taken / max(n_total, 1), 6),
            "renderable": "grounded" if n_taken else "unassignable",
            "frames": frames, "seconds": round(time.time() - t0, 2), **stats,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(out / REMOVED_FILE, fg=np.asarray(fg, np.int64),
                            bg=np.asarray(bg, np.int64))
        return result

    def edited_scene(self, edit_dir: Path, *, took: bool = False):
        """An earlier removal, from the indices it saved: the scene without them, or (``took``)
        only them -- what the removal took. None when it took nothing."""
        from reifyscene.graph.renderable_handle import drop_gaussians
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        key = (str(edit_dir), "took" if took else "edited")
        if key not in self._edited:
            with np.load(Path(edit_dir) / REMOVED_FILE) as z:
                fg, bg = z["fg"], z["bg"]
            if took:
                if len(fg) + len(bg) == 0:
                    return None
                n_fg = int(np.asarray(self.bundle["fg_means"]).shape[0])
                n_bg = (int(np.asarray(self.bundle["bg_means"]).shape[0])
                        if "bg_means" in self.bundle else 0)
                fg, bg = np.setdiff1d(np.arange(n_fg), fg), np.setdiff1d(np.arange(n_bg), bg)
            scene, _ = bundle_to_scene(drop_gaussians(self.bundle, fg, bg_indices=bg),
                                       device=self.device)
            if len(self._edited) >= 2:                  # two scenes: an edit's after and took
                self._edited.pop(next(iter(self._edited)))
            self._edited[key] = scene
        return self._edited[key]

    def render_view(self, frame: int, *, edit_dir: Path | None = None, show: str = "scene",
                    **camera) -> np.ndarray:
        """``frame``'s moment from an orbited camera (see ``CheckpointRenderer.orbit_camera``).

        ``show``: ``scene`` (unedited), ``edited`` (the removal in ``edit_dir`` applied) or
        ``took`` (only what it removed, on black).
        """
        r = self.renderer
        cam = r.orbit_camera(frame, **camera)
        if show == "scene":
            return r.render_view(r.scene, frame, cam)
        if edit_dir is None:
            raise ValueError(f"show={show!r} needs an edit")
        scene = self.edited_scene(edit_dir, took=show == "took")
        if scene is None:
            return np.zeros((cam.height, cam.width, 3), np.uint8)
        return r.render_view(scene, frame, cam)

    def render_before(self, frame: int) -> np.ndarray:
        return self.renderer.before(frame)

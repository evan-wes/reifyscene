"""The local demo app: browse a run's scene graph, query it, remove an entity, compare.

Design: ``docs/design/2026-09-29-demo-web-app.md``. The web layer holds no logic of its own --
every endpoint wraps :class:`reifyscene.query.scene_query.SceneQuery`,
:class:`reifyscene.edit.remove.RemoveEditor` or the run's id maps -- so the app cannot disagree
with ``reifyscene query`` / ``reifyscene edit``.

GPU work (renders, edits) is serialized through one lock and one worker thread: an edit queues,
the client polls it. Everything binds to 127.0.0.1; nothing is exposed.
"""

from __future__ import annotations

import io
import json
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import Query
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"
# relation roles on the frame: subject, object, between's third
OUTLINE_COLORS = ((255, 64, 64), (240, 170, 30), (170, 110, 240))
POWER_CAP_W = 400.0


def gpu_power_ok(cap: float = POWER_CAP_W) -> tuple[bool, str]:
    """Refuse to run GPU work above the power cap (it resets to 600 W on reboot)."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
        limit = float(out.strip().splitlines()[0])
    except Exception as exc:  # noqa: BLE001 -- no GPU visible is itself a refusal
        return False, f"cannot read the GPU power limit ({exc}); not starting"
    if limit > cap:
        return False, (f"GPU power limit is {limit:.0f} W, above {cap:.0f} W; "
                       f"run `sudo nvidia-smi -pl {cap:.0f}` first")
    return True, f"GPU power limit {limit:.0f} W"


class QueryIn(BaseModel):
    """Request models live at module level: FastAPI cannot resolve postponed annotations that
    name classes local to create_app, and silently misreads the body (422) if they do."""

    template: str
    args: dict[str, Any] = {}


class EditIn(BaseModel):
    op: str = "remove"
    entity: str
    slackness: float = 0.0
    filtered: bool = True


@dataclass
class Run:
    name: str
    path: Path
    checkpoint_sub: str
    _query: Any = None
    _editor: Any = None
    _maps: tuple | None = None
    _photos: dict[int, Path] | None = None
    _digest: str | None = None

    def checkpoint_digest(self) -> str:
        from reifyscene.graph.handles import checkpoint_digest

        if self._digest is None:
            self._digest = checkpoint_digest(self.path / self.checkpoint_sub
                                             / "trained_dynamic.npz")
        return self._digest

    def gaussians_layer(self) -> dict[str, Any] | None:
        """The graph's ``gaussians`` layer as the app can use it: whether it describes the
        checkpoint this app renders and edits (``current``), or another one."""
        from reifyscene.graph.handles import layer_info

        info = layer_info(self.query.graph)
        if info is None:
            return None
        return {"checkpoint": info["checkpoint"]["path"],
                "current": info["checkpoint"]["digest"] == self.checkpoint_digest(),
                "renderable": info["provides"]["renderable"], "counts": info["counts"],
                "method": info["method"]}

    @property
    def query(self):
        from reifyscene.query.scene_query import SceneQuery

        if self._query is None:
            self._query = SceneQuery.from_run(self.path)
        return self._query

    def photos(self) -> dict[int, Path]:
        """The scene's training photos (already in render pixels), by frame, for the run's
        one stream. The app has no stream picker yet, so a multi-stream run is refused rather
        than having one camera's frame 12 silently replace another's."""
        from reifyscene.ingest.streams import RunStreams, parse_key

        if self._photos is None:
            stream = RunStreams.from_run(self.path).only().id
            raw = json.loads((self.path / "scene" / "artifact.json").read_text())
            self._photos = {}
            for v in raw.get("train_views", []):
                key, f = v.get("frame_key"), v.get("image_file")
                if key and f:
                    sid, frame = parse_key(key)
                    if sid == stream:
                        self._photos[frame] = self.path / "scene" / f
        return self._photos

    def id_maps(self) -> tuple[dict, dict[int, np.ndarray]]:
        from reifyscene.graph.id_maps import export_id_maps, load_id_maps

        if self._maps is None:
            if not (self.path / "identity" / "manifest.json").is_file():
                export_id_maps(self.path)
            self._maps = load_id_maps(self.path / "identity")
        return self._maps


@dataclass
class Job:
    id: str
    run: str
    op: str
    params: dict[str, Any]
    status: str = "queued"          # queued | running | done | failed
    queued_at: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str | None = None
    out_dir: Path | None = None


def discover_runs(root: Path, checkpoint_subs: Sequence[str],
                  only: Sequence[str] | None = None) -> dict[str, Run]:
    runs: dict[str, Run] = {}
    # a run is whatever folder holds a graph and a trained checkpoint -- not a naming pattern
    names = only or sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    for name in names:
        path = Path(root) / name
        if not (path / "graph" / "artifact.json").is_file():
            continue
        sub = next((s for s in checkpoint_subs
                    if (path / s / "trained_dynamic.npz").is_file()), None)
        if sub is not None:
            runs[name] = Run(name=name, path=path, checkpoint_sub=sub)
    return runs


def create_app(
    runs_root: Path,
    *,
    checkpoint_subs: Sequence[str] = ("render",),
    only: Sequence[str] | None = None,
    device: str = "cuda",
    editor_factory: Callable[[Run], Any] | None = None,
):
    """Build the FastAPI app. ``editor_factory`` exists for tests (no GPU)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    from PIL import Image

    from reifyscene.edit.renderer import outline
    from reifyscene.query.scene_query import QueryError

    runs = discover_runs(runs_root, checkpoint_subs, only)
    jobs: dict[str, Job] = {}
    work: queue.Queue[str] = queue.Queue()
    gpu = threading.Lock()

    def make_editor(run: Run):
        from reifyscene.edit.remove import RemoveEditor

        return RemoveEditor(run.path, checkpoint_sub=run.checkpoint_sub, device=device)

    factory = editor_factory or make_editor

    def editor(run: Run):
        if run._editor is None:
            run._editor = factory(run)
        return run._editor

    def get_run(name: str) -> Run:
        if name not in runs:
            raise HTTPException(404, f"no run {name!r}")
        return runs[name]

    def worker() -> None:
        while True:
            jid = work.get()
            job = jobs[jid]
            job.status = "running"
            try:
                run = runs[job.run]
                with gpu:
                    ed = editor(run)
                    job.out_dir = run.path / "edits" / jid
                    job.result = ed.remove(job.params["node_id"],
                                           slackness=float(job.params.get("slackness", 0.0)),
                                           filtered=bool(job.params.get("filtered", True)),
                                           out_dir=job.out_dir)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 -- reported to the client, not swallowed
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            finally:
                work.task_done()

    threading.Thread(target=worker, daemon=True, name="edit-worker").start()

    app = FastAPI(title="reifyscene", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.middleware("http")
    async def no_stale_front_end(request, call_next):
        # The page, script and styles change while the app is developed alongside the graph;
        # a cached copy would silently show the old app. Images keep their own caching.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def jpeg(arr: np.ndarray) -> Response:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        out = []
        for r in runs.values():
            q = r.query
            frames = sorted({f for e in q.entities.values() for f in e.presence.observed})
            out.append({"name": r.name, "checkpoint": r.checkpoint_sub,
                        "entities": len(q.entities), "frames": len(r.photos()) or len(frames),
                        "categories": q.categories()})
        return out

    @app.get("/api/runs/{name}/graph")
    def graph(name: str) -> dict[str, Any]:
        r = get_run(name)
        q = r.query
        edges = [{"src": e.src, "dst": e.dst, "type": str(e.edge_type), "attrs": e.attrs}
                 for e in q.graph.edges if str(e.edge_type) != "observed_by"]
        layer = r.gaussians_layer()
        # an entity's Gaussians only mean something for the checkpoint they were computed on
        use = bool(layer and layer["current"])
        return {"run": name, "checkpoint": r.checkpoint_sub, "frames": sorted(r.photos()),
                "categories": q.categories(),
                "entities": [e.summary() | {"observed": e.presence.observed,
                                            "gaussians": e.attrs.get("gaussians") if use else None}
                             for e in sorted(q.entities.values(), key=lambda e: e.node_id)],
                "gaussians_layer": layer,
                "edges": edges,
                "relations_available": bool(edges),
                "relations_meta": (q.graph.metadata or {}).get("relations")}

    @app.post("/api/runs/{name}/query")
    def run_query(name: str, body: QueryIn) -> dict[str, Any]:
        try:
            return get_run(name).query.run(body.template, body.args)
        except (QueryError, KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/runs/{name}/frames/{frame}.jpg")
    def photo(name: str, frame: int):
        p = get_run(name).photos().get(frame)
        if p is None or not p.is_file():
            raise HTTPException(404, f"no photo for frame {frame}")
        return FileResponse(p)

    @app.get("/api/runs/{name}/render/{frame}.jpg")
    def render(name: str, frame: int):
        r = get_run(name)
        cache = r.path / "edits" / "cache" / r.checkpoint_sub / f"render_{frame:05d}.jpg"
        if cache.is_file():
            return FileResponse(cache)
        with gpu:
            ed = editor(r)
            if frame not in ed.renderer.time_of:
                raise HTTPException(404, f"no training camera for frame {frame}")
            arr = ed.render_before(frame)
        cache.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(cache, quality=90)
        return FileResponse(cache)

    @app.get("/api/runs/{name}/outline/{frame}.png")
    def entity_outline(name: str, frame: int, node: Annotated[list[str], Query()]):
        """Outlines of one or more entities on a frame, each in its own colour, in order:
        red, amber, violet -- the subject of a relation, its object, and between's third."""
        r = get_run(name)
        manifest, maps = r.id_maps()
        if frame not in maps:
            raise HTTPException(404, "no masks for that frame")
        index = {c["node_id"]: c["index"] for c in manifest["classes"] if c["node_id"]}
        if not any(n in index for n in node):
            raise HTTPException(404, "no mask for that entity and frame")
        m = maps[frame]
        layer = np.zeros((*m.shape, 4), np.uint8)
        for i, n in enumerate(node):
            if n in index:
                o = outline(m == index[n], width=3 if i else 2, color=OUTLINE_COLORS[i % 3])
                layer = np.where(o[..., 3:] > 0, o, layer)
        buf = io.BytesIO()
        Image.fromarray(layer, "RGBA").save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/runs/{name}/edits")
    def start_edit(name: str, body: EditIn) -> dict[str, Any]:
        r = get_run(name)
        if body.op != "remove":
            raise HTTPException(400, f"only remove is implemented, not {body.op!r}")
        try:
            node = r.query.entity(body.entity).node_id
        except (QueryError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        jid = f"remove-{node.rsplit(':', 1)[-1]}-{uuid.uuid4().hex[:8]}"
        jobs[jid] = Job(id=jid, run=name, op="remove",
                        params={"node_id": node, "slackness": body.slackness,
                                "filtered": body.filtered})
        work.put(jid)
        return {"id": jid, "status": "queued", "position": work.qsize()}

    @app.get("/api/edits/{jid}")
    def edit_status(jid: str) -> dict[str, Any]:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, f"no edit {jid!r}")
        out = {"id": job.id, "run": job.run, "op": job.op, "params": job.params,
               "status": job.status, "error": job.error}
        if job.result is not None:
            out["result"] = {k: v for k, v in job.result.items() if k != "frames"}
            out["frames"] = job.result.get("frames", [])
            out["urls"] = {k: f"/api/edits/{jid}/{k}/{{frame}}.jpg"
                           for k in ("before", "after", "took")}
        return out

    @app.get("/api/edits/{jid}/{kind}/{frame}.jpg")
    def edit_image(jid: str, kind: str, frame: int):
        job = jobs.get(jid)
        if job is None or job.out_dir is None or kind not in ("before", "after", "took"):
            raise HTTPException(404, "no such edit image")
        p = job.out_dir / f"{kind}_{frame:05d}.jpg"
        if not p.is_file():
            raise HTTPException(404, "no such edit image")
        return FileResponse(p)

    @app.get("/api/runs/{name}/view/{frame}.jpg")
    def free_view(name: str, frame: int,
                  yaw: Annotated[float, Query(ge=-180, le=180)] = 0.0,
                  pitch: Annotated[float, Query(ge=-85, le=85)] = 0.0,
                  zoom: Annotated[float, Query(ge=0.1, le=10)] = 1.0,
                  px: Annotated[float, Query(ge=-5, le=5)] = 0.0,
                  py: Annotated[float, Query(ge=-5, le=5)] = 0.0,
                  width: Annotated[int | None, Query(ge=64, le=4096)] = None,
                  edit: str | None = None,
                  show: Annotated[str, Query(pattern="^(scene|edited|took)$")] = "scene"):
        """The frame's moment from an orbited camera -- the 3D view. Never cached (every camera
        differs); refuses rather than queues while an edit holds the GPU."""
        r = get_run(name)
        edit_dir = None
        if show != "scene" and edit is None:
            raise HTTPException(400, f"show={show} needs an edit")
        if edit is not None and show != "scene":
            job = jobs.get(edit)
            if job is None or job.run != name or job.status != "done" or job.out_dir is None:
                raise HTTPException(404, f"no finished edit {edit!r} of {name}")
            edit_dir = job.out_dir
        if not gpu.acquire(timeout=0.25):
            raise HTTPException(503, "the GPU is busy with an edit")
        try:
            ed = editor(r)
            if frame not in ed.renderer.time_of:
                raise HTTPException(404, f"no training camera for frame {frame}")
            arr = ed.render_view(frame, edit_dir=edit_dir, show=show, yaw=yaw, pitch=pitch,
                                 zoom=zoom, pan_x=px, pan_y=py, width=width)
        finally:
            gpu.release()
        return jpeg(arr)

    @app.get("/api/queue")
    def queue_state() -> dict[str, Any]:
        return {"queued": [j.id for j in jobs.values() if j.status == "queued"],
                "running": [j.id for j in jobs.values() if j.status == "running"]}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app

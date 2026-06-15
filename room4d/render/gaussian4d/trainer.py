"""Native-gsplat layered renderer for assembled room4d scenes."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from room4d.common.memory import release_memory
from room4d.common.paths import ensure_dir
from room4d.render.base import RenderBackend, RenderSceneModel
from room4d.render.camera_paths import CameraPathGenerator, _to_z_up_points, _to_z_up_vec
from room4d.render.gaussian4d.rasterizer import (
    GaussianCloud,
    RenderCamera,
    build_rasterizer,
    project_to_pixels,
    require_gsplat,
)
from room4d.render.gaussian4d.render_artifact import RenderArtifact
from room4d.render.gaussian4d.scene_model import entity_cloud, static_cloud_from_artifact


class Gaussian4DTrainer(RenderBackend):
    """Render a layered static/dynamic Gaussian scene with gsplat."""

    backend_name = "gaussian4d"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_name = str(self.cfg.get("backend_name", type(self).backend_name))
        self.width = int(self.cfg.get("width", 800))
        self.height = int(self.cfg.get("height", 450))
        self.point_limit = int(self.cfg.get("point_limit", 12000))
        self.frame_stride = max(1, int(self.cfg.get("frame_stride", 1)))
        self.max_frames_per_path = self.cfg.get("max_frames_per_path")
        self.label_entities = bool(self.cfg.get("label_entities", True))
        self.debug_triptych = bool(self.cfg.get("debug_triptych", True))
        self.max_dynamic_entities_per_kind = int(self.cfg.get("max_dynamic_entities_per_kind", 3))
        self.render_scale = max(1, int(self.cfg.get("render_scale", 1)))
        self.device = str(self.cfg.get("device", "auto"))
        if self.device in ("auto", "cpu"):
            # gsplat is GPU-only; require_gsplat() raises at render time if absent.
            self.device = "cuda"
        self.camera_paths = CameraPathGenerator()
        self.rasterizer = build_rasterizer(
            {"rasterizer_backend": self.cfg.get("rasterizer_backend", "gsplat"), "device": self.device}
        )
        self.enabled_paths = {
            "orbit": True,
            "overhead": True,
            "replay": True,
            "follow_human": True,
            "follow_object": True,
        }
        self.enabled_paths.update({str(k): bool(v) for k, v in (self.cfg.get("enabled_paths", {}) or {}).items()})
        self._asset_cache: dict[str, Any] = {}

    def release_resources(self) -> None:
        self._asset_cache.clear()
        release_memory()

    def _limit_cameras(self, cameras: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limited = cameras[:: self.frame_stride] if self.frame_stride > 1 else list(cameras)
        if self.max_frames_per_path is not None:
            limited = limited[: int(self.max_frames_per_path)]
        return limited

    def _scene_contract(self, scene_artifact) -> dict[str, Any]:
        return dict(getattr(scene_artifact, "render_contract", {}) or {})

    def _static_cloud(self, scene_artifact) -> GaussianCloud:
        key = f"static_cloud::{id(scene_artifact)}"
        if key not in self._asset_cache:
            self._asset_cache[key] = static_cloud_from_artifact(
                scene_artifact, device=self.device, point_limit=self.point_limit
            )
        return self._asset_cache[key]

    @staticmethod
    def _closest_valid_center(times: np.ndarray, centers: list[Any], fallback: list[float], time_sec: float) -> np.ndarray:
        candidates: list[tuple[float, np.ndarray]] = []
        for idx, center in enumerate(centers):
            if center is None:
                continue
            arr = np.asarray(center, dtype=np.float32)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                continue
            t = float(times[idx]) if idx < times.size else float(time_sec)
            candidates.append((abs(t - float(time_sec)), arr))
        if not candidates:
            return np.asarray(fallback, dtype=np.float32)
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1].astype(np.float32)

    def _select_dynamic_layers(self, scene_artifact, time_sec: float) -> list[dict[str, Any]]:
        contract = self._scene_contract(scene_artifact)
        dynamic_layers = contract.get("dynamic_layers", {}) or {}
        selected: list[dict[str, Any]] = []
        for kind in ("humans", "objects"):
            per_kind: list[dict[str, Any]] = []
            for ent in dynamic_layers.get(kind, []):
                times = np.asarray(ent.get("times_sec", []), dtype=np.float32)
                centers = ent.get("centers_xyz", [])
                if times.size == 0:
                    center = np.asarray(ent.get("anchor_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
                else:
                    center = self._closest_valid_center(times, centers, ent.get("anchor_xyz", [0.0, 0.0, 0.0]), float(time_sec))
                if not np.all(np.isfinite(center)):
                    continue
                support_counts = ent.get("support_counts", [])
                per_kind.append({
                    "entity_id": ent["entity_id"],
                    "entity_type": ent["entity_type"],
                    "display_label": ent.get("display_label", ent["entity_type"]),
                    "center_xyz": _to_z_up_vec(center).astype(np.float32),
                    "template_offsets_xyz": _to_z_up_points(np.asarray(ent.get("template_offsets_xyz", [[0.0, 0.0, 0.0]]), dtype=np.float32)),
                    "template_scales_xyz": np.asarray(ent.get("template_scales_xyz", [[0.12, 0.12, 0.12]]), dtype=np.float32),
                    "template_opacities": np.asarray(ent.get("template_opacities", [0.8]), dtype=np.float32),
                    "color_rgb": np.asarray(ent.get("color_rgb", [255, 64, 64]), dtype=np.uint8),
                    "support_count": int(max(support_counts) if support_counts else 0),
                    "render_priority": float(ent.get("render_priority", 0.0)),
                })
            per_kind = sorted(per_kind, key=lambda e: (-e["render_priority"], -e["support_count"], e["entity_id"]))[: self.max_dynamic_entities_per_kind]
            selected.extend(per_kind)
        return selected

    def _render_frame_arrays(self, scene_artifact, camera: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int, int]:
        require_gsplat()
        render_w = self.width * self.render_scale
        render_h = self.height * self.render_scale
        static_cloud = self._static_cloud(scene_artifact)
        dynamic_entities = self._select_dynamic_layers(scene_artifact, float(camera.get("time_sec", 0.0)))
        dynamic_cloud = GaussianCloud.concat(
            [entity_cloud(ent, device=self.device) for ent in dynamic_entities], device=self.device
        )
        full_cloud = GaussianCloud.concat([static_cloud, dynamic_cloud], device=self.device)

        render_camera = RenderCamera.from_look_at(
            camera["eye"], camera["target"], camera.get("up", [0.0, 0.0, 1.0]), render_w, render_h
        )
        static_r = self.rasterizer.render(static_cloud, render_camera)
        dynamic_r = self.rasterizer.render(dynamic_cloud, render_camera)
        full_r = self.rasterizer.render(full_cloud, render_camera)
        arrays = {
            "full": full_r["rgb"],
            "static_only": static_r["rgb"],
            "dynamic_only": dynamic_r["rgb"],
            "full_alpha": full_r["alpha"],
            "static_alpha": static_r["alpha"],
            "dynamic_alpha": dynamic_r["alpha"],
        }
        return arrays, dynamic_entities, render_w, render_h

    @staticmethod
    def _canvas_to_image(rgb: np.ndarray) -> Image.Image:
        canvas = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(canvas)

    def _annotate_entities(self, image: Image.Image, dynamic_entities: list[dict[str, Any]], camera: dict[str, Any]) -> Image.Image:
        if not self.label_entities or not dynamic_entities:
            return image
        draw = ImageDraw.Draw(image)
        for entity in dynamic_entities:
            xy, valid = project_to_pixels(
                np.asarray([entity["center_xyz"]], dtype=np.float32), camera, self.width, self.height
            )
            if not bool(valid[0]):
                continue
            x, y = float(xy[0, 0]), float(xy[0, 1])
            color = tuple(int(v) for v in entity["color_rgb"].tolist())
            draw.text((x + 6, y - 10), f"{entity['display_label']}:{entity['entity_id']}", fill=color)
        return image

    def _render_frame_bundle(self, scene_artifact, camera: dict[str, Any], selection_memory: dict[str, Any] | None = None) -> tuple[dict[str, Image.Image], list[dict[str, Any]]]:
        arrays, dynamic_entities, render_w, render_h = self._render_frame_arrays(scene_artifact, camera)
        images = {
            "full": self._canvas_to_image(arrays["full"]),
            "static_only": self._canvas_to_image(arrays["static_only"]),
            "dynamic_only": self._canvas_to_image(arrays["dynamic_only"]),
        }
        if render_w != self.width or render_h != self.height:
            for key, image in list(images.items()):
                images[key] = image.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)
        images["full"] = self._annotate_entities(images["full"], dynamic_entities, camera)
        images["triptych"] = self._compose_triptych(images)
        return images, dynamic_entities

    def _compose_triptych(self, images: dict[str, Image.Image]) -> Image.Image:
        labels = [("static_only", "Static"), ("dynamic_only", "Dynamic"), ("full", "Full")]
        triptych = Image.new("RGB", (self.width * 3, self.height), color=(0, 0, 0))
        for idx, (key, label) in enumerate(labels):
            panel = images[key].copy()
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 112, 20), fill=(0, 0, 0))
            draw.text((8, 4), label, fill=(255, 255, 255))
            triptych.paste(panel, (idx * self.width, 0))
        return triptych

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _write_player_html(self, render_dir: Path, path_index: dict[str, list[str]]) -> str:
        html_path = render_dir / "novel_view_player.html"
        frames = {path: [Path(rel).as_posix() for rel in rels] for path, rels in path_index.items()}
        first_path = next(iter(frames.keys()), "")
        first_frame = frames[first_path][0] if frames.get(first_path) else ""
        html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>room4d novel-view player</title>
<style>
body {{ font-family: sans-serif; margin: 18px; }}
.controls {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
img {{ max-width: 100%; border: 1px solid #ccc; }}
pre {{ background:#f6f6f6; padding:12px; }}
</style></head>
<body>
<h2>room4d novel-view player</h2>
<div class='controls'>
<label>Path <select id='pathSelect'></select></label>
<label>Frame <input id='frameSlider' type='range' min='0' max='0' value='0' /></label>
<span id='frameLabel'></span>
</div>
<img id='viewer' src='{first_frame}' />
<pre id='summary'></pre>
<script>
const frames = {frames!r};
const pathSelect = document.getElementById('pathSelect');
const frameSlider = document.getElementById('frameSlider');
const viewer = document.getElementById('viewer');
const frameLabel = document.getElementById('frameLabel');
const summary = document.getElementById('summary');
Object.keys(frames).forEach((path) => {{
  const opt = document.createElement('option'); opt.value = path; opt.textContent = path; pathSelect.appendChild(opt);
}});
function update() {{
  const path = pathSelect.value;
  const seq = frames[path] || [];
  frameSlider.max = Math.max(seq.length - 1, 0);
  const idx = Math.min(Number(frameSlider.value || 0), Math.max(seq.length - 1, 0));
  viewer.src = seq[idx] || '';
  frameLabel.textContent = `${{path}} frame ${{idx + 1}} / ${{seq.length}}`;
  summary.textContent = JSON.stringify({{path, frame_index: idx, frames_in_path: seq.length}}, null, 2);
}}
pathSelect.value = {first_path!r};
pathSelect.addEventListener('change', () => {{ frameSlider.value = '0'; update(); }});
frameSlider.addEventListener('input', update);
update();
</script></body></html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path.name

    def _write_debug_player_html(self, render_dir: Path, layer_path_index: dict[str, dict[str, list[str]]]) -> str:
        html_path = render_dir / "novel_view_debug_player.html"
        frames = {
            path: {layer: [Path(rel).as_posix() for rel in rels] for layer, rels in layers.items()}
            for path, layers in layer_path_index.items()
        }
        first_path = next(iter(frames.keys()), "")
        first_layers = frames.get(first_path, {})
        html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>room4d debug viewer</title>
<style>
body {{ font-family: sans-serif; margin: 18px; }}
.controls {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.panel img {{ width: 100%; border: 1px solid #ccc; }}
.panel h3 {{ margin: 0 0 6px 0; }}
pre {{ background:#f6f6f6; padding:12px; }}
</style></head>
<body>
<h2>room4d debug render viewer</h2>
<div class='controls'>
<label>Path <select id='pathSelect'></select></label>
<label>Frame <input id='frameSlider' type='range' min='0' max='0' value='0' /></label>
<span id='frameLabel'></span>
</div>
<div class='grid'>
  <div class='panel'><h3>Static</h3><img id='staticImg' src='{(first_layers.get("static_only") or [""])[0] if first_layers else ""}' /></div>
  <div class='panel'><h3>Dynamic</h3><img id='dynamicImg' src='{(first_layers.get("dynamic_only") or [""])[0] if first_layers else ""}' /></div>
  <div class='panel'><h3>Full</h3><img id='fullImg' src='{(first_layers.get("full") or [""])[0] if first_layers else ""}' /></div>
</div>
<pre id='summary'></pre>
<script>
const frames = {frames!r};
const pathSelect = document.getElementById('pathSelect');
const frameSlider = document.getElementById('frameSlider');
const staticImg = document.getElementById('staticImg');
const dynamicImg = document.getElementById('dynamicImg');
const fullImg = document.getElementById('fullImg');
const frameLabel = document.getElementById('frameLabel');
const summary = document.getElementById('summary');
Object.keys(frames).forEach((path) => {{
  const opt = document.createElement('option'); opt.value = path; opt.textContent = path; pathSelect.appendChild(opt);
}});
function seqLen(layers) {{
  return Math.max((layers.full || []).length, (layers.static_only || []).length, (layers.dynamic_only || []).length, 0);
}}
function update() {{
  const path = pathSelect.value;
  const layers = frames[path] || {{}};
  const count = seqLen(layers);
  frameSlider.max = Math.max(count - 1, 0);
  const idx = Math.min(Number(frameSlider.value || 0), Math.max(count - 1, 0));
  staticImg.src = (layers.static_only || [])[idx] || '';
  dynamicImg.src = (layers.dynamic_only || [])[idx] || '';
  fullImg.src = (layers.full || [])[idx] || '';
  frameLabel.textContent = `${{path}} frame ${{idx + 1}} / ${{count}}`;
  summary.textContent = JSON.stringify({{path, frame_index: idx, frames_in_path: count}}, null, 2);
}}
pathSelect.value = {first_path!r};
pathSelect.addEventListener('change', () => {{ frameSlider.value = '0'; update(); }});
frameSlider.addEventListener('input', update);
update();
</script></body></html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path.name

    def fit(self, scene: RenderSceneModel, optimization_artifact=None, geometry_artifact=None, save_dir: str | Path | None = None) -> RenderArtifact:
        scene_artifact = scene.scene_artifact
        if scene_artifact is None:
            raise ValueError("Scene model has not been initialized from a SceneArtifact.")
        require_gsplat()
        render_dir = ensure_dir(Path(save_dir) if save_dir is not None else Path("."))
        paths: dict[str, dict[str, Any]] = {}
        if self.enabled_paths.get("orbit", True):
            paths["orbit"] = self.camera_paths.orbit(scene_artifact)
        if self.enabled_paths.get("overhead", True):
            paths["overhead"] = self.camera_paths.overhead(scene_artifact)
        if self.enabled_paths.get("replay", True):
            paths["replay"] = self.camera_paths.replay(scene_artifact, optimization_artifact=optimization_artifact, geometry_artifact=geometry_artifact)
        if self.enabled_paths.get("follow_human", True) and scene_artifact.dynamic_entities.get("humans"):
            first_human = scene_artifact.dynamic_entities["humans"][0]["entity_id"]
            paths[f"follow_human_{first_human}"] = self.camera_paths.follow_subject(scene_artifact, first_human)
        if self.enabled_paths.get("follow_object", True) and scene_artifact.dynamic_entities.get("objects"):
            first_object = scene_artifact.dynamic_entities["objects"][0]["entity_id"]
            paths[f"follow_object_{first_object}"] = self.camera_paths.follow_subject(scene_artifact, first_object)

        views_root = ensure_dir(render_dir / "novel_views")
        debug_root = ensure_dir(render_dir / "novel_views_debug")
        path_index: dict[str, list[str]] = {}
        layer_path_index: dict[str, dict[str, list[str]]] = {}
        total_frames = 0
        debug_frames = 0
        for path_name, path in paths.items():
            cameras = self._limit_cameras(path.get("cameras", []))
            if not cameras:
                continue
            path_dir = ensure_dir(views_root / path_name)
            debug_path_dir = ensure_dir(debug_root / path_name)
            rels: list[str] = []
            layer_rels = {"full": [], "static_only": [], "dynamic_only": [], "triptych": []}
            for idx, camera in enumerate(cameras):
                images, _ = self._render_frame_bundle(scene_artifact, camera)
                filename = f"frame_{idx:05d}.png"
                images["full"].save(path_dir / filename)
                rels.append(str((Path("novel_views") / path_name / filename).as_posix()))
                if self.debug_triptych:
                    for key, stem in (("full", "full"), ("static_only", "static_only"), ("dynamic_only", "dynamic_only"), ("triptych", "triptych")):
                        layer_filename = f"frame_{idx:05d}_{stem}.png"
                        images[key].save(debug_path_dir / layer_filename)
                        layer_rels[key].append(str((Path("novel_views_debug") / path_name / layer_filename).as_posix()))
                    debug_frames += 1
            path_index[path_name] = rels
            if self.debug_triptych:
                layer_path_index[path_name] = layer_rels
            total_frames += len(rels)
            release_memory()

        player_html = self._write_player_html(render_dir, path_index)
        debug_player_html = self._write_debug_player_html(render_dir, layer_path_index) if layer_path_index else None
        static_layer = self._scene_contract(scene_artifact).get("static_layer", {}) or {}
        dynamic_layers = self._scene_contract(scene_artifact).get("dynamic_layers", {}) or {}
        return RenderArtifact(
            run_id=scene_artifact.run_id,
            stage_name="render",
            save_dir=render_dir,
            scene_state={
                "backend_name": self.backend_name,
                "rasterizer": "gsplat",
                "scene_contract": scene.summary(),
                "static_points": int(scene_artifact.static_scene.get("num_points", 0)),
                "static_gaussians": int(static_layer.get("num_gaussians", 0)),
                "dynamic_entities": {k: len(v) for k, v in scene_artifact.dynamic_entities.items()},
                "dynamic_layers": {k: len(v) for k, v in dynamic_layers.items()} if isinstance(dynamic_layers, dict) else {},
                "device": self.device,
                "render_scale": self.render_scale,
                "enabled_paths": {k: bool(v) for k, v in self.enabled_paths.items()},
                "debug_triptych": self.debug_triptych,
            },
            coverage={"rendered_paths": len(path_index), "rendered_frames": total_frames, "debug_frames": debug_frames},
            path_index=path_index,
            layer_path_index=layer_path_index,
            player_html=player_html,
            debug_player_html=debug_player_html,
        )

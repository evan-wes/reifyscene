"""Simple point-splat novel-view renderer for single-video 4D scenes."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from room4d.common.paths import ensure_dir
from room4d.render.camera_paths import CameraPathGenerator
from room4d.render.gaussian4d.render_artifact import RenderArtifact


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


class Gaussian4DTrainer:
    """Fit a lightweight renderable scene representation from assembled state."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.width = int(self.cfg.get("width", 960))
        self.height = int(self.cfg.get("height", 540))
        self.point_limit = int(self.cfg.get("point_limit", 40000))
        self.camera_paths = CameraPathGenerator()

    def _select_dynamic_points(self, scene_artifact, time_sec: float) -> list[dict[str, Any]]:
        markers = []
        for kind in ("humans", "objects"):
            for ent in scene_artifact.dynamic_entities.get(kind, []):
                times = np.asarray(ent.get("times_sec", []), dtype=np.float32)
                centers = ent.get("centers_3d", [])
                if times.size == 0:
                    center = np.asarray(ent.get("anchor_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
                else:
                    idx = int(np.argmin(np.abs(times - float(time_sec))))
                    c = centers[idx] if idx < len(centers) else None
                    center = np.asarray(c if c is not None else ent.get("anchor_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
                if not np.all(np.isfinite(center)):
                    continue
                markers.append({
                    "entity_id": ent["entity_id"],
                    "entity_type": ent["entity_type"],
                    "center_xyz": center.astype(np.float32),
                    "color_rgb": np.asarray(ent.get("color_rgb", [255, 64, 64]), dtype=np.uint8),
                    "label": ent.get("display_label", ent["entity_type"]),
                })
        return markers

    def _project_points(self, points: np.ndarray, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forward = _normalize(target - eye)
        right = _normalize(np.cross(forward, up))
        if not np.any(right):
            right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        true_up = _normalize(np.cross(right, forward))
        R = np.stack([right, true_up, forward], axis=0)
        cam = (points - eye[None, :]) @ R.T
        z = cam[:, 2]
        valid = z > 1e-3
        fx = fy = 0.75 * max(self.width, self.height)
        x = fx * (cam[:, 0] / np.clip(z, 1e-3, None)) + self.width * 0.5
        y = -fy * (cam[:, 1] / np.clip(z, 1e-3, None)) + self.height * 0.5
        return np.stack([x, y], axis=1), valid

    def _render_frame(self, scene_artifact, camera: dict[str, Any]) -> Image.Image:
        bg = np.asarray(scene_artifact.static_scene.get("points_xyz", np.zeros((0, 3))), dtype=np.float32)
        colors = np.asarray(scene_artifact.static_scene.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        if bg.shape[0] > self.point_limit:
            stride = max(1, int(np.ceil(bg.shape[0] / float(self.point_limit))))
            bg = bg[::stride]
            colors = colors[::stride]
        eye = np.asarray(camera["eye"], dtype=np.float32)
        target = np.asarray(camera["target"], dtype=np.float32)
        up = np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float32)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        zbuf = np.full((self.height, self.width), np.inf, dtype=np.float32)
        if bg.size:
            xy, valid = self._project_points(bg, eye, target, up)
            cam_z = ((_normalize(target - eye)[None, :] * 0) + 1)  # placeholder
            for idx, ok in enumerate(valid.tolist()):
                if not ok:
                    continue
                x, y = int(round(float(xy[idx, 0]))), int(round(float(xy[idx, 1])))
                if 0 <= x < self.width and 0 <= y < self.height:
                    depth = float(np.linalg.norm(bg[idx] - eye))
                    if depth < zbuf[y, x]:
                        zbuf[y, x] = depth
                        canvas[y, x] = colors[idx]
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for marker in self._select_dynamic_points(scene_artifact, float(camera.get("time_sec", 0.0))):
            pts = np.asarray([marker["center_xyz"]], dtype=np.float32)
            xy, valid = self._project_points(pts, eye, target, up)
            if not bool(valid[0]):
                continue
            x, y = float(xy[0, 0]), float(xy[0, 1])
            color = tuple(int(v) for v in marker["color_rgb"].tolist())
            r = 7 if marker["entity_type"] == "human" else 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255))
            draw.text((x + r + 2, y - r), f"{marker['label']}:{marker['entity_id']}", fill=color)
        return img

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

    def fit(self, scene, optimization_artifact=None, geometry_artifact=None, save_dir: str | Path | None = None) -> RenderArtifact:
        scene_artifact = scene.scene_artifact
        if scene_artifact is None:
            raise ValueError("Scene model has not been initialized from a SceneArtifact.")
        render_dir = ensure_dir(Path(save_dir) if save_dir is not None else Path("."))
        paths = {
            "orbit": self.camera_paths.orbit(scene_artifact),
            "overhead": self.camera_paths.overhead(scene_artifact),
            "replay": self.camera_paths.replay(scene_artifact, optimization_artifact=optimization_artifact, geometry_artifact=geometry_artifact),
        }
        if scene_artifact.dynamic_entities.get("humans"):
            first_human = scene_artifact.dynamic_entities["humans"][0]["entity_id"]
            paths[f"follow_human_{first_human}"] = self.camera_paths.follow_subject(scene_artifact, first_human)
        if scene_artifact.dynamic_entities.get("objects"):
            first_object = scene_artifact.dynamic_entities["objects"][0]["entity_id"]
            paths[f"follow_object_{first_object}"] = self.camera_paths.follow_subject(scene_artifact, first_object)

        views_root = ensure_dir(render_dir / "novel_views")
        path_index: dict[str, list[str]] = {}
        total_frames = 0
        for path_name, path in paths.items():
            cameras = path.get("cameras", [])
            if not cameras:
                continue
            path_dir = ensure_dir(views_root / path_name)
            rels = []
            for idx, camera in enumerate(cameras):
                image = self._render_frame(scene_artifact, camera)
                filename = f"frame_{idx:05d}.png"
                image.save(path_dir / filename)
                rels.append(str((Path("novel_views") / path_name / filename).as_posix()))
            path_index[path_name] = rels
            total_frames += len(rels)

        player_html = self._write_player_html(render_dir, path_index)
        return RenderArtifact(
            run_id=scene_artifact.run_id,
            stage_name="render",
            save_dir=render_dir,
            scene_state={
                "static_points": int(scene_artifact.static_scene.get("num_points", 0)),
                "dynamic_entities": {k: len(v) for k, v in scene_artifact.dynamic_entities.items()},
            },
            coverage={"rendered_paths": len(path_index), "rendered_frames": total_frames},
            path_index=path_index,
            player_html=player_html,
        )

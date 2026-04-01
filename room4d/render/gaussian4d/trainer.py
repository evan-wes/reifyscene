"""Simple point-splat novel-view renderer for single-video 4D scenes."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from room4d.common.memory import release_memory
from room4d.common.paths import ensure_dir
from room4d.render.camera_paths import CameraPathGenerator, _to_z_up_points, _to_z_up_vec
from room4d.render.gaussian4d.render_artifact import RenderArtifact

try:  # optional GPU/vector math acceleration
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


class Gaussian4DTrainer:
    """Fit a lightweight renderable scene representation from assembled state."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.width = int(self.cfg.get("width", 800))
        self.height = int(self.cfg.get("height", 450))
        self.point_limit = int(self.cfg.get("point_limit", 12000))
        self.dynamic_point_budget = int(self.cfg.get("dynamic_point_budget", 240))
        self.frame_stride = max(1, int(self.cfg.get("frame_stride", 1)))
        self.max_frames_per_path = self.cfg.get("max_frames_per_path")
        self.label_entities = bool(self.cfg.get("label_entities", True))
        self.static_radius_px = int(self.cfg.get("static_radius_px", 1))
        self.human_radius_px = int(self.cfg.get("human_radius_px", 2))
        self.object_radius_px = int(self.cfg.get("object_radius_px", 1))
        self.max_dynamic_entities_per_kind = int(self.cfg.get("max_dynamic_entities_per_kind", 3))
        self.cpu_threads = self.cfg.get("cpu_threads")
        self.use_gpu_projection = bool(self.cfg.get("use_gpu_projection", True))
        self.device = self._select_device(self.cfg.get("device", "auto"))
        self.camera_paths = CameraPathGenerator()
        self.enabled_paths = {
            "orbit": True,
            "overhead": True,
            "replay": True,
            "follow_human": True,
            "follow_object": True,
        }
        self.enabled_paths.update({str(k): bool(v) for k, v in (self.cfg.get("enabled_paths", {}) or {}).items()})
        if torch is not None and self.cpu_threads is not None:
            try:
                torch.set_num_threads(int(self.cpu_threads))
            except Exception:
                pass

    def _select_device(self, requested: str | None) -> str:
        requested = str(requested or "auto").lower()
        if requested == "auto":
            if self.use_gpu_projection and torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if requested.startswith("cuda"):
            if self.use_gpu_projection and torch is not None and torch.cuda.is_available():
                return requested
            return "cpu"
        return "cpu"

    def release_resources(self) -> None:
        release_memory()

    def _limit_cameras(self, cameras: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limited = cameras[:: self.frame_stride] if self.frame_stride > 1 else list(cameras)
        if self.max_frames_per_path is not None:
            limited = limited[: int(self.max_frames_per_path)]
        return limited

    def _select_dynamic_points(self, scene_artifact, time_sec: float) -> list[dict[str, Any]]:
        entities = []
        for kind in ("humans", "objects"):
            per_kind = []
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
                center = _to_z_up_vec(center)
                per_kind.append({
                    "entity_id": ent["entity_id"],
                    "entity_type": ent["entity_type"],
                    "center_xyz": center.astype(np.float32),
                    "color_rgb": np.asarray(ent.get("color_rgb", [255, 64, 64]), dtype=np.uint8),
                    "label": ent.get("display_label", ent["entity_type"]),
                    "support_count": int(np.max(ent.get("support_counts", [0])) if ent.get("support_counts") else 0),
                })
            per_kind = sorted(per_kind, key=lambda e: (-e["support_count"], e["entity_id"]))[: self.max_dynamic_entities_per_kind]
            entities.extend(per_kind)
        return entities

    def _entity_geometry(self, entity: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(entity["center_xyz"], dtype=np.float32)
        color = np.asarray(entity["color_rgb"], dtype=np.uint8)
        if entity["entity_type"] == "human":
            height = 1.6
            radius = 0.16
            levels = np.linspace(-0.45 * height, 0.55 * height, 7, dtype=np.float32)
            angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False, dtype=np.float32)
            pts = []
            for z_off in levels:
                r = radius * (0.75 if abs(z_off) > 0.35 * height else 1.0)
                for a in angles:
                    pts.append(center + np.asarray([r * np.cos(a), r * np.sin(a), z_off], dtype=np.float32))
            pts.append(center + np.asarray([0.0, 0.0, 0.75 * height], dtype=np.float32))
            pts.append(center + np.asarray([0.0, 0.0, -0.55 * height], dtype=np.float32))
        else:
            radius = 0.14
            phi = np.linspace(0.0, np.pi, 6, dtype=np.float32)
            theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
            pts = []
            for p in phi:
                for t in theta:
                    pts.append(center + radius * np.asarray([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)], dtype=np.float32))
            pts.append(center)
        pts_arr = np.asarray(pts, dtype=np.float32)
        budget = max(1, int(self.dynamic_point_budget // max(1, 2 * self.max_dynamic_entities_per_kind)))
        if pts_arr.shape[0] > budget:
            stride = max(1, int(np.ceil(pts_arr.shape[0] / float(budget))))
            pts_arr = pts_arr[::stride]
        cols = np.repeat(color[None, :], pts_arr.shape[0], axis=0)
        return pts_arr, cols

    def _project_points_numpy(self, points: np.ndarray, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return np.stack([x, y], axis=1), valid, z

    def _project_points(self, points: np.ndarray, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            try:
                with torch.no_grad():
                    p = torch.as_tensor(points, dtype=torch.float32, device=self.device)
                    eye_t = torch.as_tensor(eye, dtype=torch.float32, device=self.device)
                    target_t = torch.as_tensor(target, dtype=torch.float32, device=self.device)
                    up_t = torch.as_tensor(up, dtype=torch.float32, device=self.device)
                    forward = target_t - eye_t
                    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-8)
                    right = torch.linalg.cross(forward, up_t)
                    right_norm = torch.linalg.norm(right)
                    if float(right_norm) < 1e-8:
                        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                    else:
                        right = right / right_norm
                    true_up = torch.linalg.cross(right, forward)
                    true_up = true_up / torch.clamp(torch.linalg.norm(true_up), min=1e-8)
                    R = torch.stack([right, true_up, forward], dim=0)
                    cam = (p - eye_t[None, :]) @ R.T
                    z = cam[:, 2]
                    valid = z > 1e-3
                    fx = fy = 0.75 * max(self.width, self.height)
                    x = fx * (cam[:, 0] / torch.clamp(z, min=1e-3)) + self.width * 0.5
                    y = -fy * (cam[:, 1] / torch.clamp(z, min=1e-3)) + self.height * 0.5
                    return (
                        torch.stack([x, y], dim=1).cpu().numpy(),
                        valid.cpu().numpy(),
                        z.cpu().numpy(),
                    )
            except Exception as exc:
                if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
                    self.device = "cpu"
                    release_memory()
                else:
                    try:
                        release_memory()
                    except Exception:
                        pass
        return self._project_points_numpy(points, eye, target, up)

    @staticmethod
    def _disk_offsets(radius_px: int) -> np.ndarray:
        offs = []
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy <= radius_px * radius_px:
                    offs.append((dx, dy))
        return np.asarray(offs, dtype=np.int32)

    def _splat_points(self, canvas: np.ndarray, zbuf: np.ndarray, points: np.ndarray, point_colors: np.ndarray, eye: np.ndarray, target: np.ndarray, up: np.ndarray, radius_px: int) -> None:
        if points.size == 0:
            return
        xy, valid, z = self._project_points(points, eye, target, up)
        valid &= np.isfinite(xy).all(axis=1) & np.isfinite(z)
        if not np.any(valid):
            return
        xy = xy[valid]
        z = z[valid].astype(np.float32)
        point_colors = point_colors[valid]
        x0 = np.rint(xy[:, 0]).astype(np.int32)
        y0 = np.rint(xy[:, 1]).astype(np.int32)
        offsets = self._disk_offsets(radius_px)
        xs = x0[:, None] + offsets[None, :, 0]
        ys = y0[:, None] + offsets[None, :, 1]
        depths = np.repeat(z[:, None], offsets.shape[0], axis=1)
        cols = np.repeat(point_colors[:, None, :], offsets.shape[0], axis=1)
        in_bounds = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        if not np.any(in_bounds):
            return
        xs = xs[in_bounds]
        ys = ys[in_bounds]
        depths = depths[in_bounds]
        cols = cols[in_bounds]
        lin = ys.astype(np.int64) * self.width + xs.astype(np.int64)
        order = np.lexsort((depths, lin))
        lin_sorted = lin[order]
        keep = np.empty(lin_sorted.shape[0], dtype=bool)
        keep[0] = True
        keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
        chosen = order[keep]
        xs = xs[chosen]
        ys = ys[chosen]
        depths = depths[chosen]
        cols = cols[chosen]
        # only overwrite pixels where the new depth is nearer
        nearer = depths < zbuf[ys, xs]
        if np.any(nearer):
            xs = xs[nearer]
            ys = ys[nearer]
            depths = depths[nearer]
            cols = cols[nearer]
            zbuf[ys, xs] = depths
            canvas[ys, xs] = cols

    def _render_frame(self, scene_artifact, camera: dict[str, Any]) -> Image.Image:
        bg = _to_z_up_points(np.asarray(scene_artifact.static_scene.get("points_xyz", np.zeros((0, 3))), dtype=np.float32))
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
            self._splat_points(canvas, zbuf, bg, colors, eye, target, up, radius_px=self.static_radius_px)

        dynamic_entities = self._select_dynamic_points(scene_artifact, float(camera.get("time_sec", 0.0)))
        for entity in dynamic_entities:
            pts, cols = self._entity_geometry(entity)
            self._splat_points(
                canvas,
                zbuf,
                pts,
                cols,
                eye,
                target,
                up,
                radius_px=self.human_radius_px if entity["entity_type"] == "human" else self.object_radius_px,
            )

        img = Image.fromarray(canvas)
        if self.label_entities:
            draw = ImageDraw.Draw(img)
            for entity in dynamic_entities:
                pts = np.asarray([entity["center_xyz"]], dtype=np.float32)
                xy, valid, _ = self._project_points(pts, eye, target, up)
                if not bool(valid[0]):
                    continue
                x, y = float(xy[0, 0]), float(xy[0, 1])
                color = tuple(int(v) for v in entity["color_rgb"].tolist())
                draw.text((x + 6, y - 10), f"{entity['label']}:{entity['entity_id']}", fill=color)
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
        path_index: dict[str, list[str]] = {}
        total_frames = 0
        for path_name, path in paths.items():
            cameras = self._limit_cameras(path.get("cameras", []))
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
            release_memory()

        player_html = self._write_player_html(render_dir, path_index)
        return RenderArtifact(
            run_id=scene_artifact.run_id,
            stage_name="render",
            save_dir=render_dir,
            scene_state={
                "static_points": int(scene_artifact.static_scene.get("num_points", 0)),
                "dynamic_entities": {k: len(v) for k, v in scene_artifact.dynamic_entities.items()},
                "device": self.device,
                "enabled_paths": {k: bool(v) for k, v in self.enabled_paths.items()},
            },
            coverage={"rendered_paths": len(path_index), "rendered_frames": total_frames},
            path_index=path_index,
            player_html=player_html,
        )

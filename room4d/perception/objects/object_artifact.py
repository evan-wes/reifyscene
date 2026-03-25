
"""Artifact for object detections, masks, embeddings, and associations."""

from __future__ import annotations

import colorsys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from rich.progress import track

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import BaseArtifact, FrameRef, ObjectState


@dataclass
class ObjectArtifact(BaseArtifact):
    """Store object detections, masks, embeddings, and debug exports."""

    objects: list[ObjectState] = field(default_factory=list)
    linked_ids: dict[str, Any] = field(default_factory=dict)
    frame_index: dict[str, list[str]] = field(default_factory=dict)
    mask_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    embedding_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _frame_key(frame_ref: FrameRef) -> str:
        return f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}"

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    @staticmethod
    def _track_color(track_like_id: str) -> str:
        numeric = abs(hash(track_like_id)) % 10_000
        hue = ((numeric * 0.6180339887498949) % 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    @classmethod
    def _build_frame_index(cls, objects: list[ObjectState]) -> dict[str, list[str]]:
        frame_index: dict[str, list[str]] = {}
        for obj in objects:
            frame_index.setdefault(cls._frame_key(obj.frame_ref), []).append(obj.object_id)
        return frame_index

    def summary(self) -> dict[str, Any]:
        unique_ids = sorted({obj.object_id for obj in self.objects})
        return {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'num_objects': len(self.objects),
            'num_unique_ids': len(unique_ids),
            'num_frames': len(self.frame_index or self._build_frame_index(self.objects)),
            'notes': list(self.notes),
        }

    def _progress_iter(self, iterable, *, total: int | None = None, description: str = ''):
        enabled = bool(self.metadata.get('save_debug_assets', False) or self.metadata.get('save_debug_html', False))
        if not enabled:
            return iterable
        return track(iterable, total=total, description=description)

    @staticmethod
    def _frame_ref_to_dict(frame: FrameRef) -> dict[str, Any]:
        return {'stream_id': frame.stream_id, 'frame_idx': int(frame.frame_idx), 'timestamp_sec': float(frame.timestamp_sec)}

    @staticmethod
    def _frame_ref_from_dict(raw: dict[str, Any]) -> FrameRef:
        return FrameRef(stream_id=str(raw['stream_id']), frame_idx=int(raw['frame_idx']), timestamp_sec=float(raw['timestamp_sec']))

    def _object_to_dict(self, obj: ObjectState) -> dict[str, Any]:
        det_id = obj.attributes.get('detection_id', obj.object_id)
        return {
            'object_id': obj.object_id,
            'frame_ref': self._frame_ref_to_dict(obj.frame_ref),
            'category': obj.category,
            'bbox_xyxy': np.asarray(obj.bbox_xyxy, dtype=float).tolist(),
            'mask_rle': obj.mask_rle,
            'embedding_present': obj.embedding is not None,
            'confidence': float(obj.confidence),
            'attributes': obj.attributes,
            'detection_id': det_id,
        }

    def _object_from_dict(self, raw: dict[str, Any], embedding_vectors: dict[str, np.ndarray]) -> ObjectState:
        det_id = raw.get('detection_id', raw['object_id'])
        embedding = embedding_vectors.get(det_id)
        return ObjectState(
            object_id=str(raw['object_id']),
            frame_ref=self._frame_ref_from_dict(raw['frame_ref']),
            category=str(raw['category']),
            bbox_xyxy=np.asarray(raw['bbox_xyxy'], dtype=np.float32),
            mask_rle=raw.get('mask_rle'),
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
            confidence=float(raw['confidence']),
            attributes=dict(raw.get('attributes', {})),
        )

    def _copy_debug_images(self, image_dir: Path) -> dict[str, str]:
        copied: dict[str, str] = {}
        source_map = self.metadata.get('debug_image_paths', {})
        if not source_map:
            return copied
        ensure_dir(image_dir)
        for frame_key, source in self._progress_iter(source_map.items(), total=len(source_map), description='[cyan]Objects · copying debug RGB'):
            src = Path(source)
            if not src.exists():
                continue
            dst = image_dir / f"{self._sanitize_key(frame_key)}{src.suffix or '.png'}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied[frame_key] = dst.name
        return copied

    def _render_overlay(self, frame_key: str, rgb: np.ndarray, frame_objects: list[ObjectState], *, trail_window: int = 4) -> np.ndarray:
        base = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        width, height = base.size

        # Precompute trajectories for current visible object ids.
        centers_by_id: dict[str, list[tuple[float, float]]] = {}
        for obj in self.objects:
            gid = obj.object_id
            centers_by_id.setdefault(gid, []).append((float((obj.bbox_xyxy[0] + obj.bbox_xyxy[2]) * 0.5), float((obj.bbox_xyxy[1] + obj.bbox_xyxy[3]) * 0.5), int(obj.frame_ref.frame_idx)))

        current_frame_idx = int(frame_objects[0].frame_ref.frame_idx) if frame_objects else 0
        for obj in frame_objects:
            gid = obj.object_id
            color = self._track_color(gid)
            rgb_color = ImageColor.getrgb(color)
            mask = self.mask_arrays.get(obj.attributes.get('detection_id', obj.object_id))
            if mask is not None and mask.shape[:2] == (height, width):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 90), mode='L')
                fill = Image.new('RGBA', base.size, (*rgb_color, 0))
                fill.putalpha(mask_img)
                overlay = Image.alpha_composite(overlay, fill)
                draw = ImageDraw.Draw(overlay, 'RGBA')
            x1, y1, x2, y2 = map(float, obj.bbox_xyxy.tolist())
            draw.rectangle([x1, y1, x2, y2], outline=(*rgb_color, 255), width=3)
            label = f"{gid} · {obj.category} · {obj.confidence:.2f}"
            draw.text((x1 + 4, max(4.0, y1 - 14.0)), label, fill=(*rgb_color, 255))
            history = [c for c in centers_by_id.get(gid, []) if current_frame_idx - trail_window <= c[2] <= current_frame_idx]
            history = sorted(history, key=lambda item: item[2])
            if len(history) >= 2:
                draw.line([(x, y) for x, y, _ in history], fill=(*rgb_color, 220), width=2)
        composed = Image.alpha_composite(base, overlay).convert('RGB')
        return np.asarray(composed, dtype=np.uint8)

    def _export_debug_viewer(self, copied_images: dict[str, str]) -> dict[str, Any]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception:
            return {}
        debug_dir = ensure_dir(self.save_dir / 'debug')
        overlay_dir = ensure_dir(debug_dir / 'overlays')
        frame_keys = sorted(copied_images.keys(), key=lambda key: (key.split(':')[0], int(key.split(':')[1])))
        if not frame_keys:
            return {}
        overlay_files: dict[str, str] = {}
        traces = []
        objects_by_frame: dict[str, list[ObjectState]] = {}
        for obj in self.objects:
            objects_by_frame.setdefault(self._frame_key(obj.frame_ref), []).append(obj)
        for frame_key in self._progress_iter(frame_keys, total=len(frame_keys), description='[cyan]Objects · building debug viewer'):
            rgb_path = self.save_dir / 'images' / copied_images[frame_key]
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            frame_objects = objects_by_frame.get(frame_key, [])
            overlay = self._render_overlay(frame_key, rgb, frame_objects)
            overlay_name = f"{self._sanitize_key(frame_key)}.png"
            Image.fromarray(overlay).save(overlay_dir / overlay_name)
            overlay_files[frame_key] = overlay_name
            categories = sorted({obj.category for obj in frame_objects})
            summary_rows = [
                ['Frame', frame_key],
                ['Detections', str(len(frame_objects))],
                ['Unique IDs', str(len({obj.object_id for obj in frame_objects}))],
                ['Categories', ', '.join(categories) if categories else 'None'],
            ]
            img_trace = go.Image(z=overlay, visible=False, name=f'overlay:{frame_key}')
            table_trace = go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in summary_rows], [row[1] for row in summary_rows]]),
                visible=False,
                name=f'summary:{frame_key}',
            )
            traces.extend([img_trace, table_trace])
        if traces:
            traces[0].visible = True
            if len(traces) > 1:
                traces[1].visible = True
        fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], specs=[[{'type': 'image'}, {'type': 'table'}]])
        for idx, trace in enumerate(traces):
            col = 1 if idx % 2 == 0 else 2
            fig.add_trace(trace, row=1, col=col)
        steps = []
        for frame_idx, frame_key in enumerate(frame_keys):
            visible = [False] * len(traces)
            visible[2 * frame_idx] = True
            visible[2 * frame_idx + 1] = True
            steps.append({
                'method': 'update',
                'label': frame_key,
                'args': [{'visible': visible}, {'title': f'Object perception viewer — {frame_key}'}],
            })
        fig.update_layout(
            title=f'Object perception viewer — {frame_keys[0]}',
            sliders=[{'steps': steps, 'active': 0}],
            margin=dict(l=20, r=20, t=60, b=20),
            height=720,
        )
        html_path = self.save_dir / 'object_debug_viewer.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        return {
            'overlay_files': overlay_files,
            'html': html_path.name,
        }

    def save(self) -> None:
        ensure_dir(self.save_dir)
        masks_dir = ensure_dir(self.save_dir / 'masks')
        emb_dir = ensure_dir(self.save_dir / 'embeddings')
        image_dir = ensure_dir(self.save_dir / 'images') if self.metadata.get('save_debug_assets', False) else None

        mask_index: dict[str, str] = {}
        for det_id, mask in self.mask_arrays.items():
            filename = f"{self._sanitize_key(det_id)}.npz"
            np.savez_compressed(masks_dir / filename, mask=np.asarray(mask, dtype=np.uint8))
            mask_index[det_id] = filename

        emb_index: dict[str, str] = {}
        for det_id, emb in self.embedding_vectors.items():
            filename = f"{self._sanitize_key(det_id)}.npz"
            np.savez_compressed(emb_dir / filename, embedding=np.asarray(emb, dtype=np.float32))
            emb_index[det_id] = filename

        copied_images = self._copy_debug_images(image_dir) if image_dir is not None else {}
        debug_exports = self._export_debug_viewer(copied_images) if self.metadata.get('save_debug_html', False) else {}

        payload = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'save_dir': str(self.save_dir),
            'metadata': dict(self.metadata),
            'objects': [self._object_to_dict(obj) for obj in self.objects],
            'linked_ids': self.linked_ids,
            'frame_index': self.frame_index or self._build_frame_index(self.objects),
            'mask_index': mask_index,
            'embedding_index': emb_index,
            'notes': list(self.notes),
            'debug_exports': debug_exports,
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> 'ObjectArtifact':
        raw = read_json(save_dir / 'artifact.json')
        mask_arrays: dict[str, np.ndarray] = {}
        for det_id, filename in raw.get('mask_index', {}).items():
            path = save_dir / 'masks' / filename
            if path.exists():
                with np.load(path, allow_pickle=True) as npz:
                    mask_arrays[det_id] = np.asarray(npz['mask'], dtype=bool)
        embedding_vectors: dict[str, np.ndarray] = {}
        for det_id, filename in raw.get('embedding_index', {}).items():
            path = save_dir / 'embeddings' / filename
            if path.exists():
                with np.load(path, allow_pickle=True) as npz:
                    embedding_vectors[det_id] = np.asarray(npz['embedding'], dtype=np.float32)
        artifact = cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(raw['save_dir']),
            metadata=raw.get('metadata', {}),
            objects=[],
            linked_ids=raw.get('linked_ids', {}),
            frame_index=raw.get('frame_index', {}),
            mask_arrays=mask_arrays,
            embedding_vectors=embedding_vectors,
            notes=raw.get('notes', []),
        )
        artifact.objects = [artifact._object_from_dict(item, embedding_vectors) for item in raw.get('objects', [])]
        return artifact

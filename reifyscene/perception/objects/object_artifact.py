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

from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.frame_keys import parse_key
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef, ObjectState


@dataclass
class ObjectArtifact(BaseArtifact):
    """Store object detections, masks, embeddings, and debug exports."""

    objects: list[ObjectState] = field(default_factory=list)
    raw_objects: list[ObjectState] = field(default_factory=list)
    linked_ids: dict[str, Any] = field(default_factory=dict)
    frame_index: dict[str, list[str]] = field(default_factory=dict)
    mask_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    embedding_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _frame_key(frame_ref: FrameRef) -> str:
        return make_frame_key(frame_ref.stream_id, frame_ref.frame_idx)

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
            'num_raw_objects': len(self.raw_objects),
            'notes': list(self.notes),
        }

    def _progress_iter(self, iterable, *, total: int | None = None, description: str = ''):
        enabled = bool(self.metadata.get('save_debug_assets', False) or self.metadata.get('save_debug_html', False))
        if not enabled:
            return iterable
        return track(iterable, total=total, description=description)

    @staticmethod
    def _frame_ref_to_dict(frame: FrameRef) -> dict[str, Any]:
        return frame.to_dict()

    @staticmethod
    def _frame_ref_from_dict(raw: dict[str, Any]) -> FrameRef:
        return FrameRef.from_dict(raw)

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
        det_id = raw['detection_id']
        # The stage files embeddings under the canonical track id (one per tracklet), not
        # the detection id; looking up only the detection id lost every one on reload.
        track_id = raw['attributes'].get('track_id')
        embedding = embedding_vectors.get(det_id)
        if embedding is None and track_id is not None:
            embedding = embedding_vectors.get(str(track_id))
        return ObjectState(
            object_id=str(raw['object_id']),
            frame_ref=self._frame_ref_from_dict(raw['frame_ref']),
            category=str(raw['category']),
            bbox_xyxy=np.asarray(raw['bbox_xyxy'], dtype=np.float32),
            mask_rle=raw['mask_rle'],
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
            confidence=float(raw['confidence']),
            attributes=dict(raw['attributes']),
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

    @staticmethod
    def _center(obj: ObjectState) -> tuple[float, float]:
        return float((obj.bbox_xyxy[0] + obj.bbox_xyxy[2]) * 0.5), float((obj.bbox_xyxy[1] + obj.bbox_xyxy[3]) * 0.5)

    @staticmethod
    def _marker_for_label(label: int | None) -> str:
        if label == 1:
            return '●'
        if label == 0:
            return '✕'
        return '◇'

    def _render_overlay(
        self,
        frame_key: str,
        rgb: np.ndarray,
        frame_objects: list[ObjectState],
        raw_frame_objects: list[ObjectState],
        *,
        trail_window: int = 6,
    ) -> np.ndarray:
        base = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        width, height = base.size

        centers_by_id: dict[str, list[tuple[float, float, int]]] = {}
        for obj in self.objects:
            gid = obj.object_id
            cx, cy = self._center(obj)
            centers_by_id.setdefault(gid, []).append((cx, cy, int(obj.frame_ref.frame_idx)))

        current_frame_idx = int(frame_objects[0].frame_ref.frame_idx) if frame_objects else (int(raw_frame_objects[0].frame_ref.frame_idx) if raw_frame_objects else 0)

        for obj in frame_objects:
            gid = obj.object_id
            color = self._track_color(gid)
            rgb_color = ImageColor.getrgb(color)
            det_id = obj.attributes.get('detection_id', obj.object_id)
            mask = self.mask_arrays.get(det_id)
            if mask is not None and mask.shape[:2] == (height, width):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 90), mode='L')
                fill = Image.new('RGBA', base.size, (*rgb_color, 0))
                fill.putalpha(mask_img)
                overlay = Image.alpha_composite(overlay, fill)
                draw = ImageDraw.Draw(overlay, 'RGBA')
            x1, y1, x2, y2 = map(float, obj.bbox_xyxy.tolist())
            draw.rectangle([x1, y1, x2, y2], outline=(*rgb_color, 255), width=3)
            consensus = obj.attributes.get('consensus_category', obj.category)
            current = obj.attributes.get('current_category', consensus)
            support_count = int(obj.attributes.get('support_track_count', len(obj.attributes.get('track_ids', []))))
            entropy = float(obj.attributes.get('category_entropy', 0.0))
            quality = float(obj.attributes.get('quality_score', 0.0))
            life = f"{obj.attributes.get('lifespan_start', current_frame_idx)}-{obj.attributes.get('lifespan_end', current_frame_idx)}"
            label = f"{gid} {consensus}"
            if current != consensus:
                label += f" (cur:{current})"
            label += f" trk:{support_count} H:{entropy:.2f} q:{quality:.2f} life:{life}"
            draw.text((x1 + 4, max(4.0, y1 - 14.0)), label, fill=(*rgb_color, 255))
            for xy in np.asarray(obj.attributes.get('support_points_xy', []), dtype=np.float32)[:64]:
                sx, sy = float(xy[0]), float(xy[1])
                draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=(*rgb_color, 220), outline=(255, 255, 255, 255))
            history = [c for c in centers_by_id.get(gid, []) if current_frame_idx - trail_window <= c[2] <= current_frame_idx]
            history = sorted(history, key=lambda item: item[2])
            if len(history) >= 2:
                draw.line([(x, y) for x, y, _ in history], fill=(*rgb_color, 220), width=2)
            if history:
                x, y, _ = history[-1]
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(*rgb_color, 255), outline=(255, 255, 255, 255))

        composed = Image.alpha_composite(base, overlay).convert('RGB')
        return np.asarray(composed, dtype=np.uint8)

    def _build_summary_rows(self, frame_key: str, frame_objects: list[ObjectState], raw_frame_objects: list[ObjectState]) -> list[list[str]]:
        track_summaries = self.linked_ids.get('track_summaries', {})
        visible_ids = sorted({obj.object_id for obj in frame_objects})
        rows = [
            ['Frame', frame_key],
            ['Raw detections', str(len(raw_frame_objects))],
            ['Kept detections', str(len(frame_objects))],
            ['Persistent IDs', str(len(visible_ids))],
        ]
        if visible_ids:
            for object_id in visible_ids[:8]:
                objs = [obj for obj in frame_objects if obj.object_id == object_id]
                obj = objs[0]
                summary = track_summaries.get(object_id, {})
                consensus = obj.attributes.get('consensus_category', obj.category)
                current = obj.attributes.get('current_category', consensus)
                support = int(summary.get('support_track_count', obj.attributes.get('support_track_count', 0)))
                entropy = float(summary.get('category_entropy', obj.attributes.get('category_entropy', 0.0)))
                quality = float(summary.get('quality_score', obj.attributes.get('quality_score', 0.0)))
                life = f"{summary.get('lifespan_start', obj.attributes.get('lifespan_start', 0))}-{summary.get('lifespan_end', obj.attributes.get('lifespan_end', obj.frame_ref.frame_idx))}"
                label = consensus if current == consensus else f"{consensus} / cur:{current}"
                rows.append([object_id, f"{label} · trk {support} · H {entropy:.2f} · q {quality:.2f} · life {life}"])
        return rows

    def _timeline_figure(self, frame_keys: list[str]):
        import plotly.graph_objects as go
        track_summaries = self.linked_ids.get('track_summaries', {})
        if not track_summaries:
            return [], []
        sorted_ids = sorted(track_summaries.keys(), key=lambda gid: (track_summaries[gid].get('lifespan_start', 0), gid))
        traces = []
        y_labels = []
        for idx, gid in enumerate(sorted_ids):
            summary = track_summaries[gid]
            y_labels.append(gid)
            color = self._track_color(gid)
            start = summary.get('lifespan_start', 0)
            end = summary.get('lifespan_end', start)
            consensus = summary.get('consensus_category', 'unknown')
            support = summary.get('support_track_count', 0)
            entropy = float(summary.get('category_entropy', 0.0))
            quality = float(summary.get('quality_score', 0.0))
            traces.append(
                go.Scatter(
                    x=[start, end],
                    y=[idx, idx],
                    mode='lines+markers',
                    line=dict(color=color, width=4),
                    marker=dict(color=color, size=8),
                    name=gid,
                    hovertemplate=(
                        f"{gid}<br>consensus={consensus}<br>support_tracks={support}<br>"
                        f"entropy={entropy:.2f}<br>quality={quality:.2f}<br>lifespan={start}-{end}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
        return traces, y_labels

    def _export_debug_viewer(self, copied_images: dict[str, str]) -> dict[str, Any]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception:
            return {}
        debug_dir = ensure_dir(self.save_dir / 'debug')
        overlay_dir = ensure_dir(debug_dir / 'overlays')
        frame_keys = sorted(copied_images.keys(), key=parse_key)
        if not frame_keys:
            return {}
        overlay_files: dict[str, str] = {}
        traces = []
        objects_by_frame: dict[str, list[ObjectState]] = {}
        raw_by_frame: dict[str, list[ObjectState]] = {}
        for obj in self.objects:
            objects_by_frame.setdefault(self._frame_key(obj.frame_ref), []).append(obj)
        for obj in self.raw_objects:
            raw_by_frame.setdefault(self._frame_key(obj.frame_ref), []).append(obj)
        timeline_traces, timeline_labels = self._timeline_figure(frame_keys)
        for frame_key in self._progress_iter(frame_keys, total=len(frame_keys), description='[cyan]Objects · building debug viewer'):
            rgb_path = self.save_dir / 'images' / copied_images[frame_key]
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            frame_objects = objects_by_frame.get(frame_key, [])
            raw_frame_objects = raw_by_frame.get(frame_key, frame_objects)
            overlay = self._render_overlay(frame_key, rgb, frame_objects, raw_frame_objects)
            overlay_name = f"{self._sanitize_key(frame_key)}.png"
            Image.fromarray(overlay).save(overlay_dir / overlay_name)
            overlay_files[frame_key] = overlay_name
            summary_rows = self._build_summary_rows(frame_key, frame_objects, raw_frame_objects)
            img_trace = go.Image(z=overlay, visible=False, name=f'overlay:{frame_key}')
            table_trace = go.Table(
                header=dict(values=['Metric / Object', 'Value']),
                cells=dict(values=[[row[0] for row in summary_rows], [row[1] for row in summary_rows]]),
                visible=False,
                name=f'summary:{frame_key}',
            )
            traces.extend([img_trace, table_trace])
        if traces:
            traces[0].visible = True
            if len(traces) > 1:
                traces[1].visible = True
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.62, 0.38],
            specs=[[{'type': 'image', 'rowspan': 2}, {'type': 'table'}], [None, {'type': 'xy'}]],
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            subplot_titles=('Object overlay', 'Summary', '', 'Track timeline'),
        )
        for idx, trace in enumerate(traces):
            col = 1 if idx % 2 == 0 else 2
            row = 1 if idx % 2 == 0 else 1
            fig.add_trace(trace, row=row, col=col)
        start_idx = len(traces)
        for tr in timeline_traces:
            fig.add_trace(tr, row=2, col=2)
        frame_numbers = [parse_key(key)[1] for key in frame_keys]
        current_frame = frame_numbers[0]
        fig.update_yaxes(title_text='Persistent object', tickmode='array', tickvals=list(range(len(timeline_labels))), ticktext=timeline_labels, row=2, col=2)
        fig.update_xaxes(title_text='Frame index', row=2, col=2)
        fig.update_layout(
            title=f'Object perception viewer — {frame_keys[0]}',
            margin=dict(l=20, r=20, t=70, b=20),
            height=860,
            shapes=[{
                'type': 'line', 'x0': current_frame, 'x1': current_frame, 'y0': -0.5, 'y1': max(len(timeline_labels) - 0.5, 0.5),
                'xref': 'x2', 'yref': 'y2', 'line': {'color': 'black', 'width': 2, 'dash': 'dash'},
            }],
        )
        steps = []
        total_traces = len(traces) + len(timeline_traces)
        for frame_idx, frame_key in enumerate(frame_keys):
            visible = [False] * total_traces
            visible[2 * frame_idx] = True
            visible[2 * frame_idx + 1] = True
            for tidx in range(len(timeline_traces)):
                visible[start_idx + tidx] = True
            current_frame = parse_key(frame_key)[1]
            steps.append({
                'method': 'update',
                'label': frame_key,
                'args': [
                    {'visible': visible},
                    {
                        'title': f'Object perception viewer — {frame_key}',
                        'shapes': [{
                            'type': 'line', 'x0': current_frame, 'x1': current_frame, 'y0': -0.5, 'y1': max(len(timeline_labels) - 0.5, 0.5),
                            'xref': 'x2', 'yref': 'y2', 'line': {'color': 'black', 'width': 2, 'dash': 'dash'},
                        }],
                    },
                ],
            })
        fig.update_layout(sliders=[{'steps': steps, 'active': 0}])
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
            'metadata': dict(self.metadata),
            'objects': [self._object_to_dict(obj) for obj in self.objects],
            'raw_objects': [self._object_to_dict(obj) for obj in self.raw_objects],
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
        raw = read_json(save_dir / cls.FILE)
        mask_arrays: dict[str, np.ndarray] = {}
        for det_id, filename in raw['mask_index'].items():
            path = require_file(save_dir / 'masks' / filename, "objects file")
            with np.load(path, allow_pickle=True) as npz:
                mask_arrays[det_id] = np.asarray(npz['mask'], dtype=bool)
        embedding_vectors: dict[str, np.ndarray] = {}
        for det_id, filename in raw['embedding_index'].items():
            path = require_file(save_dir / 'embeddings' / filename, "objects file")
            with np.load(path, allow_pickle=True) as npz:
                embedding_vectors[det_id] = np.asarray(npz['embedding'], dtype=np.float32)
        artifact = cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw['metadata'],
            objects=[],
            raw_objects=[],
            linked_ids=raw['linked_ids'],
            frame_index=raw['frame_index'],
            mask_arrays=mask_arrays,
            embedding_vectors=embedding_vectors,
            notes=raw['notes'],
        )
        artifact.objects = [artifact._object_from_dict(item, embedding_vectors) for item in raw['objects']]
        artifact.raw_objects = [artifact._object_from_dict(item, embedding_vectors) for item in raw['raw_objects']]
        return artifact

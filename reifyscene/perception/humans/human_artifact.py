"""Artifact for human detections, DensePose, SMPL, and association."""

from __future__ import annotations

import colorsys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from rich.progress import track

from reifyscene.common.backend_ladder import CapabilityManifest
from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.frame_keys import parse_key
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef, HumanState

COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


@dataclass
class HumanArtifact(BaseArtifact):
    """Store human detections, masks, densepose, Sapiens dense outputs, and debug exports."""

    humans: list[HumanState] = field(default_factory=list)
    raw_humans: list[HumanState] = field(default_factory=list)
    linked_ids: dict[str, Any] = field(default_factory=dict)
    frame_index: dict[str, list[str]] = field(default_factory=dict)
    mask_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    densepose_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    smpl_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    dense_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)  # Sapiens seg/depth/normals
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
    def _build_frame_index(cls, humans: list[HumanState]) -> dict[str, list[str]]:
        frame_index: dict[str, list[str]] = {}
        for h in humans:
            frame_index.setdefault(cls._frame_key(h.frame_ref), []).append(h.human_id)
        return frame_index

    def attach_manifest(self, manifest: CapabilityManifest) -> None:
        self.metadata['capability'] = manifest.to_dict()

    @property
    def capability(self) -> 'CapabilityManifest | None':
        raw = self.metadata.get('capability')
        return CapabilityManifest.from_dict(raw) if raw else None

    def summary(self) -> dict[str, Any]:
        unique_ids = sorted({h.human_id for h in self.humans})
        return {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'num_humans': len(self.humans),
            'num_unique_ids': len(unique_ids),
            'num_frames': len(self.frame_index or self._build_frame_index(self.humans)),
            'num_raw_humans': len(self.raw_humans),
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

    def _human_to_dict(self, h: HumanState) -> dict[str, Any]:
        det_id = h.attributes.get('detection_id', h.human_id)
        return {
            'human_id': h.human_id,
            'frame_ref': self._frame_ref_to_dict(h.frame_ref),
            'bbox_xyxy': np.asarray(h.bbox_xyxy, dtype=float).tolist(),
            'mask_rle': h.mask_rle,
            'keypoints_2d': None if h.keypoints_2d is None else np.asarray(h.keypoints_2d, dtype=float).tolist(),
            'densepose_present': h.densepose is not None,
            'smpl_present': h.smpl_params is not None,
            'dense_present': h.dense is not None,
            'confidence': float(h.confidence),
            'attributes': h.attributes,
            'detection_id': det_id,
        }

    def _human_from_dict(self, raw: dict[str, Any]) -> HumanState:
        det_id = raw['detection_id']
        return HumanState(
            human_id=str(raw['human_id']),
            frame_ref=self._frame_ref_from_dict(raw['frame_ref']),
            bbox_xyxy=np.asarray(raw['bbox_xyxy'], dtype=np.float32),
            mask_rle=raw['mask_rle'],
            keypoints_2d=None if raw['keypoints_2d'] is None else np.asarray(raw['keypoints_2d'], dtype=np.float32),
            densepose=self.densepose_payloads.get(det_id),
            smpl_params=self.smpl_payloads.get(det_id),
            confidence=float(raw['confidence']),
            attributes=dict(raw['attributes']),
            dense=self.dense_payloads.get(det_id),
        )

    def _copy_debug_images(self, image_dir: Path) -> dict[str, str]:
        copied: dict[str, str] = {}
        source_map = self.metadata.get('debug_image_paths', {})
        if not source_map:
            return copied
        ensure_dir(image_dir)
        for frame_key, source in self._progress_iter(source_map.items(), total=len(source_map), description='[cyan]Humans · copying debug RGB'):
            src = Path(source)
            if not src.exists():
                continue
            dst = image_dir / f"{self._sanitize_key(frame_key)}{src.suffix or '.png'}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied[frame_key] = dst.name
        return copied

    def _draw_skeleton(self, draw: ImageDraw.ImageDraw, keypoints: np.ndarray, color_rgb: tuple[int, int, int]) -> None:
        for i, j in COCO_SKELETON:
            if i >= keypoints.shape[0] or j >= keypoints.shape[0]:
                continue
            if float(keypoints[i, 2]) <= 0.05 or float(keypoints[j, 2]) <= 0.05:
                continue
            draw.line([tuple(keypoints[i, :2]), tuple(keypoints[j, :2])], fill=(*color_rgb, 255), width=2)
        for pt in keypoints:
            if float(pt[2]) <= 0.05:
                continue
            x, y = float(pt[0]), float(pt[1])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(*color_rgb, 255), outline=(255, 255, 255, 255))

    def _render_overlay(self, frame_key: str, rgb: np.ndarray, frame_humans: list[HumanState], *, raw_frame_humans: list[HumanState] | None = None, mode: str = 'cleaned', trail_window: int = 6, show_support: bool = False) -> np.ndarray:
        base = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        width, height = base.size
        centers_by_id: dict[str, list[tuple[float, float, int]]] = {}
        source = self.humans if mode == 'cleaned' else (raw_frame_humans or [])
        for h in (self.humans if mode == 'cleaned' else self.raw_humans):
            x1, y1, x2, y2 = map(float, h.bbox_xyxy.tolist())
            centers_by_id.setdefault(h.human_id, []).append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, int(h.frame_ref.frame_idx)))
        current_frame_idx = int(frame_humans[0].frame_ref.frame_idx) if frame_humans else (int(raw_frame_humans[0].frame_ref.frame_idx) if raw_frame_humans else 0)
        active = frame_humans if mode == 'cleaned' else (raw_frame_humans or [])
        for h in active:
            color = self._track_color(h.human_id if mode == 'cleaned' else h.attributes.get('detection_id', h.human_id))
            rgb_color = ImageColor.getrgb(color)
            det_id = h.attributes.get('detection_id', h.human_id)
            mask = self.mask_arrays.get(det_id)
            if mask is not None and mask.shape[:2] == (height, width):
                alpha = 70 if mode == 'cleaned' else 40
                mask_img = Image.fromarray((mask.astype(np.uint8) * alpha), mode='L')
                fill = Image.new('RGBA', base.size, (*rgb_color, 0))
                fill.putalpha(mask_img)
                overlay = Image.alpha_composite(overlay, fill)
                draw = ImageDraw.Draw(overlay, 'RGBA')
            x1, y1, x2, y2 = map(float, h.bbox_xyxy.tolist())
            line_color = (*rgb_color, 255)
            draw.rectangle([x1, y1, x2, y2], outline=line_color, width=3)
            if h.keypoints_2d is not None:
                self._draw_skeleton(draw, np.asarray(h.keypoints_2d, dtype=np.float32), rgb_color)
            dp = h.densepose or {}
            samples_xy = np.asarray(dp.get('samples_xy', np.zeros((0, 2))), dtype=np.float32)
            part_ids = np.asarray(dp.get('part_ids', np.zeros((samples_xy.shape[0],), dtype=np.int32)), dtype=np.int32)
            for part in np.unique(part_ids)[:6]:
                pts = samples_xy[part_ids == part]
                for xy in pts[:32]:
                    x, y = float(xy[0]), float(xy[1])
                    draw.point((x, y), fill=(*rgb_color, 180))
            if show_support:
                for xy in np.asarray(h.attributes.get('support_points_xy', []), dtype=np.float32)[:96]:
                    x, y = float(xy[0]), float(xy[1])
                    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255, 230), outline=(*rgb_color, 255))
            hist = self.linked_ids.get('track_summaries', {}).get(h.human_id, {})
            support = int(hist.get('support_track_count', h.attributes.get('support_track_count', len(h.attributes.get('track_ids', [])))))
            density = float(hist.get('mean_support_density', h.attributes.get('support_ratio', 0.0)))
            quality = float(hist.get('quality_score', h.attributes.get('quality_score', 0.0)))
            life = f"{hist.get('lifespan_start', current_frame_idx)}-{hist.get('lifespan_end', current_frame_idx)}"
            visible = int(np.sum(np.asarray(h.keypoints_2d)[:, 2] > 0.05)) if h.keypoints_2d is not None else 0
            label = f"{h.human_id} kpt:{visible} trk:{support} d:{density:.2f} q:{quality:.2f} life:{life}"
            draw.text((x1 + 4, max(4.0, y1 - 14.0)), label, fill=(*rgb_color, 255))
            history = [c for c in centers_by_id.get(h.human_id, []) if current_frame_idx - trail_window <= c[2] <= current_frame_idx]
            history = sorted(history, key=lambda item: item[2])
            if len(history) >= 2:
                draw.line([(x, y) for x, y, _ in history], fill=(*rgb_color, 220), width=2)
        composed = Image.alpha_composite(base, overlay).convert('RGB')
        return np.asarray(composed, dtype=np.uint8)

    def _build_summary_rows(self, frame_key: str, frame_humans: list[HumanState], raw_frame_humans: list[HumanState]) -> list[list[str]]:
        rows = [
            ['Frame', frame_key],
            ['Raw detections', str(len(raw_frame_humans))],
            ['Kept humans', str(len(frame_humans))],
            ['Persistent IDs', str(len({h.human_id for h in self.humans}))],
        ]
        track_summaries = self.linked_ids.get('track_summaries', {})
        for h in frame_humans[:8]:
            summary = track_summaries.get(h.human_id, {})
            visible = int(np.sum(np.asarray(h.keypoints_2d)[:, 2] > 0.05)) if h.keypoints_2d is not None else 0
            dp_count = int(np.asarray((h.densepose or {}).get('samples_xy', np.zeros((0, 2)))).shape[0])
            smpl_valid = bool((h.smpl_params or {}).get('valid', False))
            support = int(summary.get('support_track_count', len(h.attributes.get('track_ids', []))))
            density = float(summary.get('mean_support_density', h.attributes.get('support_ratio', 0.0)))
            quality = float(summary.get('quality_score', h.attributes.get('quality_score', 0.0)))
            life = f"{summary.get('lifespan_start', h.frame_ref.frame_idx)}-{summary.get('lifespan_end', h.frame_ref.frame_idx)}"
            rows.append([h.human_id, f"kpt {visible} · dp {dp_count} · smpl {int(smpl_valid)} · trk {support} · d {density:.2f} · q {quality:.2f} · life {life}"])
        return rows

    def _timeline_figure(self):
        track_summaries = self.linked_ids.get('track_summaries', {})
        if not track_summaries:
            return [], []
        traces = []
        y_labels = []
        for idx, hid in enumerate(sorted(track_summaries.keys(), key=lambda x: (track_summaries[x].get('lifespan_start', 0), x))):
            summary = track_summaries[hid]
            start = summary.get('lifespan_start', 0)
            end = summary.get('lifespan_end', start)
            color = self._track_color(hid)
            y_labels.append(hid)
            traces.append({
                'type': 'scatter',
                'x': [start, end],
                'y': [idx, idx],
                'mode': 'lines+markers',
                'line': {'color': color, 'width': 6},
                'marker': {'color': color, 'size': 8},
                'name': hid,
                'hovertemplate': (
                    f"{hid}<br>tracks={summary.get('support_track_count', 0)}<br>"
                    f"density={summary.get('mean_support_density', 0.0):.2f}<br>"
                    f"quality={summary.get('quality_score', 0.0):.2f}<br>"
                    f"lifespan={start}-{end}<extra></extra>"
                ),
                'showlegend': False,
            })
        return traces, y_labels

    def _export_debug_viewer(self, copied_images: dict[str, str]) -> dict[str, Any]:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        frame_keys = sorted((self.frame_index or self._build_frame_index(self.humans)).keys(), key=parse_key)
        if not frame_keys or not copied_images:
            return {}
        images_dir = self.save_dir / 'images'
        html_path = self.save_dir / 'human_debug_viewer.html'
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'type': 'xy'}, {'type': 'table'}], [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=('RGB overlay', 'Summary', 'Support / DensePose snapshot', 'Track timeline'),
            row_heights=[0.72, 0.28],
            column_widths=[0.7, 0.3],
        )
        timeline_traces, timeline_labels = self._timeline_figure()
        base_trace_count = 0
        for trace in timeline_traces:
            fig.add_trace(go.Scatter(**trace, visible=True), row=2, col=2)
            base_trace_count += 1
        raw_trace_indices: list[int] = []
        cleaned_trace_indices: list[int] = []
        support_trace_indices: list[int] = []
        traces_per_frame = 5
        for idx, frame_key in enumerate(self._progress_iter(frame_keys, total=len(frame_keys), description='[cyan]Humans · building debug viewer')):
            frame_humans = [h for h in self.humans if self._frame_key(h.frame_ref) == frame_key]
            raw_frame_humans = [h for h in self.raw_humans if self._frame_key(h.frame_ref) == frame_key]
            img_name = copied_images.get(frame_key)
            if img_name is None:
                continue
            rgb = np.asarray(Image.open(images_dir / img_name).convert('RGB'))
            cleaned_overlay = self._render_overlay(frame_key, rgb, frame_humans, raw_frame_humans=raw_frame_humans, mode='cleaned')
            raw_overlay = self._render_overlay(frame_key, rgb, frame_humans, raw_frame_humans=raw_frame_humans, mode='raw')
            support_preview = self._render_overlay(frame_key, rgb, frame_humans, raw_frame_humans=raw_frame_humans, mode='cleaned', show_support=True)
            summary_rows = self._build_summary_rows(frame_key, frame_humans, raw_frame_humans)
            vis = (idx == 0)
            fig.add_trace(go.Image(z=raw_overlay, visible=vis, opacity=0.0, name=f'raw:{frame_key}'), row=1, col=1)
            raw_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(go.Image(z=cleaned_overlay, visible=vis, opacity=1.0, name=f'cleaned:{frame_key}'), row=1, col=1)
            cleaned_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Table(
                    header=dict(values=['Field', 'Value']),
                    cells=dict(values=[[r[0] for r in summary_rows], [r[1] for r in summary_rows]]),
                    visible=vis,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(go.Image(z=support_preview, visible=vis, name=f'support:{frame_key}'), row=2, col=1)
            support_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scatter(
                    x=[parse_key(frame_key)[1]],
                    y=[0],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='line-ns-open'),
                    visible=vis,
                    showlegend=False,
                    hovertemplate='current frame<extra></extra>',
                ),
                row=2,
                col=2,
            )
        image_trace_start = base_trace_count
        steps = []
        total_traces = len(fig.data)
        num_frames = (total_traces - base_trace_count) // traces_per_frame if total_traces >= base_trace_count else 0
        for idx, frame_key in enumerate(frame_keys[:num_frames]):
            visible = [True] * base_trace_count + [False] * (total_traces - base_trace_count)
            start = image_trace_start + idx * traces_per_frame
            for j in range(traces_per_frame):
                if start + j < total_traces:
                    visible[start + j] = True
            steps.append({'method': 'update', 'label': frame_key, 'args': [{'visible': visible}, {'title': f'reifyscene human debug viewer — {frame_key}'}]})
        if steps:
            fig.update_layout(sliders=[{'active': 0, 'steps': steps, 'currentvalue': {'prefix': 'Frame: '}}])
        raw_opacity = [0.0] * total_traces
        cleaned_opacity = [1.0] * total_traces
        both_opacity = [1.0] * total_traces
        for idx in raw_trace_indices:
            raw_opacity[idx] = 1.0
            cleaned_opacity[idx] = 0.0
            both_opacity[idx] = 0.5
        for idx in cleaned_trace_indices:
            raw_opacity[idx] = 0.0
            cleaned_opacity[idx] = 1.0
            both_opacity[idx] = 0.85
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {'label': 'Cleaned', 'method': 'restyle', 'args': [{'opacity': cleaned_opacity}]},
                    {'label': 'Raw', 'method': 'restyle', 'args': [{'opacity': raw_opacity}]},
                    {'label': 'Both', 'method': 'restyle', 'args': [{'opacity': both_opacity}]},
                ],
                'direction': 'right',
                'x': 0.01,
                'y': 1.08,
            }]
        )
        fig.update_yaxes(autorange='reversed', visible=False, row=1, col=1)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(autorange='reversed', visible=False, row=2, col=1)
        fig.update_xaxes(visible=False, row=2, col=1)
        if timeline_labels:
            fig.update_yaxes(tickmode='array', tickvals=list(range(len(timeline_labels))), ticktext=timeline_labels, row=2, col=2)
        fig.update_layout(title='reifyscene human debug viewer', height=900, width=1400, margin=dict(l=20, r=20, t=80, b=20))
        fig.write_html(html_path, include_plotlyjs='cdn')
        return {'human_debug_viewer': str(html_path)}

    def save(self) -> None:
        ensure_dir(self.save_dir)
        masks_dir = ensure_dir(self.save_dir / 'masks')
        densepose_dir = ensure_dir(self.save_dir / 'densepose')
        smpl_dir = ensure_dir(self.save_dir / 'smpl')
        images_dir = self.save_dir / 'images'

        for det_id, mask in self.mask_arrays.items():
            np.savez_compressed(masks_dir / f"{self._sanitize_key(det_id)}.npz", mask=np.asarray(mask, dtype=bool))
        for det_id, payload in self.densepose_payloads.items():
            save_payload = {}
            for key, value in payload.items():
                if isinstance(value, np.ndarray):
                    save_payload[key] = value
                else:
                    save_payload[key] = np.asarray(value) if isinstance(value, (list, tuple)) else np.array(value, dtype=object)
            np.savez_compressed(densepose_dir / f"{self._sanitize_key(det_id)}.npz", **save_payload)
        for det_id, payload in self.smpl_payloads.items():
            save_payload = {}
            for key, value in payload.items():
                if isinstance(value, np.ndarray):
                    save_payload[key] = value
                elif isinstance(value, (list, tuple)):
                    save_payload[key] = np.asarray(value)
                elif isinstance(value, (bool, int, float)):
                    save_payload[key] = np.asarray(value)
                else:
                    save_payload[key] = np.array(value, dtype=object)
            np.savez_compressed(smpl_dir / f"{self._sanitize_key(det_id)}.npz", **save_payload)
        dense_dir = ensure_dir(self.save_dir / 'dense')
        for det_id, payload in self.dense_payloads.items():
            save_payload = {k: np.asarray(v) for k, v in payload.items() if v is not None}
            np.savez_compressed(dense_dir / f"{self._sanitize_key(det_id)}.npz", **save_payload)

        copied_images = self._copy_debug_images(images_dir) if (self.metadata.get('save_debug_assets') or self.metadata.get('save_debug_html')) else {}
        if copied_images:
            self.metadata['debug_image_paths'] = copied_images
        if self.metadata.get('save_debug_html') and copied_images:
            exports = self._export_debug_viewer(copied_images)
            self.metadata.update(exports)

        payload = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'metadata': self.metadata,
            'humans': [self._human_to_dict(h) for h in self.humans],
            'raw_humans': [self._human_to_dict(h) for h in self.raw_humans],
            'linked_ids': self.linked_ids,
            'frame_index': self.frame_index or self._build_frame_index(self.humans),
            'mask_ids': sorted(self.mask_arrays.keys()),
            'densepose_ids': sorted(self.densepose_payloads.keys()),
            'smpl_ids': sorted(self.smpl_payloads.keys()),
            'dense_ids': sorted(self.dense_payloads.keys()),
            'notes': self.notes,
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> 'HumanArtifact':
        raw = read_json(save_dir / cls.FILE)
        mask_arrays: dict[str, np.ndarray] = {}
        for det_id in raw['mask_ids']:
            path = require_file(save_dir / 'masks' / f"{cls._sanitize_key(det_id)}.npz", "humans file")
            with np.load(path, allow_pickle=True) as npz:
                mask_arrays[det_id] = np.asarray(npz['mask'], dtype=bool)
        densepose_payloads: dict[str, dict[str, Any]] = {}
        for det_id in raw['densepose_ids']:
            path = require_file(save_dir / 'densepose' / f"{cls._sanitize_key(det_id)}.npz", "humans file")
            with np.load(path, allow_pickle=True) as npz:
                densepose_payloads[det_id] = {k: np.asarray(npz[k]) for k in npz.files}
        smpl_payloads: dict[str, dict[str, Any]] = {}
        for det_id in raw['smpl_ids']:
            path = require_file(save_dir / 'smpl' / f"{cls._sanitize_key(det_id)}.npz", "humans file")
            with np.load(path, allow_pickle=True) as npz:
                payload = {}
                for k in npz.files:
                    arr = np.asarray(npz[k])
                    payload[k] = arr.item() if arr.ndim == 0 and arr.dtype == object else arr
                smpl_payloads[det_id] = payload
        dense_payloads: dict[str, dict[str, Any]] = {}
        for det_id in raw['dense_ids']:
            path = require_file(save_dir / 'dense' / f"{cls._sanitize_key(det_id)}.npz", "humans file")
            with np.load(path, allow_pickle=True) as npz:
                dense_payloads[det_id] = {k: np.asarray(npz[k]) for k in npz.files}
        artifact = cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw['metadata'],
            humans=[],
            raw_humans=[],
            linked_ids=raw['linked_ids'],
            frame_index=raw['frame_index'],
            mask_arrays=mask_arrays,
            densepose_payloads=densepose_payloads,
            smpl_payloads=smpl_payloads,
            dense_payloads=dense_payloads,
            notes=list(raw['notes']),
        )
        artifact.humans = [artifact._human_from_dict(item) for item in raw['humans']]
        artifact.raw_humans = [artifact._human_from_dict(item) for item in raw['raw_humans']]
        return artifact

"""Wrapper module for a real or heuristic CoTracker-style point tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.progress import track

from room4d.common.types import FrameRef, TrackObservation, VideoStreamRef
from room4d.ingest.decoder import VideoDecoder
from room4d.tracking.base import PointTracker
from room4d.tracking.track_artifact import TrackArtifact


class CoTrackerWrapper(PointTracker):
    """Foundation-model wrapper for point tracking.

    The wrapper prefers a real CoTracker backend via ``torch.hub`` when
    available. When the backend cannot be loaded and ``backend_mode='auto'``,
    it falls back to a deterministic grid tracker so downstream stages remain
    usable.

    Notes
    -----
    Windowed execution is supported to reduce peak memory usage. In windowed
    mode, each window seeds and tracks its own grid of points. Track IDs are
    therefore unique across windows but are **not** continuous identities across
    the full clip.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get('default_fps', 30.0)))
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cuda'))
        self.model_name = str(self.cfg.get('model_name', 'cotracker3_offline'))
        self.grid_size = int(self.cfg.get('grid_size', 12))
        self.frame_stride = max(int(self.cfg.get('frame_stride', 1)), 1)
        self.max_frames = self.cfg.get('max_frames')
        self.resize_max_dim = self.cfg.get('resize_max_dim')
        self.window_size = self.cfg.get('window_size')
        self.window_overlap = max(int(self.cfg.get('window_overlap', 0)), 0)
        self.use_autocast = bool(self.cfg.get('use_autocast', False))
        self.visibility_threshold = float(self.cfg.get('visibility_threshold', 0.5))
        self.cpu_threads = self.cfg.get('cpu_threads')
        self._model = None
        self._torch = None
        self._is_loaded = False
        self._backend_error: Exception | None = None
        self.debug = bool(self.cfg.get('debug', False))

        if self.cpu_threads is not None:
            try:
                cv2.setNumThreads(max(1, int(self.cpu_threads)))
            except Exception:
                pass

    def _repo_or_dir(self) -> tuple[str, str]:
        local_repo = self.cfg.get('torch_hub_repo_or_dir')
        if local_repo:
            return str(local_repo), 'local'
        root = Path(__file__).resolve().parents[2]
        for candidate in (root / 'third_party' / 'co-tracker', root / 'third_party' / 'cotracker'):
            if (candidate / 'hubconf.py').exists():
                return str(candidate), 'local'
        if bool(self.cfg.get('allow_github', self.backend_mode == 'real')):
            return 'facebookresearch/co-tracker', 'github'
        raise RuntimeError('No local CoTracker checkout found. Clone it under third_party/ or set allow_github=true.')

    def _load_model(self) -> None:
        if self._is_loaded or self.backend_mode == 'heuristic':
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover - import environment specific
            self._backend_error = exc
            if self.backend_mode == 'real':
                raise
            return

        if self.cpu_threads is not None and not self.device.startswith('cuda'):
            try:
                torch.set_num_threads(max(1, int(self.cpu_threads)))
            except Exception:
                pass
            try:
                interop_threads = max(1, int(self.cpu_threads) // 2)
                torch.set_num_interop_threads(interop_threads)
            except Exception:
                pass

        repo_or_dir, source = self._repo_or_dir()
        try:
            if self.debug:
                print(f"[TRACK DEBUG] Loading CoTracker backend from {repo_or_dir} (source={source}) on device={self.device}")
            model = torch.hub.load(repo_or_dir, self.model_name, source=source)
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            self._backend_error = exc
            if self.backend_mode == 'real':
                raise
            return

        self._torch = torch
        self._model = model
        self._is_loaded = True

    @staticmethod
    def _resize_frame(frame: np.ndarray, max_dim: int | None) -> tuple[np.ndarray, float, float]:
        if max_dim is None:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        scale = min(float(max_dim) / max(h, w), 1.0)
        if scale >= 1.0:
            return frame, 1.0, 1.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, float(w) / float(new_w), float(h) / float(new_h)

    def _read_video_frames(self, stream: VideoStreamRef) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        frame_indices: list[int] = list(range(0, max(stream.num_frames, 1), self.frame_stride))
        if self.max_frames is not None:
            frame_indices = frame_indices[: int(self.max_frames)]
        frames: list[np.ndarray] = []
        model_wh: tuple[int, int] | None = None
        frame_iter = frame_indices
        if self.debug:
            frame_iter = track(frame_indices, total=len(frame_indices), description=f"[cyan]Tracking · loading frames ({stream.stream_id})")
        for frame_idx in frame_iter:
            frame = self.decoder.get_frame(stream, frame_idx)
            resized, _, _ = self._resize_frame(frame, self.resize_max_dim)
            frames.append(resized)
            model_wh = (resized.shape[1], resized.shape[0])
        if not frames:
            frame = self.decoder.get_frame(stream, 0)
            resized, _, _ = self._resize_frame(frame, self.resize_max_dim)
            frames = [resized]
            frame_indices = [0]
            model_wh = (resized.shape[1], resized.shape[0])
        video = np.stack(frames, axis=0)
        original_wh = (int(stream.width or frames[0].shape[1]), int(stream.height or frames[0].shape[0]))
        assert model_wh is not None
        return video, np.asarray(frame_indices, dtype=np.int32), original_wh, model_wh

    @staticmethod
    def _grid_seed_points(width: int, height: int, grid_size: int) -> np.ndarray:
        xs = np.linspace(0.5 * width / grid_size, width - 0.5 * width / grid_size, grid_size, dtype=np.float32)
        ys = np.linspace(0.5 * height / grid_size, height - 0.5 * height / grid_size, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)

    def _window_slices(self, num_frames: int) -> list[tuple[int, int]]:
        window_size = self.window_size
        if window_size is None:
            return [(0, num_frames)]
        window_size = int(window_size)
        if window_size <= 0 or window_size >= num_frames:
            return [(0, num_frames)]
        overlap = min(self.window_overlap, max(window_size - 1, 0))
        step = max(1, window_size - overlap)

        windows: list[tuple[int, int]] = []
        start = 0
        while start < num_frames:
            end = min(num_frames, start + window_size)
            if not windows or windows[-1] != (start, end):
                windows.append((start, end))
            if end >= num_frames:
                break
            next_start = start + step
            if next_start + window_size > num_frames:
                next_start = max(num_frames - window_size, 0)
            if next_start <= start:
                break
            start = next_start
        return windows

    def _run_real_backend_single(self, video_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._load_model()
        if not self._is_loaded or self._model is None or self._torch is None:
            raise RuntimeError('CoTracker backend is unavailable.')
        torch = self._torch
        if self.debug:
            print(f"[TRACK DEBUG] Running CoTracker on video tensor with shape={video_rgb.shape}, grid_size={self.grid_size}, device={self.device}")
        video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2)[None].float().to(self.device)
        with torch.inference_mode():
            if self.device.startswith('cuda') and self.use_autocast:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_tracks, pred_visibility = self._model(video, grid_size=self.grid_size)
            else:
                pred_tracks, pred_visibility = self._model(video, grid_size=self.grid_size)
        tracks = pred_tracks[0].detach().cpu().float().numpy()
        visibility = pred_visibility[0].detach().cpu().float().numpy()
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        return tracks, visibility

    def _run_real_backend(self, video_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        windows = self._window_slices(video_rgb.shape[0])
        if len(windows) == 1:
            tracks, visibility = self._run_real_backend_single(video_rgb)
            return tracks, visibility, {
                'windowed': False,
                'windows': [{'start': 0, 'end': int(video_rgb.shape[0]), 'track_offset': 0, 'num_tracks': int(tracks.shape[1])}],
            }

        window_iter = windows
        if self.debug:
            window_iter = track(windows, total=len(windows), description='[cyan]Tracking · CoTracker windows')

        blocks: list[dict[str, Any]] = []
        total_tracks = 0
        for start, end in window_iter:
            chunk_tracks, chunk_visibility = self._run_real_backend_single(video_rgb[start:end])
            blocks.append({
                'start': int(start),
                'end': int(end),
                'tracks': chunk_tracks,
                'visibility': chunk_visibility,
                'track_offset': int(total_tracks),
                'num_tracks': int(chunk_tracks.shape[1]),
            })
            total_tracks += int(chunk_tracks.shape[1])

        t = int(video_rgb.shape[0])
        tracks = np.zeros((t, total_tracks, 2), dtype=np.float32)
        visibility = np.zeros((t, total_tracks), dtype=np.float32)
        for block in blocks:
            start = block['start']
            end = block['end']
            offset = block['track_offset']
            num_tracks = block['num_tracks']
            tracks[start:end, offset:offset + num_tracks] = block['tracks']
            visibility[start:end, offset:offset + num_tracks] = block['visibility']

        return tracks, visibility, {
            'windowed': True,
            'windows': [
                {
                    'start': block['start'],
                    'end': block['end'],
                    'track_offset': block['track_offset'],
                    'num_tracks': block['num_tracks'],
                }
                for block in blocks
            ],
        }

    def _run_heuristic_backend(self, video_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t, h, w, _ = video_rgb.shape
        seeds = self._grid_seed_points(w, h, self.grid_size)
        n = seeds.shape[0]
        tracks = np.repeat(seeds[None, :, :], t, axis=0).astype(np.float32)
        visibility = np.ones((t, n), dtype=np.float32)
        return tracks, visibility

    def _payload_to_observations(self, stream: VideoStreamRef, payload: dict[str, Any]) -> list[TrackObservation]:
        return TrackArtifact._observations_from_payload(stream.stream_id, payload)

    def track(
        self,
        stream: VideoStreamRef,
        *,
        frame_refs: list[FrameRef] | None = None,
        seed_points: np.ndarray | None = None,
    ) -> TrackArtifact:
        """Track a sparse grid of points through the full stream.

        Notes
        -----
        ``frame_refs`` is accepted for interface compatibility but the real
        backend tracks the dense clip and later slices whichever frames are
        needed downstream.
        """
        del frame_refs, seed_points
        if self.debug:
            print(f"[TRACK DEBUG] Preparing tracking clip for {stream.stream_id} (fps={stream.fps}, frames={stream.num_frames})")
        video_rgb, frame_indices, original_wh, model_wh = self._read_video_frames(stream)
        backend_used = 'heuristic'
        tracks_xy_model: np.ndarray
        visibility: np.ndarray
        backend_meta: dict[str, Any] = {'windowed': False, 'windows': []}
        if self.backend_mode == 'heuristic':
            tracks_xy_model, visibility = self._run_heuristic_backend(video_rgb)
        else:
            try:
                tracks_xy_model, visibility, backend_meta = self._run_real_backend(video_rgb)
                backend_used = 'cotracker_windowed' if backend_meta.get('windowed') else 'cotracker'
            except Exception as exc:
                self._backend_error = exc
                if self.backend_mode == 'real':
                    raise
                tracks_xy_model, visibility = self._run_heuristic_backend(video_rgb)
                backend_used = 'heuristic'

        conf = np.asarray(visibility, dtype=np.float32)
        vis_bool = conf >= self.visibility_threshold
        scale_x = float(original_wh[0]) / float(model_wh[0]) if model_wh[0] > 0 else 1.0
        scale_y = float(original_wh[1]) / float(model_wh[1]) if model_wh[1] > 0 else 1.0
        tracks_xy = np.asarray(tracks_xy_model, dtype=np.float32).copy()
        tracks_xy[..., 0] *= scale_x
        tracks_xy[..., 1] *= scale_y
        track_ids = np.arange(tracks_xy.shape[1], dtype=np.int32)
        timestamps = frame_indices.astype(np.float32) / max(float(stream.fps), 1e-6)
        payload = {
            'frame_indices': frame_indices,
            'timestamps_sec': timestamps,
            'track_ids': track_ids,
            'points_xy': tracks_xy,
            'visibility': vis_bool,
            'confidence': conf,
            'query_frame_idx': 0,
            'original_size_wh': np.asarray(original_wh, dtype=np.int32),
            'model_size_wh': np.asarray(model_wh, dtype=np.int32),
            'backend_used': backend_used,
            'windowing': backend_meta,
        }
        notes = [
            f"Tracker backend used: {backend_used}.",
            f"Tracked {tracks_xy.shape[1]} points over {tracks_xy.shape[0]} frames.",
        ]
        if backend_meta.get('windowed'):
            notes.append(
                f"Windowed tracking used {len(backend_meta.get('windows', []))} windows; track IDs are unique per window and not continuous across the full clip."
            )
        if self._backend_error is not None and not backend_used.startswith('cotracker'):
            notes.append(f"Real backend unavailable: {type(self._backend_error).__name__}: {self._backend_error}")
        if self.debug:
            extra = ''
            if backend_meta.get('windowed'):
                extra = f", windows={len(backend_meta.get('windows', []))}, window_size={self.window_size}, window_overlap={self.window_overlap}"
            print(
                f"[TRACK DEBUG] Backend={backend_used}, track_count={tracks_xy.shape[1]}, "
                f"frame_count={tracks_xy.shape[0]}, original_wh={original_wh}, model_wh={model_wh}{extra}"
            )
        return TrackArtifact(
            run_id='dev',
            stage_name='tracking',
            save_dir=Path('.'),
            per_stream_observations={stream.stream_id: self._payload_to_observations(stream, payload)},
            seed_points_by_stream={stream.stream_id: self._grid_seed_points(model_wh[0], model_wh[1], self.grid_size).tolist()},
            merged_track_groups={},
            visibility_masks={stream.stream_id: {'visibility_threshold': self.visibility_threshold}},
            static_dynamic_hints={stream.stream_id: {'status': 'unpartitioned'}},
            track_payloads_by_stream={stream.stream_id: payload},
            notes=notes,
        )

    def track_many(
        self,
        streams: list[VideoStreamRef],
        *,
        frame_refs_by_stream: dict[str, list[FrameRef]] | None = None,
    ) -> TrackArtifact:
        per_stream_observations: dict[str, list[TrackObservation]] = {}
        seed_points_by_stream: dict[str, list[list[float]]] = {}
        visibility_masks: dict[str, Any] = {}
        static_dynamic_hints: dict[str, Any] = {}
        track_payloads_by_stream: dict[str, dict[str, Any]] = {}
        notes: list[str] = []

        for stream in streams:
            artifact = self.track(stream, frame_refs=(frame_refs_by_stream or {}).get(stream.stream_id))
            per_stream_observations.update(artifact.per_stream_observations)
            seed_points_by_stream.update(artifact.seed_points_by_stream)
            visibility_masks.update(artifact.visibility_masks)
            static_dynamic_hints.update(artifact.static_dynamic_hints)
            track_payloads_by_stream.update(artifact.track_payloads_by_stream)
            notes.extend(artifact.notes)

        if self.debug:
            print(f"[TRACK DEBUG] Finished tracking {len(streams)} stream(s).")
            for stream in streams:
                payload = track_payloads_by_stream.get(stream.stream_id)
                if payload is None:
                    print(f"[TRACK DEBUG] stream={stream.stream_id}: no payload")
                    continue

                backend_used = payload.get('backend_used', 'unknown')
                points_xy = np.asarray(payload.get('points_xy'))
                frame_indices = np.asarray(payload.get('frame_indices'))
                original_wh = tuple(np.asarray(payload.get('original_size_wh')).tolist())
                model_wh = tuple(np.asarray(payload.get('model_size_wh')).tolist())
                windowing = payload.get('windowing', {}) or {}

                track_count = int(points_xy.shape[1]) if points_xy.ndim >= 2 else 0
                frame_count = int(points_xy.shape[0]) if points_xy.ndim >= 1 else len(frame_indices)

                msg = (
                    f"[TRACK DEBUG] stream={stream.stream_id} "
                    f"backend={backend_used} "
                    f"track_count={track_count} "
                    f"frame_count={frame_count} "
                    f"original_wh={original_wh} "
                    f"model_wh={model_wh}"
                )
                if windowing.get('windowed'):
                    msg += f" windows={len(windowing.get('windows', []))}"
                print(msg)

        return TrackArtifact(
            run_id='dev',
            stage_name='tracking',
            save_dir=Path('.'),
            per_stream_observations=per_stream_observations,
            seed_points_by_stream=seed_points_by_stream,
            merged_track_groups={},
            visibility_masks=visibility_masks,
            static_dynamic_hints=static_dynamic_hints,
            track_payloads_by_stream=track_payloads_by_stream,
            notes=notes,
        )

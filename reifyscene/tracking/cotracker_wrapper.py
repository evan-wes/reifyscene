"""Wrapper module for a real or heuristic CoTracker-style point tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.progress import track

from reifyscene.common.types import FrameRef, TrackObservation, VideoStreamRef
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.tracking.base import PointTracker
from reifyscene.tracking.mask_seeds import entity_seed_queries, queries_for_window
from reifyscene.tracking.track_artifact import TrackArtifact


def _seeding_note(backend_used: str, entity_queries: np.ndarray, grid_size: int) -> str:
    """One line saying where the tracker's query points came from."""
    if backend_used == 'heuristic':
        return "Query points: none (the heuristic backend synthesizes tracks)."
    count = 0 if entity_queries is None else int(np.asarray(entity_queries).shape[0])
    if count == 0:
        return (
            f"Query points: a {grid_size}x{grid_size} uniform grid on the first frame only "
            "(no instance masks were available to seed the entities)."
        )
    return (
        f"Query points: {count} sampled on the entity masks, plus a {grid_size}x{grid_size} "
        "uniform grid per window for the background."
    )


class CoTrackerWrapper(PointTracker):
    """Foundation-model wrapper for point tracking.

    The wrapper prefers a real CoTracker backend via ``torch.hub`` when
    available. When the backend cannot be loaded and ``backend_mode='auto'``,
    it falls back to a deterministic grid tracker so downstream stages remain
    usable.

    Notes
    -----
    Windowed execution is supported to reduce peak memory usage. In windowed
    mode, overlapping windows are stitched when possible using trajectory
    agreement in the overlap region. Unmatched tracks remain separate IDs, so
    the output may still contain fragmented trajectories, but continuity is
    substantially better than naive per-window concatenation.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.decoder = VideoDecoder()
        self.backend_mode = str(self.cfg.get('backend_mode', 'auto'))
        self.device = str(self.cfg.get('device', 'cuda'))
        self.model_name = str(self.cfg.get('model_name', 'cotracker3_offline'))
        self.grid_size = int(self.cfg.get('grid_size', 12))
        # Mask seeding: points per entity scale with masked area, with a floor so a
        # small object still survives losing some, and a ceiling so one near-frame-
        # filling entity cannot crowd out the rest. A gap longer than reseed_gap means
        # the entity was unobserved and its points may not have survived it.
        self.seed_min_points = int(self.cfg.get('seed_min_points', 16))
        self.seed_per_kilopixel = float(self.cfg.get('seed_per_kilopixel', 2.0))
        self.seed_max_points = int(self.cfg.get('seed_max_points', 256))
        self.seed_reseed_gap = int(self.cfg.get('seed_reseed_gap', 4))
        self.frame_stride = max(int(self.cfg.get('frame_stride', 1)), 1)
        self.frame_spacing_sec = self.cfg.get('frame_spacing_sec')
        self.max_frames = self.cfg.get('max_frames')
        self.resize_max_dim = self.cfg.get('resize_max_dim')
        self.window_size = self.cfg.get('window_size')
        self.window_overlap = max(int(self.cfg.get('window_overlap', 0)), 0)
        self.stitch_windows = bool(self.cfg.get('stitch_windows', True))
        self.stitch_distance_px = float(self.cfg.get('stitch_distance_px', 8.0))
        self.stitch_min_overlap_frames = max(int(self.cfg.get('stitch_min_overlap_frames', 2)), 1)
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


    def _planned_frame_indices(self, stream: VideoStreamRef) -> list[int]:
        """The stream's own frame numbers, every ``frame_stride``-th -- or spaced by
        ``frame_spacing_sec`` converted with the recorded fps (refused when there is none)."""
        if self.frame_spacing_sec is not None:
            stride = stream.stride_frames(seconds=float(self.frame_spacing_sec),
                                          setting="tracking.frame_spacing_sec")
        else:
            stride = stream.stride_frames(frames=self.frame_stride, setting="tracking.frame_stride")
        frame_indices = stream.frame_numbers[::stride]
        if self.max_frames is not None:
            frame_indices = frame_indices[: int(self.max_frames)]
        if not frame_indices:
            raise ValueError(f"stream {stream.stream_id!r} has no frames to track")
        return frame_indices

    def plan_frame_refs(self, stream: VideoStreamRef) -> list[FrameRef]:
        """Return the sampled frame refs this tracker would process for a stream."""
        return [stream.frame_ref(i) for i in self._planned_frame_indices(stream)]

    def release_resources(self) -> None:
        self._model = None
        self._torch = None
        self._is_loaded = False
        self._backend_error = None

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
        frame_indices = self._planned_frame_indices(stream)
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
        video = np.stack(frames, axis=0)
        if not (stream.width and stream.height):
            # the resized frame's size is not the source's: tracks would never be scaled back
            raise ValueError(f"stream {stream.stream_id!r} records no frame size")
        original_wh = (int(stream.width), int(stream.height))
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

    def _run_real_backend_single(
        self, video_rgb: np.ndarray, queries: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track one clip. ``queries`` are ``(N, 3)`` ``(t, x, y)`` in model pixels.

        With queries the tracker is told what to follow -- the uniform grid plus points
        sampled on each entity's mask (:mod:`reifyscene.tracking.mask_seeds`). Without
        them it falls back to CoTracker's own grid, which is the behaviour when no
        instance masks are available.
        """
        self._load_model()
        if not self._is_loaded or self._model is None or self._torch is None:
            raise RuntimeError('CoTracker backend is unavailable.')
        torch = self._torch
        use_queries = queries is not None and np.asarray(queries).size > 0
        if self.debug:
            how = f"queries={np.asarray(queries).shape[0]}" if use_queries else f"grid_size={self.grid_size}"
            print(f"[TRACK DEBUG] Running CoTracker on video tensor with shape={video_rgb.shape}, {how}, device={self.device}")
        video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2)[None].float().to(self.device)
        kwargs: dict[str, Any] = {}
        if use_queries:
            kwargs['queries'] = torch.from_numpy(
                np.asarray(queries, dtype=np.float32)
            )[None].float().to(self.device)
        else:
            kwargs['grid_size'] = self.grid_size
        with torch.inference_mode():
            if self.device.startswith('cuda') and self.use_autocast:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_tracks, pred_visibility = self._model(video, **kwargs)
            else:
                pred_tracks, pred_visibility = self._model(video, **kwargs)
        tracks = pred_tracks[0].detach().cpu().float().numpy()
        visibility = pred_visibility[0].detach().cpu().float().numpy()
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        return tracks, visibility

    @staticmethod
    def _merge_track_observations(
        global_tracks: np.ndarray,
        global_conf: np.ndarray,
        track_idx: int,
        start: int,
        end: int,
        local_track: np.ndarray,
        local_conf: np.ndarray,
    ) -> None:
        """Merge one local track segment into a global track slot."""
        for rel_t, frame_t in enumerate(range(start, end)):
            conf = float(local_conf[rel_t])
            if conf <= 0.0:
                continue
            if global_conf[frame_t, track_idx] > 0.0 and np.isfinite(global_tracks[frame_t, track_idx]).all():
                prev_conf = float(global_conf[frame_t, track_idx])
                total = prev_conf + conf
                if total > 0.0:
                    global_tracks[frame_t, track_idx] = (
                        global_tracks[frame_t, track_idx] * prev_conf + local_track[rel_t] * conf
                    ) / total
                global_conf[frame_t, track_idx] = max(prev_conf, conf)
            else:
                global_tracks[frame_t, track_idx] = local_track[rel_t]
                global_conf[frame_t, track_idx] = conf

    def _stitch_window_blocks(
        self,
        blocks: list[dict[str, Any]],
        total_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Stitch overlapping window tracks into a shared global ID space.

        Tracks are matched greedily using mean trajectory distance in the
        overlapping frames. This keeps memory bounded while recovering much of
        the continuity lost by naive per-window concatenation.
        """
        if not blocks:
            return (
                np.zeros((total_frames, 0, 2), dtype=np.float32),
                np.zeros((total_frames, 0), dtype=np.float32),
                {'windowed': True, 'stitched': True, 'windows': [], 'matched_pairs': 0, 'global_track_count': 0},
            )

        first = blocks[0]
        num_tracks0 = int(first['num_tracks'])
        global_tracks = np.full((total_frames, num_tracks0, 2), np.nan, dtype=np.float32)
        global_conf = np.zeros((total_frames, num_tracks0), dtype=np.float32)
        global_tracks[first['start']:first['end'], :num_tracks0] = first['tracks']
        global_conf[first['start']:first['end'], :num_tracks0] = first['visibility']

        window_infos = [{
            'start': int(first['start']),
            'end': int(first['end']),
            'track_offset': 0,
            'num_tracks': num_tracks0,
            'matched_tracks': 0,
            'new_tracks': num_tracks0,
        }]
        matched_pairs_total = 0
        covered_until = int(first['end'])

        for block in blocks[1:]:
            start = int(block['start'])
            end = int(block['end'])
            local_tracks = np.asarray(block['tracks'], dtype=np.float32)
            local_conf = np.asarray(block['visibility'], dtype=np.float32)
            local_n = int(block['num_tracks'])

            overlap_len = max(0, min(end, covered_until) - start)
            matches: dict[int, int] = {}

            if self.stitch_windows and overlap_len > 0 and local_n > 0 and global_tracks.shape[1] > 0:
                candidate_pairs: list[tuple[float, float, int, int]] = []
                min_shared = min(self.stitch_min_overlap_frames, overlap_len)
                global_overlap_conf = global_conf[start:start + overlap_len]
                global_overlap_tracks = global_tracks[start:start + overlap_len]

                for local_idx in range(local_n):
                    local_overlap_conf = local_conf[:overlap_len, local_idx]
                    if np.count_nonzero(local_overlap_conf > 0.0) < min_shared:
                        continue
                    local_overlap_track = local_tracks[:overlap_len, local_idx]
                    for global_idx in range(global_tracks.shape[1]):
                        shared = (local_overlap_conf > 0.0) & (global_overlap_conf[:, global_idx] > 0.0)
                        shared_count = int(np.count_nonzero(shared))
                        if shared_count < min_shared:
                            continue
                        diffs = local_overlap_track[shared] - global_overlap_tracks[shared, global_idx]
                        dists = np.linalg.norm(diffs, axis=1)
                        mean_dist = float(np.mean(dists))
                        first_dist = float(dists[0])
                        if mean_dist <= self.stitch_distance_px and first_dist <= (self.stitch_distance_px * 1.5):
                            candidate_pairs.append((mean_dist, first_dist, global_idx, local_idx))

                used_global: set[int] = set()
                used_local: set[int] = set()
                for mean_dist, first_dist, global_idx, local_idx in sorted(candidate_pairs, key=lambda x: (x[0], x[1])):
                    if global_idx in used_global or local_idx in used_local:
                        continue
                    matches[local_idx] = global_idx
                    used_global.add(global_idx)
                    used_local.add(local_idx)

            matched_this_window = 0
            unmatched_local = [idx for idx in range(local_n) if idx not in matches]
            append_count = len(unmatched_local)
            track_offset = int(global_tracks.shape[1])
            if append_count > 0:
                global_tracks = np.concatenate(
                    [global_tracks, np.full((total_frames, append_count, 2), np.nan, dtype=np.float32)],
                    axis=1,
                )
                global_conf = np.concatenate(
                    [global_conf, np.zeros((total_frames, append_count), dtype=np.float32)],
                    axis=1,
                )

            for local_idx, global_idx in matches.items():
                self._merge_track_observations(global_tracks, global_conf, global_idx, start, end, local_tracks[:, local_idx], local_conf[:, local_idx])
                matched_this_window += 1

            for append_idx, local_idx in enumerate(unmatched_local):
                global_idx = track_offset + append_idx
                self._merge_track_observations(global_tracks, global_conf, global_idx, start, end, local_tracks[:, local_idx], local_conf[:, local_idx])

            matched_pairs_total += matched_this_window
            covered_until = max(covered_until, end)
            window_infos.append({
                'start': start,
                'end': end,
                'track_offset': track_offset,
                'num_tracks': local_n,
                'matched_tracks': matched_this_window,
                'new_tracks': append_count,
            })

        visibility = np.asarray(global_conf, dtype=np.float32)
        tracks = np.asarray(global_tracks, dtype=np.float32)
        tracks[~np.isfinite(tracks)] = 0.0
        return tracks, visibility, {
            'windowed': True,
            'stitched': True,
            'windows': window_infos,
            'matched_pairs': int(matched_pairs_total),
            'global_track_count': int(tracks.shape[1]),
        }

    def _run_real_backend(
        self, video_rgb: np.ndarray, entity_queries: np.ndarray | None = None,
        model_wh: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        grid_xy = None
        if entity_queries is not None and np.asarray(entity_queries).size and model_wh:
            grid_xy = self._grid_seed_points(model_wh[0], model_wh[1], self.grid_size)

        def window_queries(start: int, end: int) -> np.ndarray | None:
            if grid_xy is None:
                return None
            return queries_for_window(entity_queries, start=start, end=end, grid_xy=grid_xy)

        windows = self._window_slices(video_rgb.shape[0])
        if len(windows) == 1:
            tracks, visibility = self._run_real_backend_single(
                video_rgb, window_queries(0, int(video_rgb.shape[0]))
            )
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
            chunk_tracks, chunk_visibility = self._run_real_backend_single(
                video_rgb[start:end], window_queries(int(start), int(end))
            )
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
        if self.stitch_windows:
            tracks, visibility, meta = self._stitch_window_blocks(blocks, t)
            return tracks, visibility, meta

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
            'stitched': False,
            'windows': [
                {
                    'start': block['start'],
                    'end': block['end'],
                    'track_offset': block['track_offset'],
                    'num_tracks': block['num_tracks'],
                }
                for block in blocks
            ],
            'matched_pairs': 0,
            'global_track_count': int(total_tracks),
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

    def _entity_queries(
        self, instance_artifact, stream, frame_indices, model_wh, original_wh
    ) -> np.ndarray:
        """Query points on this stream's entities, or an empty array if there are none.

        Missing masks are not an error -- the grid alone is a valid, weaker answer, and
        is exactly what ran before masks were available here.
        """
        if instance_artifact is None:
            return np.zeros((0, 3), dtype=np.float32)
        masks_by_frame: dict[int, dict[str, Any]] = {}
        for obs in getattr(instance_artifact, 'observations', []):
            if obs.frame_ref.stream_id != stream.stream_id:
                continue
            mask = instance_artifact.mask_arrays.get(obs.instance_id)
            if mask is None:
                continue
            masks_by_frame.setdefault(int(obs.frame_ref.frame_idx), {})[str(obs.track_id)] = mask
        if not masks_by_frame:
            return np.zeros((0, 3), dtype=np.float32)
        return entity_seed_queries(
            masks_by_frame,
            frame_indices=frame_indices,
            model_wh=model_wh,
            source_wh=original_wh,
            min_points=self.seed_min_points,
            per_kilopixel=self.seed_per_kilopixel,
            max_points=self.seed_max_points,
            reseed_gap=self.seed_reseed_gap,
        )

    def track(
        self,
        stream: VideoStreamRef,
        *,
        frame_refs: list[FrameRef] | None = None,
        seed_points: np.ndarray | None = None,
        instance_artifact: Any = None,
    ) -> TrackArtifact:
        """Track points through the full stream, seeded on the entities when it can be.

        With ``instance_artifact`` the query points are the uniform grid **plus** points
        sampled inside each entity's mask, at the frame it appears and again after a gap
        (:mod:`reifyscene.tracking.mask_seeds`). Without it the grid alone is used, which
        is the previous behaviour: an entity that is small or absent at frame 0 gets
        almost no points and nothing downstream can recover them.

        Notes
        -----
        ``frame_refs`` is accepted for interface compatibility but the real
        backend tracks the dense clip and later slices whichever frames are
        needed downstream.
        """
        del seed_points
        if self.debug:
            print(f"[TRACK DEBUG] Preparing tracking clip for {stream.stream_id} (fps={stream.fps}, frames={stream.num_frames})")
        video_rgb, frame_indices, original_wh, model_wh = self._read_video_frames(stream)
        entity_queries = self._entity_queries(
            instance_artifact, stream, frame_indices, model_wh, original_wh
        )
        backend_used = 'heuristic'
        tracks_xy_model: np.ndarray
        visibility: np.ndarray
        backend_meta: dict[str, Any] = {'windowed': False, 'windows': []}
        if self.backend_mode == 'heuristic':
            tracks_xy_model, visibility = self._run_heuristic_backend(video_rgb)
        else:
            try:
                tracks_xy_model, visibility, backend_meta = self._run_real_backend(
                    video_rgb, entity_queries, model_wh
                )
                backend_used = 'cotracker_windowed' if backend_meta.get('windowed') else 'cotracker'
                backend_meta['seeding'] = 'mask+grid' if entity_queries.size else 'grid'
                backend_meta['num_entity_queries'] = int(entity_queries.shape[0])
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
        payload = {
            'frame_indices': frame_indices,
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
            # How the points were chosen decides what the tracks can answer, so the run has
            # to say it. The payload's `windowing` block does not survive serialization, and
            # a run that cannot report its own seeding is the silent degradation the ladder
            # design exists to prevent.
            _seeding_note(backend_used, entity_queries, self.grid_size),
        ]
        if backend_meta.get('windowed'):
            notes.append(
                f"Windowed tracking used {len(backend_meta.get('windows', []))} windows and produced {int(backend_meta.get('global_track_count', tracks_xy.shape[1]))} global tracks."
            )
            if backend_meta.get('stitched'):
                notes.append(
                    f"Inter-window stitching merged {int(backend_meta.get('matched_pairs', 0))} track pairs across overlaps."
                )
        if self._backend_error is not None and not backend_used.startswith('cotracker'):
            notes.append(f"Real backend unavailable: {type(self._backend_error).__name__}: {self._backend_error}")
        if self.debug:
            extra = ''
            if backend_meta.get('windowed'):
                extra = (
                    f", windows={len(backend_meta.get('windows', []))}, window_size={self.window_size}, "
                    f"window_overlap={self.window_overlap}, stitched={backend_meta.get('stitched', False)}, "
                    f"matched_pairs={backend_meta.get('matched_pairs', 0)}, global_tracks={backend_meta.get('global_track_count', tracks_xy.shape[1])}"
                )
            print(
                f"[TRACK DEBUG] Backend={backend_used}, track_count={tracks_xy.shape[1]}, "
                f"frame_count={tracks_xy.shape[0]}, original_wh={original_wh}, model_wh={model_wh}{extra}"
            )

        selected_frames = list(frame_refs) if frame_refs is not None else self.plan_frame_refs(stream)

        return TrackArtifact(
            run_id='dev',
            stage_name='tracking',
            save_dir=Path('.'),
            selected_frames=selected_frames,
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
        instance_artifact: Any = None,
    ) -> TrackArtifact:
        per_stream_observations: dict[str, list[TrackObservation]] = {}
        seed_points_by_stream: dict[str, list[list[float]]] = {}
        visibility_masks: dict[str, Any] = {}
        static_dynamic_hints: dict[str, Any] = {}
        track_payloads_by_stream: dict[str, dict[str, Any]] = {}
        notes: list[str] = []

        for stream in streams:
            artifact = self.track(
                stream,
                frame_refs=(frame_refs_by_stream or {}).get(stream.stream_id),
                instance_artifact=instance_artifact,
            )
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

        selected_frames: list[FrameRef] = []
        if frame_refs_by_stream is not None:
            for stream_id in sorted(frame_refs_by_stream):
                selected_frames.extend(frame_refs_by_stream[stream_id])
        else:
            for stream in streams:
                selected_frames.extend(self.plan_frame_refs(stream))

        selected_frames = sorted(
            selected_frames,
            key=lambda fr: (fr.stream_id, int(fr.frame_idx)),
        )

        return TrackArtifact(
            run_id='dev',
            stage_name='tracking',
            save_dir=Path('.'),
            selected_frames=selected_frames,
            per_stream_observations=per_stream_observations,
            seed_points_by_stream=seed_points_by_stream,
            merged_track_groups={},
            visibility_masks=visibility_masks,
            static_dynamic_hints=static_dynamic_hints,
            track_payloads_by_stream=track_payloads_by_stream,
            notes=notes,
        )

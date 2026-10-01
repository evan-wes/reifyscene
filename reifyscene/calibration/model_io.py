"""COLMAP text-model parsing helpers for calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState


@dataclass
class ParsedColmapModel:
    """In-memory representation of a COLMAP sparse text model."""

    intrinsics_by_camera_id: dict[int, CameraIntrinsics]
    image_records: dict[str, dict[str, Any]]
    point_records: dict[int, dict[str, Any]]
    connected_components: list[list[str]]
    reprojection_stats: dict[str, float]


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _parse_camera_model(model: str, params: list[float]) -> CameraIntrinsics:
    model = model.upper()
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy, distortion_model="simple_pinhole", distortion_params=None)
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, distortion_model="pinhole", distortion_params=None)
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        return CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy, distortion_model="simple_radial", distortion_params=np.asarray([k1], dtype=np.float32))
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        return CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy, distortion_model="radial", distortion_params=np.asarray([k1, k2], dtype=np.float32))
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, distortion_model="opencv", distortion_params=np.asarray([k1, k2, p1, p2], dtype=np.float32))
    if model == "FULL_OPENCV":
        fx, fy, cx, cy, *rest = params
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, distortion_model="full_opencv", distortion_params=np.asarray(rest, dtype=np.float32))
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, *rest = params
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, distortion_model="opencv_fisheye", distortion_params=np.asarray(rest, dtype=np.float32))
    # Fallback: preserve first-order pinhole values and stash all params as distortion.
    fx = params[0] if params else 1.0
    fy = params[1] if len(params) > 1 else fx
    cx = params[2] if len(params) > 2 else 0.0
    cy = params[3] if len(params) > 3 else 0.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, distortion_model=model.lower(), distortion_params=np.asarray(params[4:], dtype=np.float32) if len(params) > 4 else None)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def image_name_to_frame_ref(image_name: str, image_manifest: dict[str, dict[str, Any]]) -> FrameRef:
    payload = image_manifest[image_name]
    return FrameRef(
        stream_id=str(payload["stream_id"]),
        frame_idx=int(payload["frame_idx"]),
        timestamp_sec=float(payload["timestamp_sec"]),
    )


def timed_camera_from_image_record(image_name: str, record: dict[str, Any], image_manifest: dict[str, dict[str, Any]], intr: CameraIntrinsics) -> TimedCameraState:
    frame_ref = image_name_to_frame_ref(image_name, image_manifest)
    qvec = np.asarray(record["qvec"], dtype=np.float64)
    tvec = np.asarray(record["tvec"], dtype=np.float64)
    R_cw = qvec_to_rotmat(qvec)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec
    T_wc = np.linalg.inv(T_cw)
    return TimedCameraState(
        stream_id=frame_ref.stream_id,
        time_sec=float(frame_ref.timestamp_sec),
        intrinsics=intr,
        pose=CameraPose(T_world_cam=T_wc.astype(np.float32)),
    )


def parse_colmap_text_model(model_dir: str | Path) -> ParsedColmapModel:
    model_dir = Path(model_dir)
    cameras_path = model_dir / "cameras.txt"
    images_path = model_dir / "images.txt"
    points_path = model_dir / "points3D.txt"
    if not cameras_path.exists() or not images_path.exists() or not points_path.exists():
        raise FileNotFoundError(f"Expected COLMAP text model under {model_dir}, but one or more of cameras.txt, images.txt, or points3D.txt are missing.")

    intrinsics_by_camera_id: dict[int, CameraIntrinsics] = {}
    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        camera_id = int(toks[0])
        model = toks[1]
        params = [float(v) for v in toks[4:]]
        intrinsics_by_camera_id[camera_id] = _parse_camera_model(model, params)

    image_records: dict[str, dict[str, Any]] = {}
    image_point_ids: dict[str, list[int]] = {}
    image_lines = [ln.strip() for ln in images_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    idx = 0
    while idx < len(image_lines):
        header = image_lines[idx].split()
        if len(header) < 10:
            raise ValueError(f"Malformed COLMAP image header line: '{image_lines[idx]}'")
        image_id = int(header[0])
        qvec = [float(v) for v in header[1:5]]
        tvec = [float(v) for v in header[5:8]]
        camera_id = int(header[8])
        name = header[9]
        points_line = image_lines[idx + 1].split() if idx + 1 < len(image_lines) else []
        point3d_ids: list[int] = []
        for trip_idx in range(0, len(points_line), 3):
            try:
                pid = int(points_line[trip_idx + 2])
            except (IndexError, ValueError):
                continue
            if pid >= 0:
                point3d_ids.append(pid)
        image_records[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "qvec": qvec,
            "tvec": tvec,
            "num_observed_points": len(point3d_ids),
        }
        image_point_ids[name] = point3d_ids
        idx += 2

    point_records: dict[int, dict[str, Any]] = {}
    for line in points_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        point_id = int(toks[0])
        xyz = [float(v) for v in toks[1:4]]
        rgb = [int(v) for v in toks[4:7]]
        error = float(toks[7])
        track = toks[8:]
        image_ids: list[int] = []
        for j in range(0, len(track), 2):
            try:
                image_ids.append(int(track[j]))
            except (IndexError, ValueError):
                continue
        point_records[point_id] = {"xyz": xyz, "rgb": rgb, "error": error, "image_ids": image_ids}

    id_to_name = {rec["image_id"]: name for name, rec in image_records.items()}
    uf = _UnionFind(list(image_records.keys()))
    for point in point_records.values():
        names = [id_to_name[image_id] for image_id in point.get("image_ids", []) if image_id in id_to_name]
        if len(names) < 2:
            continue
        anchor = names[0]
        for other in names[1:]:
            uf.union(anchor, other)

    comps: dict[str, list[str]] = {}
    for name in image_records:
        comps.setdefault(uf.find(name), []).append(name)
    connected_components = [sorted(vals) for vals in sorted(comps.values(), key=lambda v: (-len(v), v[0]))]

    errors = np.asarray([float(v["error"]) for v in point_records.values()], dtype=np.float32)
    if errors.size:
        reprojection_stats = {
            "num_points3d": int(errors.size),
            "registered_images": int(len(image_records)),
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "max": float(errors.max()),
            "num_connected_components": int(len(connected_components)),
            "largest_component_size": int(max((len(c) for c in connected_components), default=0)),
        }
    else:
        reprojection_stats = {
            "num_points3d": 0,
            "registered_images": int(len(image_records)),
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "num_connected_components": int(len(connected_components)),
            "largest_component_size": int(max((len(c) for c in connected_components), default=0)),
        }

    return ParsedColmapModel(
        intrinsics_by_camera_id=intrinsics_by_camera_id,
        image_records=image_records,
        point_records=point_records,
        connected_components=connected_components,
        reprojection_stats=reprojection_stats,
    )


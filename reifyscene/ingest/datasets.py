"""The on-disk layout of the datasets the pipeline reads -- the only place it is written down.

- :class:`DyCheckSequence`: a sequence in the DyCheck / Shape-of-Motion format (DyCheck
  iPhone and NVIDIA Dynamic Scenes share it): ``camera/<name>.json``,
  ``rgb|depth/<k>x/<name>``, ``covisible/<k>x/<split>/``, ``splits/<split>.json``,
  ``extra.json`` (its frame rate) and the ``flow3d_preprocessed/`` products Shape-of-Motion
  derives (COLMAP, aligned depth, Track-Anything masks, 2D tracks). Frames are named
  ``<camera>_<time:05d>``.
- :class:`DavisSequence`: a DAVIS 2017 sequence: ``JPEGImages/<res>/<seq>/<stem>.jpg`` and its
  instance annotation ``Annotations/<res>/<seq>/<stem>.png``.

Anything that reads one of these datasets goes through these classes; nothing else formats
their paths or names.
"""

from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np


class DatasetLayoutError(ValueError):
    """The folder is not laid out as the dataset says."""


class DyCheckSequence:
    """One DyCheck-format sequence directory."""

    NAME_PAD = 5

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def of_frames_dir(cls, frames_dir: str | Path) -> tuple[DyCheckSequence, int] | None:
        """``(sequence, factor)`` when ``frames_dir`` is a sequence's ``rgb/<k>x`` folder (the
        sequence has ``camera/`` and ``extra.json``), else None."""
        path = Path(frames_dir)
        m = re.match(r"^(\d+)x$", path.name)
        root = path.parent.parent
        if m and path.parent.name == "rgb" and (root / "camera").is_dir() \
                and (root / "extra.json").is_file():
            return cls(root), int(m.group(1))
        return None

    # -- names ---------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.root.name

    def frame_name(self, camera: Any, time: int) -> str:
        """``<camera>_<time:05d>``."""
        return f"{camera}_{int(time):0{self.NAME_PAD}d}"

    @staticmethod
    def camera_of(frame_name: str) -> str:
        return str(frame_name).split("_", 1)[0]

    # -- records ---------------------------------------------------------------------------------

    @cached_property
    def extra(self) -> dict[str, Any]:
        path = self.root / "extra.json"
        if not path.is_file():
            raise DatasetLayoutError(f"{self.root}: no extra.json")
        return json.loads(path.read_text())

    @property
    def fps(self) -> float | None:
        """The frame rate the dataset records (``extra.json``), or None."""
        fps = self.extra.get("fps")
        return float(fps) if fps else None

    def split_frames(self, split: str) -> list[tuple[str, int]]:
        """``(camera, time)`` pairs of a split (``splits/<split>.json``)."""
        meta = json.loads((self.root / "splits" / f"{split}.json").read_text())
        return list(zip([str(c) for c in meta["camera_ids"]],
                        [int(t) for t in meta["time_ids"]], strict=True))

    # -- paths -----------------------------------------------------------------------------------

    def camera_path(self, frame_name: str) -> Path:
        return self.root / "camera" / f"{frame_name}.json"

    def rgb_dir(self, factor: int) -> Path:
        return self.root / "rgb" / f"{int(factor)}x"

    def rgb_path(self, frame_name: str, factor: int) -> Path:
        return self.rgb_dir(factor) / f"{frame_name}.png"

    def depth_path(self, frame_name: str, factor: int) -> Path:
        return self.root / "depth" / f"{int(factor)}x" / f"{frame_name}.npy"

    def covisible_path(self, frame_name: str, factor: int, split: str) -> Path:
        return self.root / "covisible" / f"{int(factor)}x" / split / f"{frame_name}.png"

    @property
    def preprocessed(self) -> Path:
        """Shape-of-Motion's derived products for this sequence."""
        return self.root / "flow3d_preprocessed"

    def fg_mask_path(self, frame_name: str, factor: int) -> Path:
        return self.preprocessed / "track_anything" / f"{int(factor)}x" / f"{frame_name}.png"

    @property
    def colmap_sparse_dir(self) -> Path:
        return self.preprocessed / "colmap" / "sparse"

    def aligned_depth_path(self, frame_name: str, factor: int, depth_type: str) -> Path:
        return (self.preprocessed / f"aligned_{depth_type}" / f"{int(factor)}x"
                / f"{frame_name}.npy")

    def tracks_dir(self, factor: int) -> Path:
        return self.preprocessed / "2d_tracks" / f"{int(factor)}x"

    def frames(self, factor: int) -> list[str]:
        """Frame names that have both an ``rgb/<factor>x`` image and a camera json."""
        rgb, cams = self.rgb_dir(factor), self.root / "camera"
        if not rgb.is_dir() or not cams.is_dir():
            raise DatasetLayoutError(f"{self.root}: missing rgb/{int(factor)}x/ or camera/")
        names = sorted({p.stem for p in rgb.glob("*.png")} & {p.stem for p in cams.glob("*.json")})
        if not names:
            raise DatasetLayoutError(
                f"{self.root}: no frames with both an rgb/{int(factor)}x png and a camera json")
        return names

    # -- readers ---------------------------------------------------------------------------------

    @staticmethod
    def _distortion(cam: dict[str, Any]) -> np.ndarray | None:
        radial = list(cam.get("radial_distortion") or [])
        tangential = list(cam.get("tangential_distortion") or [])
        k1, k2, k3 = (radial + [0.0, 0.0, 0.0])[:3]
        p1, p2 = (tangential + [0.0, 0.0])[:2]
        params = np.asarray([k1, k2, p1, p2, k3], dtype=float)
        return None if np.allclose(params, 0.0) else params

    def load_camera(self, frame_name: str, factor: int):
        """``(intrinsics, T_world_cam 4x4, (width, height))`` at the ``<factor>x`` resolution."""
        from reifyscene.common.types import CameraIntrinsics

        cam = json.loads(self.camera_path(frame_name).read_text())
        full_w, full_h = (int(round(float(v))) for v in cam["image_size"])
        focal = float(cam["focal_length"])
        aspect = float(cam.get("pixel_aspect_ratio", 1.0))
        cx, cy = (float(v) for v in cam["principal_point"])
        intrinsics = CameraIntrinsics(
            fx=focal / factor, fy=(focal * aspect) / factor, cx=cx / factor, cy=cy / factor,
            distortion_model="brown", distortion_params=self._distortion(cam),
        )
        rot = np.asarray(cam["orientation"], dtype=np.float64).reshape(3, 3)
        pos = np.asarray(cam["position"], dtype=np.float64).reshape(3)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rot
        w2c[:3, 3] = -rot @ pos
        return intrinsics, np.linalg.inv(w2c).astype(np.float32), (full_w // factor,
                                                                  full_h // factor)

    def load_depth(self, frame_name: str, factor: int) -> np.ndarray:
        """Metric depth (HxW float32); a trailing channel is squeezed."""
        return np.squeeze(np.asarray(np.load(self.depth_path(frame_name, factor)), np.float32))

    def load_rgb(self, frame_name: str, factor: int) -> np.ndarray:
        from PIL import Image

        with Image.open(self.rgb_path(frame_name, factor)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def load_fg_mask(self, frame_name: str, factor: int) -> np.ndarray:
        """Track-Anything foreground mask (HxW float in [0, 1]; 1 = foreground)."""
        from PIL import Image

        with Image.open(self.fg_mask_path(frame_name, factor)) as img:
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def load_covisible(self, frame_name: str, factor: int, split: str) -> np.ndarray:
        """Co-visibility mask (HxW bool, True = covisible) of a held-out frame."""
        from PIL import Image

        with Image.open(self.covisible_path(frame_name, factor, split)) as img:
            return np.asarray(img.convert("L")) > 0


class DavisSequence:
    """One DAVIS 2017 sequence: its frames and per-frame instance annotations."""

    def __init__(self, root: str | Path, resolution: str, name: str) -> None:
        self.root, self.resolution, self.name = Path(root), str(resolution), str(name)

    @classmethod
    def of_frames_dir(cls, frames_dir: str | Path) -> DavisSequence | None:
        """The sequence a ``JPEGImages/<res>/<seq>`` folder is, or None if it is not one."""
        path = Path(frames_dir)
        if path.parent.parent.name != "JPEGImages":
            return None
        seq = cls(path.parent.parent.parent, path.parent.name, path.name)
        return seq if seq.annotation_dir.is_dir() else None

    @classmethod
    def all(cls, root: str | Path, resolution: str = "480p") -> list[DavisSequence]:
        frames = Path(root) / "JPEGImages" / resolution
        return [cls(root, resolution, p.name) for p in sorted(frames.iterdir()) if p.is_dir()]

    @property
    def frames_dir(self) -> Path:
        return self.root / "JPEGImages" / self.resolution / self.name

    @property
    def annotation_dir(self) -> Path:
        return self.root / "Annotations" / self.resolution / self.name

    def frame_files(self) -> list[Path]:
        return sorted(self.frames_dir.glob("*.jpg"))

    def annotation_for(self, frame_file: str | Path) -> np.ndarray | None:
        """The instance-id annotation of a frame (found by the frame file's stem), or None
        when DAVIS annotates no such frame."""
        from PIL import Image

        path = self.annotation_dir / f"{Path(frame_file).stem}.png"
        if not path.is_file():
            return None
        with Image.open(path) as img:
            return np.asarray(img)

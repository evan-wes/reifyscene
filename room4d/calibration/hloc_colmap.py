"""hloc + COLMAP / pycolmap orchestration for calibration mode."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from room4d.common.paths import ensure_dir


class HlocColmapSfMRunner:
    """Run hloc matching and COLMAP/pycolmap sparse reconstruction."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def _image_options(self) -> dict[str, Any]:
        image_options: dict[str, Any] = {}
        if self.cfg.get("camera_model"):
            image_options["camera_model"] = str(self.cfg["camera_model"])
        if self.cfg.get("camera_params"):
            # Known-intrinsics protocol (gt_scoring D4): hloc applies
            # image_options GLOBALLY — one camera for the entire import —
            # so this only expresses rigs whose cameras share intrinsics.
            # Per-camera injection needs per-folder imports or COLMAP DB
            # writes (post-spike work, see the calibration design doc).
            image_options["camera_params"] = str(self.cfg["camera_params"])
        return image_options

    def _camera_mode(self, pycolmap) -> Any:
        mode_name = str(self.cfg.get("camera_mode", "PER_FOLDER")).upper()
        return getattr(pycolmap.CameraMode, mode_name, pycolmap.CameraMode.AUTO)

    def run(self, image_dir: str | Path, output_dir: str | Path) -> Path:
        image_dir = Path(image_dir)
        output_dir = ensure_dir(output_dir)
        mock_model = self.cfg.get("mock_sparse_model_dir")
        if mock_model:
            mock_model = Path(mock_model)
            sparse_dir = ensure_dir(output_dir / "sparse")
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                shutil.copy2(mock_model / name, sparse_dir / name)
            return sparse_dir

        try:
            import pycolmap
            from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction
        except Exception as exc:  # pragma: no cover - exercised in real environments
            raise RuntimeError(
                "Calibration mode requires the optional 'hloc' and 'pycolmap' packages. "
                "Install Hierarchical-Localization and pycolmap before running calibration."
            ) from exc

        hloc_dir = ensure_dir(output_dir / "hloc")
        sparse_dir = ensure_dir(output_dir / "sparse")
        retrieval_name = str(self.cfg.get("retrieval_conf", "netvlad"))
        feature_name = str(self.cfg.get("feature_conf", "superpoint_aachen"))
        matcher_name = str(self.cfg.get("matcher_conf", "superglue"))
        top_k = int(self.cfg.get("retrieval_top_k", 20))
        retrieval_conf = extract_features.confs[retrieval_name]
        feature_conf = extract_features.confs[feature_name]
        matcher_conf = match_features.confs[matcher_name]
        sfm_pairs = hloc_dir / "pairs-sfm.txt"
        retrieval_path = extract_features.main(retrieval_conf, image_dir, hloc_dir)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=top_k)
        features = extract_features.main(feature_conf, image_dir, hloc_dir)
        matches = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], hloc_dir)

        image_options = self._image_options()
        mapper_options = dict(self.cfg.get("mapper_options", {}))
        if self.cfg.get("fix_intrinsics", False):
            mapper_options.setdefault("ba_refine_focal_length", False)
            mapper_options.setdefault("ba_refine_extra_params", False)
        reconstruction.main(
            sfm_dir=sparse_dir,
            image_dir=image_dir,
            pairs=sfm_pairs,
            features=features,
            matches=matches,
            camera_mode=self._camera_mode(pycolmap),
            image_options=image_options or None,
            mapper_options=mapper_options or None,
            verbose=bool(self.cfg.get("verbose", False)),
        )
        return sparse_dir

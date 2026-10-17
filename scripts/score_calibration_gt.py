#!/usr/bin/env python3
"""Score hloc+COLMAP calibration against dataset ground-truth extrinsics.

M0-minimal of the calibration arc (design doc D2–D12): collect dataset
frames with their GT poses, run the existing static SfM, and report
pairwise-relative pose errors (primary) plus guarded Sim(3) APE
(secondary). Runs in the ``reifyscene`` env (numpy + calibration module — this
script is NOT stdlib-only, unlike run_reference_benchmark.py).

Examples:
  # NVIDIA static rig: 4 cameras x 5 timesteps, free intrinsics
  python scripts/score_calibration_gt.py --dataset nvidia \
    --data-dir nvidia_dynamic/Balloon1 --run-dir balloon1-calib

  # Known-intrinsics protocol (shared-intrinsics datasets only)
  python scripts/score_calibration_gt.py --dataset nvidia --fix-intrinsics \
    --data-dir nvidia_dynamic/Balloon1 --run-dir balloon1-calib-fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from reifyscene.common import data_dir
from reifyscene.calibration.gt_scoring import (
    DATASETS,
    DEFAULT_NUM_FRAMES,
    DEFAULT_TIMESTEPS,
    GTCamera,
    GTSchemaError,
    SCORE_SCHEMA,
    collect_frames,
    enumerate_frames,
    load_gt_cameras,
    load_manifest,
    score_against_gt,
    select_frames,
    shared_intrinsics,
)
from reifyscene.calibration.hloc_colmap import HlocColmapSfMRunner
from reifyscene.calibration.model_io import parse_colmap_text_model


def ensure_text_model(sparse_dir: Path, out_dir: Path) -> Path:
    """COLMAP text model from whatever the SfM run produced.

    hloc/pycolmap may write a binary model; parse_colmap_text_model needs
    text. Searches sparse_dir (and models/*) for either format and converts
    binary via pycolmap when required.
    """
    candidates = [sparse_dir, *sorted(sparse_dir.glob("models/*"))]
    for cand in candidates:
        if (cand / "images.txt").is_file():
            return cand
    for cand in candidates:
        if (cand / "images.bin").is_file():
            import pycolmap

            out_dir.mkdir(parents=True, exist_ok=True)
            pycolmap.Reconstruction(str(cand)).write_text(str(out_dir))
            return out_dir
    sys.exit(
        f"no COLMAP model (images.txt/images.bin) found under {sparse_dir} — "
        "the SfM run produced nothing parseable"
    )


def build_sfm_cfg(fix_intrinsics: bool, cameras: list[GTCamera]) -> dict:
    """SfM runner config for the requested intrinsics protocol.

    Known-intrinsics mode injects the shared GT camera as a SIMPLE_PINHOLE
    ``"focal,cx,cy"`` params string with refinement disabled (D4, scoped to
    shared-intrinsics rigs per D12). Pure function so tests can pin the
    exact payload — a silently swapped cx/cy would otherwise only surface
    on a real GPU run.
    """
    if not fix_intrinsics:
        # Free mode still pins camera_mode explicitly: the runner's default
        # (PER_FOLDER) on a flat image dir means one shared camera anyway —
        # making it explicit + recorded keeps the protocol honest. Per-image
        # /per-physical-camera free modes are post-spike work.
        return {"camera_mode": "SINGLE"}
    rep = shared_intrinsics(cameras)
    if rep is None:
        raise GTSchemaError(
            "--fix-intrinsics needs shared intrinsics across collected "
            "cameras (hloc applies image_options globally; per-camera "
            "injection is post-spike work — see the design doc)"
        )
    return {
        "camera_model": "SIMPLE_PINHOLE",
        "camera_params": f"{rep.focal},{rep.principal_point[0]},{rep.principal_point[1]}",
        "fix_intrinsics": True,
        "camera_mode": "SINGLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", required=True,
                        help="Processed sequence, by name in the data directory's datasets/")
    parser.add_argument("--run-dir", required=True,
                        help="Output (images/, work/, score json), by name in runs/")
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"nvidia: timesteps per camera (default {DEFAULT_TIMESTEPS})")
    parser.add_argument("--num-frames", type=int, default=None,
                        help=f"iphone: frames along the trajectory (default {DEFAULT_NUM_FRAMES})")
    parser.add_argument(
        "--fix-intrinsics", action="store_true",
        help="Known-intrinsics protocol: inject shared GT intrinsics into "
        "COLMAP and disable refinement (aborts if intrinsics differ per camera)",
    )
    parser.add_argument("--output",
                        help="Score JSON, by name in runs/ "
                             "(default: <run-dir>/calibration_gt_score.json)")
    # test hook: routes the SfM runner through its existing mock path so the
    # CLI is exercisable without hloc/GPU (mirrors CalibrationRunner tests)
    parser.add_argument("--mock-sparse-model-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    seq_dir = data_dir.path(data_dir.DATASETS, args.data_dir)
    run_dir = data_dir.path(data_dir.RUNS, args.run_dir)

    # The selection knobs are dataset-specific; a silently ignored flag
    # would misdescribe the run. Validate bounds while we're at it.
    for name, value, applies_to in (
        ("--timesteps", args.timesteps, "nvidia"),
        ("--num-frames", args.num_frames, "iphone"),
    ):
        if value is None:
            continue
        if value < 1:
            parser.error(f"{name} must be >= 1")
        if args.dataset != applies_to:
            print(f"[warn] {name} only applies to --dataset {applies_to}; ignored", flush=True)
    timesteps = args.timesteps if args.timesteps is not None else DEFAULT_TIMESTEPS
    num_frames = args.num_frames if args.num_frames is not None else DEFAULT_NUM_FRAMES

    try:
        frames = enumerate_frames(args.dataset, seq_dir)
        selected = select_frames(
            args.dataset, frames, timesteps=timesteps, num_frames=num_frames
        )
        cameras = load_gt_cameras(args.dataset, seq_dir, selected)
        image_dir = run_dir / "images"
        manifest_path = collect_frames(cameras, image_dir)
    except GTSchemaError as exc:
        sys.exit(f"GT collection failed: {exc}")

    try:
        sfm_cfg = build_sfm_cfg(args.fix_intrinsics, cameras)
    except GTSchemaError as exc:
        sys.exit(str(exc))
    if args.mock_sparse_model_dir:
        sfm_cfg["mock_sparse_model_dir"] = args.mock_sparse_model_dir

    print(
        f"[collect] {len(cameras)} frames from {args.dataset}:{seq_dir.name} "
        f"-> {image_dir}",
        flush=True,
    )
    sparse_dir = HlocColmapSfMRunner(sfm_cfg).run(image_dir, run_dir / "work")
    text_dir = ensure_text_model(Path(sparse_dir), run_dir / "colmap_text")
    model = parse_colmap_text_model(text_dir)

    # The artifact must be able to PROVE what was measured: record the
    # recovered intrinsics and, in fixed-gt mode, verify they match the
    # injected values (a silent injection failure is otherwise
    # indistinguishable from a run where the protocol held).
    recovered = {
        str(cid): {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
        for cid, intr in model.intrinsics_by_camera_id.items()
    }
    intrinsics_match = None
    if args.fix_intrinsics:
        rep = shared_intrinsics(cameras)
        intrinsics_match = all(
            abs(v["fx"] - rep.focal) < 1e-3
            and abs(v["cx"] - rep.principal_point[0]) < 1e-3
            and abs(v["cy"] - rep.principal_point[1]) < 1e-3
            for v in recovered.values()
        )
        if not intrinsics_match:
            print(
                "[warn] fixed-gt run but RECOVERED intrinsics differ from the "
                f"injected GT values: {recovered} — the injection did not "
                "hold; treat these numbers as free-mode.",
                flush=True,
            )

    result = score_against_gt(model, load_manifest(manifest_path))
    result = {
        "schema": SCORE_SCHEMA,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "run_dir": args.run_dir,
        "frames": [c.source_frame for c in cameras],
        "intrinsics_mode": "fixed-gt" if args.fix_intrinsics else "free",
        "intrinsics_match_injected": intrinsics_match,
        "camera_mode": sfm_cfg.get("camera_mode", "PER_FOLDER"),
        "recovered_intrinsics": recovered,
        "colmap_model_dir": data_dir.record_name(text_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result,
    }
    out_path = (data_dir.path(data_dir.RUNS, args.output) if args.output
                else run_dir / "calibration_gt_score.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nCalibration GT score written to: {out_path}\n")

    reg = result["registration"]
    print(
        f"  cameras registered   {reg['cameras_registered']}/{reg['cameras_total']}"
        f"   images {reg['images_registered']}/{reg['images_total']}"
    )
    frag = result["fragmentation"]
    print(f"  components           {frag['num_components']} {frag['component_sizes']}")
    if not result["alignable"]:
        sys.exit(f"  NOT ALIGNABLE: {result['error']}")
    print(
        f"  scored component     {result['scored_component_size']} images "
        f"across {len(result['per_camera_images'])} camera(s), "
        f"coverage {result['scored_coverage']:.2f}"
    )
    pw = result["metrics"]["pairwise"]
    if pw["rot_deg"]:
        print(f"  pairwise rot (deg)   mean {pw['rot_deg']['mean']:.4f}  median {pw['rot_deg']['median']:.4f}  max {pw['rot_deg']['max']:.4f}")
    else:
        print("  pairwise rot (deg)   no non-degenerate pairs (see rot_deg_degenerate_pairs)")
    if pw["trans_dir_deg"]:
        print(f"  pairwise dir (deg)   mean {pw['trans_dir_deg']['mean']:.4f}  median {pw['trans_dir_deg']['median']:.4f}  max {pw['trans_dir_deg']['max']:.4f}")
    if pw["baseline_ratio_err"]:
        print(f"  baseline ratio err   mean {pw['baseline_ratio_err']['mean']:.4f}  max {pw['baseline_ratio_err']['max']:.4f}")
    ape = result["metrics"]["sim3_ape"]
    if "skipped" in ape:
        print(f"  sim3 APE             skipped: {ape['skipped']}")
    else:
        # only 'planar' can flag here: a colinear rig already took the
        # skipped branch above
        flags = " [planar]" if ape["conditioning"].get("planar") else ""
        print(f"  sim3 rot (deg)       mean {ape['rot_deg']['mean']:.4f}  max {ape['rot_deg']['max']:.4f}{flags}")
        print(f"  sim3 trans (GT u.)   mean {ape['trans']['mean']:.4f}  max {ape['trans']['max']:.4f}  scale {ape['scale']:.4f}")


if __name__ == "__main__":
    main()

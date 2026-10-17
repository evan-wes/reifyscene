"""Generate reifyscene's *native* 2D point tracks (SoM ``2d_tracks`` format).

Runs reifyscene's own tapnet (bootstapir) tracker over a DyCheck-format sequence's
train split and writes all-pairs ``<name_i>_<name_j>.npy`` tracks — the exact
layout ``reifyscene.lifting.native_track_lift.lift_tracks_3d`` reads and that ships
borrowed as ``flow3d_preprocessed/2d_tracks/``. This is the last borrowed input
in the native motion-init path; point the lift at this output to drop it.

Runs in the plain ``reifyscene`` env (no SoM reference env): the model is the
vendored ``reifyscene.tracking.tapnet`` TAPIR with the checkpoint fetched by
``scripts/fetch_tapir_checkpoint.sh``.

Example (reifyscene env):
    python scripts/compute_tracks_native.py \
        --dataset-root dycheck_iphone/paper-windmill \
        --out native_tracks/paper-windmill

Then lift/train from it:
    python scripts/run_render_proof.py dynamic --native --own-tracker \
        --own-tracks-dir native_tracks/paper-windmill \
        --dataset-root dycheck_iphone/paper-windmill

Names are inside the data directory ($REIFYSCENE_DATA_DIR): the sequence in datasets/, the
tracks in reference/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reifyscene.common import data_dir
from reifyscene.lifting.native_track_lift import generate_native_tracks


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reifyscene's native 2D tracks (SoM 2d_tracks format)")
    ap.add_argument("--dataset-root", required=True, help="the sequence's name in datasets/")
    ap.add_argument("--out", required=True,
                    help="output root's name in reference/; tracks land in <out>/<factor>x/")
    ap.add_argument("--model-type", default="bootstapir", choices=["bootstapir", "tapir"])
    ap.add_argument("--grid-size", type=int, default=4, help="query-grid stride in full-res px (SoM used 4)")
    ap.add_argument("--camera-type", default="refined", choices=["refined", "original"])
    ap.add_argument("--checkpoint-path", default=None,
                    help="the .pt's name in checkpoints/ (else tapnet/<model's file>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tracker_cfg = {
        "model_type": args.model_type,
        "grid_size": args.grid_size,
        "device": args.device,
    }
    if args.checkpoint_path:
        tracker_cfg["checkpoint_path"] = args.checkpoint_path

    factor_dir = generate_native_tracks(
        data_dir.path(data_dir.DATASETS, args.dataset_root),
        data_dir.path(data_dir.REFERENCE, args.out), factor=1,
        camera_type=args.camera_type, tracker_cfg=tracker_cfg,
    )
    n_files = sum(1 for _ in Path(factor_dir).glob("*.npy"))
    print(f"native tracks -> {data_dir.record_name(factor_dir)}  ({n_files} files)")


if __name__ == "__main__":
    main()

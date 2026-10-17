#!/usr/bin/env bash
set -euo pipefail

# Validate a manually-downloaded reference dataset sequence for the reifyscene
# comparison tool. We deliberately do NOT auto-download: the Shape of Motion
# datasets live in Google Drive folders that are simplest to grab in a browser.
# This script just records WHERE to get the data and verifies that an extracted
# sequence directory has the layout Shape of Motion's dataloaders expect, then
# prints the exact --data.data-dir to pass to run_training.py.
#
# Dataset sources (from third_party/shape-of-motion/README.md):
#   - Processed iPhone / DyCheck sequences (paper-windmill, apple, spin, ...):
#       https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe
#   - Processed NVIDIA Dynamic Scenes + custom:
#       https://drive.google.com/drive/folders/1xzn-Mu_jyr-JTsrERRU-Mh2hQ-NWdfv8
#
# Usage:
#   scripts/verify_reference_data.sh --dataset iphone  --seq-dir /path/to/paper-windmill
#   scripts/verify_reference_data.sh --dataset nvidia  --seq-dir /path/to/<scene>

DATASET=""
SEQ_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --seq-dir) SEQ_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/verify_reference_data.sh --dataset {iphone|nvidia} --seq-dir DIR

Validates that an already-downloaded/extracted reference dataset sequence has
the layout Shape of Motion expects, and prints the --data.data-dir to use.
Does NOT download anything (grab the Google Drive folders in a browser; links
are documented at the top of this script and in the Shape of Motion README).
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DATASET}" || -z "${SEQ_DIR}" ]]; then
  echo "Both --dataset and --seq-dir are required. See --help." >&2
  exit 2
fi

if [[ ! -d "${SEQ_DIR}" ]]; then
  echo "Sequence directory does not exist: ${SEQ_DIR}" >&2
  exit 1
fi

# Required entries per dataset. These lists MUST stay aligned with the
# *_REQUIRED_DIRS / *_REQUIRED_FILES lists in run_reference_benchmark.py
# (tests/test_reference_benchmark.py checks both validators against the same
# fixture). Verified against the pinned SoM commit:
#   - iphone: the dataloader opens dataset.json, extra.json,
#     splits/{train,val}.json unconditionally, and evaluate_iphone.py needs
#     colmap/scale.npy, covisible/ and keypoint/. metadata.json ships with
#     the data but SoM never reads it — not required.
#   - nvidia: the dataloader hardcodes factor=2, so training opens the 2x
#     variants (plus rgb/1x for the intrinsics rescale); depth and 2d_tracks
#     feed the train split, covisible/2x/val the val split's masked metrics.
#     dataset.json/scene.json/colmap/ ship with the data but the nvidia
#     loader never reads them — not required.
required_dirs=()
required_files=()
case "${DATASET}" in
  iphone)
    required_dirs=(rgb depth camera splits flow3d_preprocessed
                   flow3d_preprocessed/2d_tracks flow3d_preprocessed/covisible
                   keypoint)
    required_files=(dataset.json extra.json splits/train.json splits/val.json
                    flow3d_preprocessed/colmap/scale.npy)
    ;;
  nvidia)
    required_dirs=(rgb/1x rgb/2x camera splits depth/2x
                   flow3d_preprocessed/2d_tracks/2x
                   flow3d_preprocessed/track_anything/2x
                   covisible/2x/val)
    required_files=(splits/train.json splits/val.json)
    ;;
  *)
    echo "Unknown --dataset '${DATASET}' (expected: iphone | nvidia)." >&2
    exit 2 ;;
esac

missing=0
echo "Validating ${DATASET} sequence: ${SEQ_DIR}"
for d in "${required_dirs[@]}"; do
  if [[ -d "${SEQ_DIR}/${d}" ]]; then
    echo "  [ok]   dir  ${d}/"
  else
    echo "  [MISS] dir  ${d}/"; missing=1
  fi
done
for f in "${required_files[@]}"; do
  if [[ -e "${SEQ_DIR}/${f}" ]]; then
    echo "  [ok]   file ${f}"
  else
    echo "  [MISS] file ${f}"; missing=1
  fi
done

# (scene_depth.npy is intentionally NOT checked — it only exists for
# depth_type=monst3r; shipped sequences use aligned_depth_anything_colmap/.)

# nvidia content checks (mirroring validate_sequence in
# run_reference_benchmark.py): an empty val split trains for hours then
# yields no metrics, and the dataloader loads RANDOM pairwise track files
# per batch, so a partial download passes layout checks yet crashes
# mid-training. Both are detectable up front.
if [[ "${DATASET}" == "nvidia" && ${missing} -eq 0 ]]; then
  python3 - "${SEQ_DIR}" <<'PY' || missing=1
import json, os, sys

seq = sys.argv[1]

def frame_names(split_file):
    try:
        with open(os.path.join(seq, "splits", split_file)) as f:
            split = json.load(f)
    except (OSError, ValueError):
        return None
    return split.get("frame_names", []) if isinstance(split, dict) else None

ok = True
val, train = frame_names("val.json"), frame_names("train.json")
if val is None or train is None:
    print("  [MISS] splits/{val,train}.json unreadable or not a JSON object")
    ok = False
elif not val:
    print("  [MISS] splits/val.json frame_names is empty (no val split -> no metrics)")
    ok = False
if train:
    tracks = os.path.join(seq, "flow3d_preprocessed", "2d_tracks", "2x")
    have = len(os.listdir(tracks))
    want = len(train) ** 2
    if have < want:
        print(f"  [MISS] 2d_tracks/2x holds {have} files, expected {want}"
              " (len(train)^2 pairwise tracks) - partial download")
        ok = False
sys.exit(0 if ok else 1)
PY
fi

echo
if [[ ${missing} -ne 0 ]]; then
  echo "RESULT: INCOMPLETE — see [MISS] above. Re-download the sequence folder." >&2
  exit 1
fi

echo "RESULT: OK — sequence looks complete."
echo "Use with Shape of Motion:"
echo "  cd third_party/shape-of-motion"
echo "  python run_training.py --work-dir <OUT> data:${DATASET} --data.data-dir ${SEQ_DIR}"

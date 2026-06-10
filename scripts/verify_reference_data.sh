#!/usr/bin/env bash
set -euo pipefail

# Validate a manually-downloaded reference dataset sequence for the room4d
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

# Required entries per dataset. The iphone lists MUST stay aligned with
# IPHONE_REQUIRED_DIRS / IPHONE_REQUIRED_FILES in run_reference_benchmark.py
# (tests/test_reference_benchmark.py checks both validators against the same
# fixture). Verified against the pinned SoM commit: the dataloader opens
# dataset.json, extra.json, splits/{train,val}.json unconditionally, and
# evaluate_iphone.py needs colmap/scale.npy, covisible/ and keypoint/.
# metadata.json ships with the data but SoM never reads it — not required.
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
    required_dirs=(rgb)
    required_files=()
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

echo
if [[ ${missing} -ne 0 ]]; then
  echo "RESULT: INCOMPLETE — see [MISS] above. Re-download the sequence folder." >&2
  exit 1
fi

if [[ "${DATASET}" == "nvidia" ]]; then
  echo "RESULT: OK — minimal nvidia layout present (only rgb/ is checked;"
  echo "full nvidia support in the benchmark wrapper is tracked in TODOS.md)."
else
  echo "RESULT: OK — sequence looks complete."
fi
echo "Use with Shape of Motion:"
echo "  cd third_party/shape-of-motion"
echo "  python run_training.py --work-dir <OUT> data:${DATASET} --data.data-dir ${SEQ_DIR}"

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

# Required entries per dataset, matching flow3d/data/*_dataset.py expectations.
required_dirs=()
required_files=()
case "${DATASET}" in
  iphone)
    required_dirs=(rgb depth camera splits flow3d_preprocessed)
    required_files=(dataset.json metadata.json splits/train.json)
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

# iPhone: the preprocessed depth/covisible used for training/eval must exist.
if [[ "${DATASET}" == "iphone" ]]; then
  if [[ -e "${SEQ_DIR}/flow3d_preprocessed/scene_depth.npy" ]]; then
    echo "  [ok]   file flow3d_preprocessed/scene_depth.npy"
  else
    echo "  [warn] flow3d_preprocessed/scene_depth.npy not found"
    echo "         (training expects SoM's preprocessed depth; re-check the download)"
    missing=1
  fi
fi

echo
if [[ ${missing} -ne 0 ]]; then
  echo "RESULT: INCOMPLETE — see [MISS]/[warn] above. Re-download the sequence folder." >&2
  exit 1
fi

echo "RESULT: OK — sequence looks complete."
echo "Use with Shape of Motion:"
echo "  cd third_party/shape-of-motion"
echo "  python run_training.py --work-dir <OUT> data:${DATASET} --data.data-dir ${SEQ_DIR}"

#!/usr/bin/env bash
set -euo pipefail

# Fetch the tapnet point-tracker checkpoint used by reifyscene's native bootstapir
# tracker backend (reifyscene/tracking/tapir_wrapper.py -> reifyscene/tracking/tapnet/).
#
# These are the same DeepMind torch checkpoints Shape of Motion's preproc uses
# (third_party/shape-of-motion/preproc/setup_dependencies.sh +
# compute_tracks_torch.py). Unlike the SoM datasets (Google Drive, grabbed in a
# browser — see scripts/verify_reference_data.sh), the checkpoints live at
# direct https URLs, so this fetch is fully scripted and idempotent.
#
# The checkpoint goes where the tracker looks for it: checkpoints/tapnet/ in the data
# directory ($REIFYSCENE_DATA_DIR).
#
# Usage:
#   scripts/fetch_tapir_checkpoint.sh                 # bootstapir (default)
#   scripts/fetch_tapir_checkpoint.sh --model tapir   # tapir panning variant
#   scripts/fetch_tapir_checkpoint.sh --force         # re-download even if present

MODEL="bootstapir"
FORCE=0

# Minimum plausible checkpoint size (MB). Both checkpoints are ~120 MB; a
# truncated/HTML-error download is far smaller, so this guards against a
# half-written file masquerading as complete (same idea as the VGGT-Long
# weights size guard in setup_reifyscene.sh).
MIN_MB="${TAPNET_CKPT_MIN_MB:-80}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/fetch_tapir_checkpoint.sh [--model {bootstapir|tapir}] [--force]

Downloads a tapnet torch checkpoint for reifyscene's native point tracker into
checkpoints/tapnet/ in the data directory ($REIFYSCENE_DATA_DIR). Idempotent: skips
when a plausibly-sized file already exists (use --force to re-download).
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${MODEL}" in
  bootstapir)
    URL="https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
    FILENAME="bootstapir_checkpoint_v2.pt" ;;
  tapir)
    URL="https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt"
    FILENAME="tapir_checkpoint_panning.pt" ;;
  *)
    echo "Unknown --model '${MODEL}' (expected bootstapir or tapir)." >&2
    exit 2 ;;
esac

: "${REIFYSCENE_DATA_DIR:?set REIFYSCENE_DATA_DIR to your data directory}"
DEST_DIR="${REIFYSCENE_DATA_DIR}/checkpoints/tapnet"
DEST="${DEST_DIR}/${FILENAME}"

file_big_enough() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local mb
  mb="$(du -m "${path}" 2>/dev/null | awk '{print $1}')"
  [[ -n "${mb}" && "${mb}" -ge "${MIN_MB}" ]]
}

if [[ ${FORCE} -eq 0 ]] && file_big_enough "${DEST}"; then
  echo "Checkpoint already present (>= ${MIN_MB} MB): ${DEST}"
  echo "Pass --force to re-download."
  exit 0
fi

mkdir -p "${DEST_DIR}"
echo "Downloading ${MODEL} checkpoint:"
echo "  from: ${URL}"
echo "  to:   ${DEST}"

# Download to a temp file and mv into place so an interrupted transfer never
# leaves a partial file that the size guard might later accept.
TMP="$(mktemp "${DEST}.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "${TMP}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TMP}" "${URL}"
else
  echo "Neither curl nor wget is available to download the checkpoint." >&2
  exit 1
fi

if ! file_big_enough "${TMP}"; then
  echo "Downloaded file is smaller than ${MIN_MB} MB — treating as a failed/truncated download." >&2
  exit 1
fi

mv "${TMP}" "${DEST}"
trap - EXIT
echo "Done: ${DEST} ($(du -m "${DEST}" | awk '{print $1}') MB)"

#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the current room4d prototype.
#
# What this installs today:
# - the local room4d package in editable mode
# - Python dependencies declared in pyproject.toml
# - optional detectron2 from official tag v0.6
# - optional VGGT from the official repo at a pinned commit, cloned into
#   third_party/vggt and installed into the active environment
#
# What this intentionally does NOT auto-install:
# - RealSense SDK. Use scripts/setup_realsense_sdk.sh if you want the optional
#   camera SDK on Ubuntu.

# ------------------------------------------------------------------------------
# Pinned external dependency refs
# ------------------------------------------------------------------------------

DETECTRON2_REPO="https://github.com/facebookresearch/detectron2.git"
DETECTRON2_TAG="v0.6"

VGGT_REPO="https://github.com/facebookresearch/vggt.git"
VGGT_COMMIT="44b3afbd1869d8bde4894dd8ea1e293112dd5eba"

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DETECTRON2=0
INSTALL_VGGT=0
INSTALL_DEV=0
INSTALL_VISION=1
INSTALL_GEOMETRY=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --venv)
      VENV_DIR="$2"; shift 2 ;;
    --with-detectron2)
      INSTALL_DETECTRON2=1; shift ;;
    --with-vggt)
      INSTALL_VGGT=1; shift ;;
    --with-dev)
      INSTALL_DEV=1; shift ;;
    --no-vision)
      INSTALL_VISION=0; shift ;;
    --no-geometry)
      INSTALL_GEOMETRY=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_room4d.sh [options]

Options:
  --python PATH           Python executable to use (default: python3)
  --venv PATH             Virtualenv path (default: .venv)
  --with-detectron2       Install detectron2 from official tag v0.6
  --with-vggt             Clone/install VGGT into third_party/vggt at pinned commit
  --with-dev              Install dev tools (pytest, ruff, mypy)
  --no-vision             Skip vision extras
  --no-geometry           Skip geometry extras
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

EXTRAS=()
[[ $INSTALL_VISION -eq 1 ]] && EXTRAS+=(vision)
[[ $INSTALL_GEOMETRY -eq 1 ]] && EXTRAS+=(geometry)
[[ $INSTALL_DEV -eq 1 ]] && EXTRAS+=(dev)

# Install room4d and all Python dependencies from pyproject.toml only.
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
  EXTRA_CSV=$(IFS=,; echo "${EXTRAS[*]}")
  python -m pip install -e "${ROOT_DIR}[${EXTRA_CSV}]"
else
  python -m pip install -e "${ROOT_DIR}"
fi

if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
  TMP_DIR=$(mktemp -d)
  echo "Installing detectron2 from official tag ${DETECTRON2_TAG}..."
  git clone --depth 1 --branch "${DETECTRON2_TAG}" --single-branch "${DETECTRON2_REPO}" "$TMP_DIR/detectron2"
  python -m pip install -e "$TMP_DIR/detectron2"
fi

if [[ $INSTALL_VGGT -eq 1 ]]; then
  mkdir -p "${THIRD_PARTY_DIR}"

  if [[ ! -d "${VGGT_DIR}/.git" ]]; then
    echo "Cloning VGGT into ${VGGT_DIR}..."
    git clone "${VGGT_REPO}" "${VGGT_DIR}"
  fi

  echo "Checking out VGGT commit ${VGGT_COMMIT}..."
  git -C "${VGGT_DIR}" fetch origin
  git -C "${VGGT_DIR}" checkout "${VGGT_COMMIT}"

  echo "Installing VGGT from ${VGGT_DIR}..."
  python -m pip install -e "${VGGT_DIR}"
fi

cat <<EOF

Setup complete.

Installed:
- room4d in editable mode from:
    ${ROOT_DIR}
- Python dependencies from pyproject.toml
$( [[ $INSTALL_DETECTRON2 -eq 1 ]] && echo "- detectron2 from ${DETECTRON2_TAG}" )
$( [[ $INSTALL_VGGT -eq 1 ]] && echo "- VGGT from commit ${VGGT_COMMIT} at ${VGGT_DIR}" )

Notes:
- third_party/ should be gitignored so external source checkouts are never tracked.
- For Intel RealSense SDK / pyrealsense2 on Ubuntu, see:
    scripts/setup_realsense_sdk.sh
- For a first RGB-only camera test without RealSense SDK, you can use:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
EOF

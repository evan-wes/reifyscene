#!/usr/bin/env bash
set -euo pipefail

# Optional installer for Intel RealSense SDK on Ubuntu.
# This script is intentionally separate from the core package install because
# you do not need RealSense SDK for RGB-only testing when the camera is exposed
# as a standard UVC device.
#
# Pinned to official librealsense release tag v2.53.1.
#
# For Ubuntu 18/20 with kernel 5, Intel's release notes call out frame-drop
# issues and recommend the RSUSB backend as a mitigation for applicable cases.
# This script enables FORCE_RSUSB_BACKEND by default.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREFIX_DIR="${PREFIX_DIR:-$HOME/.local}"
WORK_DIR="${WORK_DIR:-$HOME/src}"
FORCE_RSUSB=1
BUILD_PYTHON=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --prefix)
      PREFIX_DIR="$2"; shift 2 ;;
    --workdir)
      WORK_DIR="$2"; shift 2 ;;
    --no-rsusb)
      FORCE_RSUSB=0; shift ;;
    --no-python)
      BUILD_PYTHON=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_realsense_sdk.sh [options]

Options:
  --python PATH      Python executable for pyrealsense2 build (default: python3)
  --prefix PATH      Install prefix (default: $HOME/.local)
  --workdir PATH     Clone/build workspace (default: $HOME/src)
  --no-rsusb         Build without FORCE_RSUSB_BACKEND
  --no-python        Skip Python bindings build
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

sudo apt-get update
sudo apt-get install -y   git build-essential cmake pkg-config   libssl-dev libusb-1.0-0-dev libgtk-3-dev   libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev   python3-dev python3-pip python3-venv

mkdir -p "$WORK_DIR"
if [[ ! -d "$WORK_DIR/librealsense/.git" ]]; then
  git clone --depth 1 --branch v2.53.1 --single-branch https://github.com/IntelRealSense/librealsense.git "$WORK_DIR/librealsense"
else
  echo "Existing librealsense clone detected at $WORK_DIR/librealsense"
fi

mkdir -p "$WORK_DIR/librealsense/build"
cd "$WORK_DIR/librealsense/build"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$PREFIX_DIR"
  -DBUILD_EXAMPLES=false
  -DBUILD_GRAPHICAL_EXAMPLES=false
)

if [[ $FORCE_RSUSB -eq 1 ]]; then
  CMAKE_ARGS+=( -DFORCE_RSUSB_BACKEND=ON )
fi

if [[ $BUILD_PYTHON -eq 1 ]]; then
  CMAKE_ARGS+=(
    -DBUILD_PYTHON_BINDINGS=ON
    -DPYTHON_EXECUTABLE="$(command -v "$PYTHON_BIN")"
  )
fi

cmake .. "${CMAKE_ARGS[@]}"
cmake --build . --config Release -j"$(nproc)"
cmake --install .

cat <<EOF

librealsense v2.53.1 build complete.

Next steps:
- If you only want RGB capture, try the simpler OpenCV/UVC route first:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
- If pyrealsense2 built successfully and is on your PYTHONPATH, you can also use:
    python scripts/capture_realsense_rgb.py --output out.mp4
EOF

#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the current room4d prototype.
#
# What this installs today:
# - the local room4d package in editable mode
# - Python dependencies declared in pyproject.toml
# - optional detectron2 from its official v0.6 release tag
# - optional VGGT from the official repo at a pinned commit, cloned into
#   third_party/vggt and installed into the active environment
# - optional CoTracker from the official repo at a pinned commit, cloned into
#   third_party/co-tracker and installed into the active environment
# - optional VGGT-Long from the official repo at a pinned commit, cloned into
#   third_party/VGGT-Long and installed into the active environment
#
# Environment behavior:
# - By default, this script expects to use a conda environment.
# - You may provide --conda-env NAME to require / activate a specific env.
# - If no conda env is active and none can be activated, the script exits.
# - There is NO silent fallback to venv.
# - To explicitly use a repo-local venv instead, pass --use-venv.

# ------------------------------------------------------------------------------
# Pinned external dependency refs
# ------------------------------------------------------------------------------

DETECTRON2_REPO="https://github.com/facebookresearch/detectron2.git"
DETECTRON2_TAG="v0.6"

VGGT_REPO="https://github.com/facebookresearch/vggt.git"
VGGT_COMMIT="44b3afbd1869d8bde4894dd8ea1e293112dd5eba"

COTRACKER_REPO="https://github.com/facebookresearch/co-tracker.git"
COTRACKER_COMMIT="82e02e8029753ad4ef13cf06be7f4fc5facdda4d"

VGGT_LONG_REPO="https://github.com/DengKaiCQ/VGGT-Long.git"
VGGT_LONG_COMMIT="c160869d1d99c96bb227f414afb3bc68c29c9a76"

REQUIRED_GXX_MAJOR=9
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-11}"
# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
USE_VENV=0

INSTALL_DETECTRON2=0
INSTALL_VGGT=0
INSTALL_COTRACKER=0
INSTALL_VGGT_LONG=0
INSTALL_DEV=0
INSTALL_VISION=1
INSTALL_GEOMETRY=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"
COTRACKER_DIR="${THIRD_PARTY_DIR}/co-tracker"
VGGT_LONG_DIR="${THIRD_PARTY_DIR}/VGGT-Long"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --venv)
      VENV_DIR="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV_NAME="$2"; shift 2 ;;
    --use-venv)
      USE_VENV=1; shift ;;
    --with-detectron2)
      INSTALL_DETECTRON2=1; shift ;;
    --with-vggt)
      INSTALL_VGGT=1; shift ;;
    --with-cotracker)
      INSTALL_COTRACKER=1; shift ;;
    --with-vggt-long)
      INSTALL_VGGT_LONG=1; shift ;;
    --with-dev)
      INSTALL_DEV=1; shift ;;
    --no-vision)
      INSTALL_VISION=0; shift ;;
    --no-geometry)
      INSTALL_GEOMETRY=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_room4d.sh [options]

Environment options:
  --conda-env NAME        Require / activate this conda environment
  --use-venv              Explicitly use a repo-local venv instead of conda
  --python PATH           Python executable to use (default: python3)
  --venv PATH             Virtualenv path (default: .venv)

Install options:
  --with-detectron2       Install detectron2 from official tag v0.6
  --with-vggt             Clone/install VGGT into third_party/vggt at pinned commit
  --with-cotracker        Clone/install CoTracker into third_party/co-tracker at pinned commit
  --with-vggt-long        Clone/install VGGT-Long into third_party/VGGT-Long at pinned commit
  --with-dev              Install dev tools (pytest, ruff, mypy)
  --no-vision             Skip vision extras
  --no-geometry           Skip geometry extras

Behavior:
  - Default mode is conda-only.
  - If no conda environment is active and none can be activated, the script exits.
  - There is no silent fallback to venv.
  - Use --use-venv if you explicitly want a repo-local virtualenv.
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if [[ $USE_VENV -eq 1 && -n "${CONDA_ENV_NAME}" ]]; then
  echo "Cannot use --use-venv and --conda-env together." >&2
  exit 2
fi

clone_or_update_checkout() {
  local repo_url="$1"
  local dest_dir="$2"
  local ref="$3"

  mkdir -p "$(dirname "$dest_dir")"

  if [[ ! -d "${dest_dir}/.git" ]]; then
    echo "Cloning ${repo_url} into ${dest_dir}..."
    git clone "${repo_url}" "${dest_dir}"
  fi

  echo "Fetching updates in ${dest_dir}..."
  git -C "${dest_dir}" fetch origin

  echo "Checking out ${ref} in ${dest_dir}..."
  git -C "${dest_dir}" checkout "${ref}"
}

compiler_major_version() {
  local compiler="$1"
  if ! command -v "${compiler}" >/dev/null 2>&1; then
    echo ""
    return
  fi
  "${compiler}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1 | cut -d. -f1
}

run_conda_install_relaxed() {
  # Conda activation/deactivation hooks are not always nounset-safe.
  # This script uses `set -u`, so temporarily relax that around conda install.
  local had_nounset=0
  case "$-" in
    *u*) had_nounset=1 ;;
  esac

  set +u
  conda install "$@"
  local rc=$?

  if [[ ${had_nounset} -eq 1 ]]; then
    set -u
  fi

  return ${rc}
}

ensure_modern_conda_compilers() {
  if [[ $USE_VENV -eq 1 ]]; then
    local current_compiler="${CXX:-c++}"
    local current_major
    current_major="$(compiler_major_version "${current_compiler}")"

    if [[ -z "${current_major}" || "${current_major}" -lt "${REQUIRED_GXX_MAJOR}" ]]; then
      echo "VGGT-Long requires GCC/G++ ${REQUIRED_GXX_MAJOR}+." >&2
      echo "Current compiler: ${current_compiler}" >&2
      echo "Detected version: $("${current_compiler}" --version 2>/dev/null | head -n 1 || echo unknown)" >&2
      echo "When using --use-venv, please provide a newer system compiler manually." >&2
      exit 1
    fi

    echo "Using existing compiler for venv mode: ${current_compiler}"
    return
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring modern compilers." >&2
    exit 1
  fi

  local conda_cxx="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
  local conda_cc="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"

  local current_compiler="${CXX:-c++}"
  local current_major
  current_major="$(compiler_major_version "${current_compiler}")"

  # If conda compilers already exist, prefer them for reproducibility.
  if [[ -x "${conda_cxx}" && -x "${conda_cc}" ]]; then
    export CC="${conda_cc}"
    export CXX="${conda_cxx}"
    echo "Using conda compilers:"
    echo "  CC=${CC}"
    echo "  CXX=${CXX}"
    echo "  CXX version: $("${CXX}" --version | head -n 1)"
    return
  fi

  if [[ -n "${current_major}" && "${current_major}" -ge "${REQUIRED_GXX_MAJOR}" ]]; then
    echo "Existing compiler is already new enough:"
    echo "  ${current_compiler} ($( "${current_compiler}" --version | head -n 1 ))"
    return
  fi

  echo "Current compiler is too old for VGGT-Long/PyTorch extension builds."
  echo "Installing newer conda compilers into env '${CONDA_DEFAULT_ENV:-unknown}'..."
  run_conda_install_relaxed -y -c conda-forge \
    "gcc_linux-64=${CONDA_GCC_VERSION}" \
    "gxx_linux-64=${CONDA_GCC_VERSION}"

  if [[ ! -x "${conda_cxx}" || ! -x "${conda_cc}" ]]; then
    echo "Expected conda compilers were not found after installation." >&2
    echo "Expected:"
    echo "  ${conda_cc}"
    echo "  ${conda_cxx}"
    exit 1
  fi

  export CC="${conda_cc}"
  export CXX="${conda_cxx}"

  echo "Installed and selected conda compilers:"
  echo "  CC=${CC}"
  echo "  CXX=${CXX}"
  echo "  CXX version: $("${CXX}" --version | head -n 1)"
}

activate_conda_env() {
  local target_env="$1"

  if ! command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_EXE:-}" ]]; then
    echo "Conda is required unless --use-venv is provided, but 'conda' was not found." >&2
    exit 1
  fi

  local conda_base=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
  else
    conda_base="$(conda info --base)"
  fi

  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"

  if [[ -n "${target_env}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${target_env}" ]]; then
      echo "Using already-active conda environment: ${target_env}"
      return
    fi

    echo "Activating conda environment: ${target_env}"
    if ! conda activate "${target_env}"; then
      echo "Failed to activate conda environment '${target_env}'." >&2
      exit 1
    fi
    return
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using already-active conda environment: ${CONDA_DEFAULT_ENV}"
    return
  fi

  echo "No active conda environment detected. Provide --conda-env NAME or use --use-venv." >&2
  exit 1
}

setup_python_env() {
  if [[ $USE_VENV -eq 1 ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "Python executable not found: $PYTHON_BIN" >&2
      exit 1
    fi

    echo "Creating / using repo-local venv: ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  else
    activate_conda_env "${CONDA_ENV_NAME}"
  fi

  echo "Python executable: $(command -v python)"
  python -m pip install --upgrade pip setuptools wheel
}

install_vggt_long_requirements() {
  local req_file="${VGGT_LONG_DIR}/requirements.txt"

  if [[ ! -f "${req_file}" ]]; then
    return
  fi

  echo "Installing VGGT-Long requirements from ${req_file}..."

  local tmp_req
  tmp_req="$(mktemp)"

  # Remove packages we intentionally do not install automatically here:
  # - xformers: requires building against a local CUDA toolkit matching torch
  # - faiss-gpu: should be installed via conda if needed
  grep -Eiv '^[[:space:]]*(xformers|faiss-gpu)([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}" > "${tmp_req}" || true

  if [[ -s "${tmp_req}" ]]; then
    python -m pip install -r "${tmp_req}"
  fi
  rm -f "${tmp_req}"

  if grep -Eqi '^[[:space:]]*xformers([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}"; then
    echo "Skipping xformers automatic install."
    echo "Reason: local CUDA toolkit must match the CUDA version used to compile torch."
    echo "Current torch is cu121, but your local build toolchain is detecting CUDA 11.2."
    echo "Install xformers manually later only if VGGT-Long actually needs it."
  fi

  if grep -Eqi '^[[:space:]]*faiss-gpu([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}"; then
    echo "Skipping faiss-gpu from pip requirements."
    echo "If you need it later, install it separately via conda."
  fi
}

setup_python_env

EXTRAS=()
[[ $INSTALL_VISION -eq 1 ]] && EXTRAS+=(vision)
[[ $INSTALL_GEOMETRY -eq 1 ]] && EXTRAS+=(geometry)
[[ $INSTALL_VGGT_LONG -eq 1 ]] && EXTRAS+=(vggt_long)
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
  python -m pip install --no-build-isolation -e "$TMP_DIR/detectron2"
fi

if [[ $INSTALL_VGGT -eq 1 ]]; then
  clone_or_update_checkout "${VGGT_REPO}" "${VGGT_DIR}" "${VGGT_COMMIT}"
  echo "Installing VGGT from ${VGGT_DIR}..."
  python -m pip install -e "${VGGT_DIR}"
fi

if [[ $INSTALL_COTRACKER -eq 1 ]]; then
  clone_or_update_checkout "${COTRACKER_REPO}" "${COTRACKER_DIR}" "${COTRACKER_COMMIT}"
  echo "Installing CoTracker from ${COTRACKER_DIR}..."
  python -m pip install -e "${COTRACKER_DIR}"
fi

if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
  ensure_modern_conda_compilers
  clone_or_update_checkout "${VGGT_LONG_REPO}" "${VGGT_LONG_DIR}" "${VGGT_LONG_COMMIT}"
  install_vggt_long_requirements

  if [[ -f "${VGGT_LONG_DIR}/scripts/download_weights.sh" ]]; then
    echo "Patching VGGT-Long download_weights.sh for current Hugging Face CLI..."
    sed -i 's/\bhuggingface-cli\b/hf/g' "${VGGT_LONG_DIR}/scripts/download_weights.sh"
    sed -i 's/[[:space:]]*--local-dir-use-symlinks[[:space:]]\+False//g' "${VGGT_LONG_DIR}/scripts/download_weights.sh"
  fi

  # Provide a compatibility shim if the environment has `hf` but not `huggingface-cli`.
  if ! command -v huggingface-cli >/dev/null 2>&1 && command -v hf >/dev/null 2>&1; then
    mkdir -p "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin"
    cat > "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin/huggingface-cli" <<'EOF'
#!/usr/bin/env bash
exec hf "$@"
EOF
    chmod +x "${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin/huggingface-cli"
    export PATH="${CONDA_PREFIX:-${ROOT_DIR}/.room4d_cache}/bin:${PATH}"
  fi

  echo "Downloading VGGT-Long weights into ${VGGT_LONG_DIR}/weights ..."
  (
    cd "${VGGT_LONG_DIR}"
    bash ./scripts/download_weights.sh
  )

  echo "Installing VGGT-Long from ${VGGT_LONG_DIR}..."
  python -m pip install --no-build-isolation -e "${VGGT_LONG_DIR}"
fi

cat <<EOF

Setup complete.

Environment:
$( [[ $USE_VENV -eq 1 ]] && echo "- repo-local venv: ${VENV_DIR}" || echo "- conda env: ${CONDA_DEFAULT_ENV:-unknown}" )

Installed:
- room4d in editable mode from:
    ${ROOT_DIR}
- Python dependencies from pyproject.toml
$( [[ $INSTALL_DETECTRON2 -eq 1 ]] && echo "- detectron2 from ${DETECTRON2_TAG}" )
$( [[ $INSTALL_VGGT -eq 1 ]] && echo "- VGGT from commit ${VGGT_COMMIT} at ${VGGT_DIR}" )
$( [[ $INSTALL_COTRACKER -eq 1 ]] && echo "- CoTracker from commit ${COTRACKER_COMMIT} at ${COTRACKER_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long from commit ${VGGT_LONG_COMMIT} at ${VGGT_LONG_DIR}" )

Notes:
- third_party/ should be gitignored so external source checkouts are never tracked.
- .room4d_cache/ should be gitignored too.
- For Intel RealSense SDK / pyrealsense2 on Ubuntu, see:
    scripts/setup_realsense_sdk.sh
- For a first RGB-only camera test without RealSense SDK, you can use:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
EOF

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
# - optional calibration dependencies: pycolmap pinned to a COLMAP release,
#   plus hloc cloned into third_party/Hierarchical-Localization at a pinned tag
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

HLOC_REPO="https://github.com/cvg/Hierarchical-Localization.git"
HLOC_TAG="v1.4"
COLMAP_UPSTREAM_TAG="3.13.0"
PYCOLMAP_VERSION="3.13.0"

# gsplat is built from source for sm_120 (no prebuilt wheel exists for this
# torch/CUDA line). Pinned to the same Blackwell-proven commit as
# setup_reference_tool.sh; the reference env built it against CUDA 12.8 —
# this env builds the same ref against CUDA 13.0.
GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_REF="4e52698e45eaaed929ed3a5065e96a688d085df6"
GSPLAT_ARCH_LIST="12.0"   # TORCH_CUDA_ARCH_LIST for the 5090 (sm_120)

REQUIRED_GXX_MAJOR=9
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-11}"
# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
USE_VENV=0
UNLOCKED=0

# Full pip-layer lockfile (see scripts/gen_env_locks.sh). Installed before the
# editable room4d install so the transitive dependency tree replays the
# known-good snapshot instead of re-resolving. torch +cu130 wheels only exist
# on the pytorch index, so the lock install needs it on a fresh env.
ROOM4D_LOCK_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/locks/room4d.lock"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

INSTALL_DETECTRON2=0
INSTALL_VGGT=0
INSTALL_COTRACKER=0
INSTALL_VGGT_LONG=0
INSTALL_GSPLAT=0
INSTALL_DEV=0
INSTALL_VISION=1
INSTALL_GEOMETRY=1
INSTALL_CALIBRATION=1
FORCE_VGGT_LONG_WEIGHTS=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"
COTRACKER_DIR="${THIRD_PARTY_DIR}/co-tracker"
VGGT_LONG_DIR="${THIRD_PARTY_DIR}/VGGT-Long"
HLOC_DIR="${THIRD_PARTY_DIR}/Hierarchical-Localization"
VGGT_LONG_WEIGHTS_DIR="${VGGT_LONG_DIR}/weights"
VGGT_LONG_WEIGHTS_STAMP="${VGGT_LONG_WEIGHTS_DIR}/.room4d_download_complete"
VGGT_LONG_WEIGHTS_MIN_MB="${VGGT_LONG_WEIGHTS_MIN_MB:-8000}"

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
    --with-gsplat)
      INSTALL_GSPLAT=1; shift ;;
    --force-vggt-long-weights)
      FORCE_VGGT_LONG_WEIGHTS=1; shift ;;
    --with-dev)
      INSTALL_DEV=1; shift ;;
    --no-vision)
      INSTALL_VISION=0; shift ;;
    --no-geometry)
      INSTALL_GEOMETRY=0; shift ;;
    --no-calibration)
      INSTALL_CALIBRATION=0; shift ;;
    --unlocked)
      UNLOCKED=1; shift ;;
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
  --with-gsplat           Build gsplat from source for sm_120 at pinned commit
  --force-vggt-long-weights
                          Re-download VGGT-Long weights even if already present
  --with-dev              Install dev tools (pytest, ruff, mypy)
  --no-vision             Skip vision extras
  --no-geometry           Skip geometry extras
  --no-calibration        Skip calibration extras and hloc install
  --unlocked              Resolve dependencies fresh instead of installing the
                          pinned tree from scripts/locks/room4d.lock
                          (regenerate locks with scripts/gen_env_locks.sh)

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

python_one_liner() {
  local snippet="$1"
  python - <<PY
${snippet}
PY
}

detect_torch_cuda_version() {
  python_one_liner "import torch; print(torch.version.cuda or '')" 2>/dev/null || true
}

detect_torch_cuda_home() {
  python_one_liner "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')" 2>/dev/null || true
}

nvcc_cuda_version() {
  if ! command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1
}

ensure_cuda_home_exports() {
  local cuda_home="$1"
  if [[ -z "${cuda_home}" ]]; then
    return 1
  fi

  export CUDA_HOME="${cuda_home}"

  if [[ -d "${CUDA_HOME}/bin" ]]; then
    case ":${PATH}:" in
      *":${CUDA_HOME}/bin:"*) ;;
      *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
  fi

  local lib_dir
  for lib_dir in \
    "${CUDA_HOME}/lib64" \
    "${CUDA_HOME}/lib" \
    "${CUDA_HOME}/targets/x86_64-linux/lib"
  do
    if [[ -d "${lib_dir}" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${lib_dir}:"*) ;;
        *) export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done

  return 0
}

ensure_cuda_toolkit_for_torch_extensions() {
  local existing_cuda_home=""
  existing_cuda_home="$(detect_torch_cuda_home | tr -d '[:space:]')"

  if [[ -n "${existing_cuda_home}" && -x "${existing_cuda_home}/bin/nvcc" ]]; then
    ensure_cuda_home_exports "${existing_cuda_home}"
    echo "Using existing CUDA toolkit detected by torch: ${CUDA_HOME}"
    echo "  nvcc: $(command -v nvcc)"
    echo "  nvcc version: $(nvcc --version | tail -n 1)"
    return
  fi

  if [[ $USE_VENV -eq 1 ]]; then
    echo "This install step needs a CUDA toolkit with nvcc, but none was detected." >&2
    echo "In --use-venv mode the setup script does not install CUDA automatically." >&2
    echo "Either install a local CUDA toolkit manually and set CUDA_HOME, or use a conda env." >&2
    exit 1
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected while ensuring CUDA toolkit." >&2
    exit 1
  fi

  local torch_cuda=""
  torch_cuda="$(detect_torch_cuda_version | tr -d '[:space:]')"
  if [[ -z "${torch_cuda}" ]]; then
    echo "Could not determine torch.version.cuda from the active environment." >&2
    echo "Make sure torch is installed before requesting CUDA-extension builds." >&2
    exit 1
  fi

  local -a channels=()
  local patch
  for patch in 2 1 0; do
    channels+=("nvidia/label/cuda-${torch_cuda}.${patch}")
  done
  channels+=("nvidia")

  local installed=0
  local channel
  for channel in "${channels[@]}"; do
    echo "Attempting to install CUDA toolkit from conda channel: ${channel}"
    if run_conda_install_relaxed -y -c "${channel}" cuda-toolkit; then
      installed=1
      break
    fi
    echo "Conda install attempt from ${channel} did not succeed; trying next option..."
  done

  if [[ ${installed} -ne 1 ]]; then
    echo "Failed to install a CUDA toolkit into the active conda environment." >&2
    echo "Tried torch CUDA line ${torch_cuda} and a generic NVIDIA channel fallback." >&2
    exit 1
  fi

  ensure_cuda_home_exports "${CONDA_PREFIX}"

  local refreshed_cuda_home=""
  refreshed_cuda_home="$(detect_torch_cuda_home | tr -d '[:space:]')"
  if [[ -n "${refreshed_cuda_home}" ]]; then
    ensure_cuda_home_exports "${refreshed_cuda_home}"
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA toolkit install completed, but nvcc is still not on PATH." >&2
    echo "CUDA_HOME=${CUDA_HOME:-unset}" >&2
    exit 1
  fi

  echo "Installed CUDA toolkit for extension builds:"
  echo "  torch.version.cuda=${torch_cuda}"
  echo "  CUDA_HOME=${CUDA_HOME}"
  echo "  nvcc=$(command -v nvcc)"
  echo "  nvcc version: $(nvcc --version | tail -n 1)"
}

vggt_long_weights_present() {
  if [[ ! -d "${VGGT_LONG_WEIGHTS_DIR}" ]]; then
    return 1
  fi

  if [[ -f "${VGGT_LONG_WEIGHTS_STAMP}" ]]; then
    return 0
  fi

  local total_mb=""
  total_mb="$(du -sm "${VGGT_LONG_WEIGHTS_DIR}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${total_mb}" || "${total_mb}" -lt "${VGGT_LONG_WEIGHTS_MIN_MB}" ]]; then
    return 1
  fi

  [[ -f "${VGGT_LONG_WEIGHTS_DIR}/dinov2_vitb14_pretrain.pth" ]] || return 1
  [[ -f "${VGGT_LONG_WEIGHTS_DIR}/ORBvoc.txt" ]] || return 1

  if [[ -d "${VGGT_LONG_WEIGHTS_DIR}/Pi3" || -d "${VGGT_LONG_WEIGHTS_DIR}/pi3" || -d "${VGGT_LONG_WEIGHTS_DIR}/map_anything" ]]; then
    return 0
  fi

  return 1
}

download_vggt_long_weights_if_needed() {
  mkdir -p "${VGGT_LONG_WEIGHTS_DIR}"

  if [[ ${FORCE_VGGT_LONG_WEIGHTS} -eq 1 ]]; then
    echo "Forcing VGGT-Long weights download because --force-vggt-long-weights was provided."
  elif vggt_long_weights_present; then
    echo "Skipping VGGT-Long weights download; existing weights look complete in ${VGGT_LONG_WEIGHTS_DIR}."
    return
  fi

  echo "Downloading VGGT-Long weights into ${VGGT_LONG_WEIGHTS_DIR} ..."
  (
    cd "${VGGT_LONG_DIR}"
    bash ./scripts/download_weights.sh
  )
  touch "${VGGT_LONG_WEIGHTS_STAMP}"
}

remove_gsplat_leaked_top_level_packages() {
  # The pinned gsplat wheel ships top-level `tests` and `experimental`
  # packages into site-packages (upstream packaging bug). The leaked `tests`
  # shadows this repo's own tests package and breaks pytest collection, so
  # remove every top-level package gsplat's RECORD claims outside its own
  # namespace. Deleting RECORD-listed files makes a later pip uninstall warn
  # about missing files; that is harmless.
  python - <<'PY'
import shutil
import sysconfig
from importlib.metadata import distribution
from pathlib import Path

site = Path(sysconfig.get_paths()["purelib"])
files = distribution("gsplat").files or []
removed = set()
for f in files:
    top = f.parts[0]
    if top.startswith("gsplat") or top.startswith("."):
        continue
    target = site / top
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        removed.add(top)
if removed:
    print(f"Removed gsplat-leaked top-level packages: {sorted(removed)}")
PY
}

install_gsplat() {
  # gsplat is installed with --no-deps below so its torch/numpy declarations
  # cannot fight the locked tree. Its only declared deps not already in the
  # lock are jaxtyping and ninja — install them explicitly, or the env fails
  # `pip check` after every gsplat install. ninja also parallelizes the CUDA
  # source build.
  python -m pip install ninja jaxtyping

  # Idempotency guard: the from-source CUDA build is the single most expensive
  # step in this installer (minutes at MAX_JOBS=4), and pip's wheel cache does
  # not help VCS builds — skip when the pinned COMMIT is already installed.
  # The comparison must be the VCS SHA from pip's direct_url.json, NOT
  # __version__: different gsplat commits report the same version string, so a
  # version check can never correct a wrong-SHA install.
  local installed_sha
  installed_sha="$(python - <<'PY' 2>/dev/null || true
import json
from importlib.metadata import distribution
try:
    dist = distribution("gsplat")
    url_text = dist.read_text("direct_url.json")
    print(json.loads(url_text)["vcs_info"]["commit_id"] if url_text else "")
except Exception:
    print("")
PY
)"
  if [[ "${installed_sha}" == "${GSPLAT_REF}" ]]; then
    echo "gsplat already installed at pinned commit ${GSPLAT_REF:0:12} — skipping source build."
    remove_gsplat_leaked_top_level_packages
    return
  fi
  [[ -n "${installed_sha}" ]] && echo "gsplat installed at ${installed_sha:0:12}, pin is ${GSPLAT_REF:0:12} — rebuilding."
  local spec="git+${GSPLAT_REPO}"
  [[ -n "${GSPLAT_REF}" ]] && spec="${spec}@${GSPLAT_REF}"
  echo "Building gsplat from source for sm_120 (TORCH_CUDA_ARCH_LIST=${GSPLAT_ARCH_LIST})..."
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST="${GSPLAT_ARCH_LIST}" \
  MAX_JOBS="${MAX_JOBS:-4}" \
    python -m pip install --no-build-isolation --force-reinstall --no-deps "${spec}"
  remove_gsplat_leaked_top_level_packages
}

ensure_modern_conda_compilers() {
  if [[ $USE_VENV -eq 1 ]]; then
    local current_compiler="${CXX:-c++}"
    local current_major
    current_major="$(compiler_major_version "${current_compiler}")"

    if [[ -z "${current_major}" || "${current_major}" -lt "${REQUIRED_GXX_MAJOR}" ]]; then
      echo "PyTorch CUDA-extension builds require GCC/G++ ${REQUIRED_GXX_MAJOR}+." >&2
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

  echo "Current compiler is too old for PyTorch CUDA-extension builds."
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
    local torch_cuda_version=""
    local local_nvcc_version=""
    torch_cuda_version="$(detect_torch_cuda_version | tr -d '[:space:]')"
    local_nvcc_version="$(nvcc_cuda_version | tr -d '[:space:]')"
    echo "Skipping xformers automatic install."
    echo "Reason: xformers often needs a local CUDA toolkit/build chain aligned with torch."
    echo "Current torch CUDA: ${torch_cuda_version:-unknown}; local nvcc CUDA: ${local_nvcc_version:-not detected}."
    echo "Install xformers manually later only if VGGT-Long actually needs it."
  fi

  if grep -Eqi '^[[:space:]]*faiss-gpu([[:space:]]*(==|=|>=|<=|!=|~=).*)?[[:space:]]*$' "${req_file}"; then
    echo "Skipping faiss-gpu from pip requirements."
    echo "If you need it later, install it separately via conda."
  fi
}

smoke_test_python_module() {
  local label="$1"
  local module_name="$2"
  local dist_name="${3:-$2}"
  local extra_path="${4:-}"

  python - "$label" "$module_name" "$dist_name" "$extra_path" <<'PYCODE'
import importlib
import importlib.metadata as metadata
import sys

label, module_name, dist_name, extra_path = sys.argv[1:5]
if extra_path:
    sys.path.insert(0, extra_path)
module = importlib.import_module(module_name)
try:
    version = metadata.version(dist_name)
except Exception:
    version = getattr(module, '__version__', 'unknown')
location = getattr(module, '__file__', None)
if location is None:
    path = getattr(module, '__path__', None)
    if path:
        try:
            location = next(iter(path))
        except Exception:
            location = '<namespace>'
print(f"[smoke-test] OK {label}: module={module_name} version={version} path={location or '<unknown>'}")
PYCODE
}

smoke_test_gsplat_rasterization() {
  # An import-only check would pass a broken build — rasterize a trivial scene
  # so the sm_120 CUDA kernels the source build exists for actually execute.
  python - <<'PYCODE'
import torch
import gsplat

assert torch.cuda.is_available(), "gsplat smoke test needs a CUDA device"
n = 64
means = torch.randn(n, 3, device="cuda")
quats = torch.randn(n, 4, device="cuda")
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(n, 3, device="cuda") * 0.1
opacities = torch.rand(n, device="cuda")
colors = torch.rand(n, 3, device="cuda")
viewmat = torch.eye(4, device="cuda")[None]
K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device="cuda")
rgb, alpha, _ = gsplat.rasterization(
    means, quats, scales, opacities, colors, viewmat, K, 128, 128
)
assert rgb.shape == (1, 128, 128, 3), f"unexpected output shape {tuple(rgb.shape)}"
version = getattr(gsplat, "__version__", "unknown")
print(f"[smoke-test] OK gsplat rasterization: version={version} output={tuple(rgb.shape)}")
PYCODE
}

run_post_install_smoke_tests() {
  echo
  echo "Running post-install smoke tests..."
  smoke_test_python_module "room4d" "room4d" "room4d"

  if [[ $INSTALL_DETECTRON2 -eq 1 ]]; then
    smoke_test_python_module "detectron2" "detectron2" "detectron2"
  fi
  if [[ $INSTALL_VGGT -eq 1 ]]; then
    smoke_test_python_module "VGGT" "vggt" "vggt"
  fi
  if [[ $INSTALL_COTRACKER -eq 1 ]]; then
    smoke_test_python_module "CoTracker" "cotracker" "cotracker"
  fi
  if [[ $INSTALL_CALIBRATION -eq 1 ]]; then
    smoke_test_python_module "pycolmap" "pycolmap" "pycolmap"
    smoke_test_python_module "hloc" "hloc" "hloc"
  fi
  if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
    smoke_test_python_module "VGGT-Long" "vggt_long" "vggt-long" "${VGGT_LONG_DIR}"
  fi
  if [[ $INSTALL_GSPLAT -eq 1 ]]; then
    smoke_test_gsplat_rasterization
  fi
}

setup_python_env

# Install the locked pip tree first (default when the lock exists): the
# editable room4d install below then finds every dependency already satisfied
# at the exact known-good version instead of re-resolving ranges.
if [[ $UNLOCKED -eq 0 && -f "${ROOM4D_LOCK_FILE}" ]]; then
  echo "Installing pinned dependency tree from ${ROOM4D_LOCK_FILE} (use --unlocked to re-resolve)..."
  python -m pip install -r "${ROOM4D_LOCK_FILE}" \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "https://pypi.nvidia.com"
elif [[ $UNLOCKED -eq 0 ]]; then
  echo "No lockfile at ${ROOM4D_LOCK_FILE} — falling back to fresh resolution."
fi

EXTRAS=()
[[ $INSTALL_VISION -eq 1 ]] && EXTRAS+=(vision)
[[ $INSTALL_GEOMETRY -eq 1 ]] && EXTRAS+=(geometry)
[[ $INSTALL_CALIBRATION -eq 1 ]] && EXTRAS+=(calibration)
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

if [[ $INSTALL_CALIBRATION -eq 1 ]]; then
  clone_or_update_checkout "${HLOC_REPO}" "${HLOC_DIR}" "${HLOC_TAG}"
  if [[ -f "${HLOC_DIR}/.gitmodules" ]]; then
    git -C "${HLOC_DIR}" submodule update --init --recursive
  fi
  echo "Installing hloc from ${HLOC_DIR} (tag ${HLOC_TAG})..."
  python -m pip install -e "${HLOC_DIR}"
fi

if [[ $INSTALL_VGGT_LONG -eq 1 ]]; then
  ensure_modern_conda_compilers
  clone_or_update_checkout "${VGGT_LONG_REPO}" "${VGGT_LONG_DIR}" "${VGGT_LONG_COMMIT}"
  install_vggt_long_requirements
  ensure_cuda_toolkit_for_torch_extensions

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

  download_vggt_long_weights_if_needed

  echo "Installing VGGT-Long from ${VGGT_LONG_DIR}..."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation -e "${VGGT_LONG_DIR}"
fi

if [[ $INSTALL_GSPLAT -eq 1 ]]; then
  ensure_modern_conda_compilers
  ensure_cuda_toolkit_for_torch_extensions
  install_gsplat
fi

run_post_install_smoke_tests

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
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- calibration extras from pyproject.toml (pycolmap ${PYCOLMAP_VERSION}, matching COLMAP ${COLMAP_UPSTREAM_TAG})" )
$( [[ $INSTALL_CALIBRATION -eq 1 ]] && echo "- hloc from tag ${HLOC_TAG} at ${HLOC_DIR}" )
$( [[ $INSTALL_VGGT_LONG -eq 1 ]] && echo "- VGGT-Long from commit ${VGGT_LONG_COMMIT} at ${VGGT_LONG_DIR}" )
$( [[ $INSTALL_GSPLAT -eq 1 ]] && echo "- gsplat built from source for sm_120 at commit ${GSPLAT_REF}" )

Notes:
- third_party/ should be gitignored so external source checkouts are never tracked.
- .room4d_cache/ should be gitignored too.
- Calibration mode is pinned to hloc ${HLOC_TAG} and pycolmap ${PYCOLMAP_VERSION}
  (aligned with upstream COLMAP ${COLMAP_UPSTREAM_TAG}).
- Post-install smoke tests verify the third-party repos and pinned Python packages
  this script installs directly into the active environment.
- For Intel RealSense SDK / pyrealsense2 on Ubuntu, see:
    scripts/setup_realsense_sdk.sh
- For a first RGB-only camera test without RealSense SDK, you can use:
    python scripts/capture_uvc_video.py --device 0 --output out.mp4
EOF

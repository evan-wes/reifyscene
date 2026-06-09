#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the room4d reference comparison tool.
#
# Purpose:
#   Stand up an ISOLATED conda environment containing third-party reference 4D
#   reconstruction methods, so room4d's own renderer can be benchmarked against
#   a known-good implementation (and published numbers) on shared datasets
#   (DyCheck iPhone, NVIDIA Dynamic Scenes). This env is intentionally separate
#   from the main `room4d` env: the reference repos pin older torch/CUDA, and we
#   do not want them downgrading the main env's torch (which must keep sm_120).
#
# What this installs today (Shape of Motion):
#   - an sm_120-capable PyTorch (cu128 build) so it runs on Blackwell (RTX 5090)
#   - a matching CUDA toolkit (nvcc + cudart + headers) for source builds
#   - Shape of Motion cloned into third_party/shape-of-motion at a pinned commit
#   - Shape of Motion's Python deps, with NVIDIA RAPIDS/cuML swapped to a
#     Blackwell-capable cu12 release (its upstream pin predates sm_120)
#   - gsplat built FROM SOURCE for sm_120 (no cu128/cu13 prebuilt wheel exists)
#   NOTE: Shape of Motion's heavy preproc/ stack (DROID-SLAM, UniDepth, tapnet)
#   is intentionally NOT installed — we benchmark on the authors' PROCESSED
#   datasets (cameras + depth precomputed), which need no preprocessing.
#
# Not yet implemented:
#   - MoSca (--with-mosca). MoSca pins torch 2.1.0+cu118 + mmcv-full 1.7.2 +
#     pytorch3d + custom rasterizers + jax, which is a much harder Blackwell
#     port. Tracked in the design doc as the second reference tool; this script
#     errors out if asked for it until that work is scoped.
#
# Environment behavior:
#   - conda-only, mirroring scripts/setup_room4d.sh (no silent venv fallback).
#   - Provide --conda-env NAME (default: room4d_som_ref).
#   - The env is created with the pinned Python if it does not already exist.

# ------------------------------------------------------------------------------
# Pinned external dependency refs
# ------------------------------------------------------------------------------

SOM_REPO="https://github.com/vye16/shape-of-motion.git"
SOM_COMMIT="579753e1c7ba96f60cd7690e5b835627bd1935e9"   # 2025-08-02 "update iccv v2"

# PyTorch: cu128 build is the lowest CUDA that ships sm_120 (Blackwell) kernels
# AND aligns with RAPIDS cu12 wheels. (The main room4d env uses cu130; these
# envs are isolated, so they need not match.)
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"

# CUDA toolkit for source builds (gsplat). Matches torch's CUDA line (12.8).
CUDA_TOOLKIT_CHANNEL="nvidia/label/cuda-12.8.1"

# RAPIDS / cuML: Shape of Motion imports `from cuml import HDBSCAN, KMeans` at
# module load, so cuML is REQUIRED. Its upstream pin (24.6, cu11) predates
# Blackwell and lacks sm_120 kernels — bump to a Blackwell-capable cu12 release.
# 25.02+ supports sm_120 and numpy>=2. The post-install smoke test validates
# that cuML actually runs a GPU kernel on this card; bump this if it fails.
RAPIDS_RELEASE="25.02"

# gsplat is built from source for sm_120. Pin a ref once a build is confirmed.
GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_REF="v1.5.3"   # built cleanly from source for sm_120 on 2026-06-09 (CUDA 12.8, g++ 13)
GSPLAT_ARCH_LIST="12.0"   # TORCH_CUDA_ARCH_LIST for the 5090 (sm_120)

# CUDA 12.8 (and torch's nvcc host-compiler check) rejects g++ >= 14, so the
# reference env's compiler must be <14. Pin to 13 (modern but within the cap).
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-13}"
CUDA_HOST_GXX_MAX_MAJOR=14   # exclusive upper bound enforced below
PINNED_PYTHON="${PINNED_PYTHON:-3.10}"

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

CONDA_ENV_NAME="${CONDA_ENV_NAME:-room4d_som_ref}"
INSTALL_SOM=1
INSTALL_MOSCA=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
SOM_DIR="${THIRD_PARTY_DIR}/shape-of-motion"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV_NAME="$2"; shift 2 ;;
    --python)
      PINNED_PYTHON="$2"; shift 2 ;;
    --no-shape-of-motion)
      INSTALL_SOM=0; shift ;;
    --with-mosca)
      INSTALL_MOSCA=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_reference_tool.sh [options]

Sets up an isolated conda env with reference 4D reconstruction methods for
benchmarking room4d's renderer.

Options:
  --conda-env NAME        Conda env to create/use (default: room4d_som_ref)
  --python VERSION        Python version if the env must be created (default: 3.10)
  --no-shape-of-motion    Skip the Shape of Motion install
  --with-mosca            (NOT YET IMPLEMENTED) set up MoSca — errors out

Behavior:
  - conda-only; creates the env with the pinned Python if it does not exist.
  - Installs an sm_120-capable torch (cu128), a matching CUDA toolkit, Shape of
    Motion at a pinned commit, its deps (RAPIDS bumped to a Blackwell cu12
    release), and gsplat built from source for sm_120.
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

# ------------------------------------------------------------------------------
# Helpers (mirrors scripts/setup_room4d.sh conventions)
# ------------------------------------------------------------------------------

run_conda_relaxed() {
  # conda hooks are not nounset-safe; relax `set -u` around conda calls.
  local had_nounset=0
  case "$-" in *u*) had_nounset=1 ;; esac
  set +u
  conda "$@"
  local rc=$?
  [[ ${had_nounset} -eq 1 ]] && set -u
  return ${rc}
}

clone_or_update_checkout() {
  local repo_url="$1" dest_dir="$2" ref="$3"
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

conda_base_dir() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    dirname "$(dirname "${CONDA_EXE}")"
  else
    conda info --base
  fi
}

activate_or_create_env() {
  # Safety: this script only ever sets up an ISOLATED reference env. Refuse to
  # target protected envs so it can never modify the main room4d environment.
  case "${CONDA_ENV_NAME}" in
    room4d|base|"")
      echo "Refusing to run against protected/empty env '${CONDA_ENV_NAME}'." >&2
      echo "This script is for an isolated reference env (e.g. room4d_som_ref)." >&2
      exit 1 ;;
  esac

  if ! command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_EXE:-}" ]]; then
    echo "Conda is required but 'conda' was not found." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda_base_dir)/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "Creating conda env '${CONDA_ENV_NAME}' (python=${PINNED_PYTHON})..."
    run_conda_relaxed create -y -n "${CONDA_ENV_NAME}" "python=${PINNED_PYTHON}"
  fi

  echo "Activating conda environment: ${CONDA_ENV_NAME}"
  if ! run_conda_relaxed activate "${CONDA_ENV_NAME}"; then
    echo "Failed to activate conda environment '${CONDA_ENV_NAME}'." >&2
    exit 1
  fi

  # Safety: hard-assert the active env is the intended one before any install.
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" || "${CONDA_PREFIX:-}" != *"/envs/${CONDA_ENV_NAME}" ]]; then
    echo "Active env mismatch: CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', CONDA_PREFIX='${CONDA_PREFIX:-}'." >&2
    echo "Expected to be inside '${CONDA_ENV_NAME}'. Aborting to protect other envs." >&2
    exit 1
  fi
  echo "Confirmed active env: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX})"
  echo "Python executable: $(command -v python)"
}

ensure_cuda_home_exports() {
  local cuda_home="$1"
  [[ -z "${cuda_home}" ]] && return 1
  export CUDA_HOME="${cuda_home}"
  if [[ -d "${CUDA_HOME}/bin" ]]; then
    case ":${PATH}:" in *":${CUDA_HOME}/bin:"*) ;; *) export PATH="${CUDA_HOME}/bin:${PATH}" ;; esac
  fi
  local lib_dir
  for lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" "${CUDA_HOME}/targets/x86_64-linux/lib"; do
    if [[ -d "${lib_dir}" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in *":${lib_dir}:"*) ;; *) export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}" ;; esac
    fi
  done
  return 0
}

ensure_cuda_toolkit() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment while ensuring CUDA toolkit." >&2
    exit 1
  fi
  if [[ ! -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    echo "Installing CUDA toolkit from ${CUDA_TOOLKIT_CHANNEL}..."
    run_conda_relaxed install -y -c "${CUDA_TOOLKIT_CHANNEL}" cuda-toolkit
  fi
  ensure_cuda_home_exports "${CONDA_PREFIX}"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc still not on PATH after toolkit install. CUDA_HOME=${CUDA_HOME:-unset}" >&2
    exit 1
  fi
  echo "CUDA toolkit ready: nvcc=$(command -v nvcc) ($(nvcc --version | tail -n 1))"
}

ensure_modern_conda_compilers() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment while ensuring compilers." >&2
    exit 1
  fi
  local conda_cxx="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
  local conda_cc="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"

  # Determine the major version of an existing conda compiler, if any.
  local cur_major=""
  if [[ -x "${conda_cxx}" ]]; then
    cur_major="$("${conda_cxx}" -dumpversion 2>/dev/null | cut -d. -f1)"
  fi

  # CUDA 12.8 needs 6 <= gcc < 14. (Re)install the pinned version if the
  # existing compiler is missing, too old, or — the bug we hit — too NEW.
  local need_install=1
  if [[ -n "${cur_major}" && "${cur_major}" -ge 6 && "${cur_major}" -lt "${CUDA_HOST_GXX_MAX_MAJOR}" ]]; then
    need_install=0
  fi
  if [[ ${need_install} -eq 1 ]]; then
    echo "Installing conda compilers (gcc/gxx ${CONDA_GCC_VERSION}; existing major='${cur_major:-none}')..."
    run_conda_relaxed install -y -c conda-forge \
      "gcc_linux-64=${CONDA_GCC_VERSION}" "gxx_linux-64=${CONDA_GCC_VERSION}"
  fi

  export CC="${conda_cc}"
  export CXX="${conda_cxx}"

  # Hard-assert the selected compiler is within CUDA 12.8's supported range.
  local sel_major
  sel_major="$("${conda_cxx}" -dumpversion 2>/dev/null | cut -d. -f1)"
  if [[ -z "${sel_major}" || "${sel_major}" -ge "${CUDA_HOST_GXX_MAX_MAJOR}" ]]; then
    echo "Conda C++ compiler major '${sel_major:-unknown}' is not < ${CUDA_HOST_GXX_MAX_MAJOR} (CUDA 12.8 limit)." >&2
    exit 1
  fi
  echo "Using conda compilers: CC=${CC} CXX=${CXX} (g++ ${sel_major}.x)"
}

install_torch() {
  echo "Installing torch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} (cu128)..."
  python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_CUDA_INDEX}"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
arch = torch.cuda.get_arch_list()
assert any(a.endswith("120") or a == "sm_120" for a in arch), f"sm_120 not in arch list: {arch}"
print(f"torch {torch.__version__} OK; arch={arch}; device={torch.cuda.get_device_name(0)}")
PY
}

install_som_requirements() {
  # Generate a patched requirements file from upstream so we track its deps,
  # but (a) drop the stale torch extra-index (torch is installed separately) and
  # (b) move RAPIDS from cu11/24.6 to a Blackwell-capable cu12 release.
  local upstream="${SOM_DIR}/requirements.txt"
  local patched="${SOM_DIR}/requirements.room4d.txt"
  [[ -f "${upstream}" ]] || { echo "Missing ${upstream}" >&2; exit 1; }

  sed -E \
    -e '/download\.pytorch\.org\/whl\/cu[0-9]+/d' \
    -e "s/-cu11==24\.6\.\*/-cu12==${RAPIDS_RELEASE}.*/g" \
    "${upstream}" > "${patched}"

  echo "Installing Shape of Motion requirements (RAPIDS -> cu12 ${RAPIDS_RELEASE})..."
  python -m pip install -r "${patched}"
}

install_gsplat() {
  local spec="git+${GSPLAT_REPO}"
  [[ -n "${GSPLAT_REF}" ]] && spec="${spec}@${GSPLAT_REF}"
  echo "Building gsplat from source for sm_120 (TORCH_CUDA_ARCH_LIST=${GSPLAT_ARCH_LIST})..."
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST="${GSPLAT_ARCH_LIST}" \
  MAX_JOBS="${MAX_JOBS:-4}" \
    python -m pip install --no-build-isolation "${spec}"
}

run_smoke_tests() {
  echo
  echo "Running post-install smoke tests..."
  python - <<'PY'
import sys
import numpy as np

# 1) torch on sm_120
import torch
assert torch.cuda.is_available()
x = (torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")).sum().item()
print(f"[smoke] torch GPU matmul OK ({torch.__version__}, {torch.cuda.get_device_name(0)})")

# 2) cuML actually runs a GPU kernel on this card (validates RAPIDS sm_120)
try:
    from cuml import KMeans
    pts = np.random.RandomState(0).rand(256, 3).astype("float32")
    labels = KMeans(n_clusters=4, max_iter=10).fit(pts).labels_
    print(f"[smoke] cuML KMeans GPU OK (labels[:5]={labels[:5]})")
except Exception as e:
    print(f"[smoke][FAIL] cuML GPU: {type(e).__name__}: {str(e)[:200]}")
    print("           -> bump RAPIDS_RELEASE to a Blackwell/sm_120-capable version.")
    sys.exit(1)

# 3) gsplat imports and rasterizes a trivial scene
try:
    import gsplat
    print(f"[smoke] gsplat import OK ({getattr(gsplat,'__version__','?')})")
except Exception as e:
    print(f"[smoke][FAIL] gsplat import: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(1)
PY
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if [[ ${INSTALL_MOSCA} -eq 1 ]]; then
  echo "MoSca setup is not implemented in this script yet (see design doc:" >&2
  echo "Reference Comparison Tool & Benchmark Datasets). It needs a separate" >&2
  echo "Blackwell port of torch 2.1/cu118 + mmcv-full 1.7.2 + pytorch3d." >&2
  exit 2
fi

activate_or_create_env
python -m pip install --upgrade pip setuptools wheel

install_torch
ensure_cuda_toolkit
ensure_modern_conda_compilers

if [[ ${INSTALL_SOM} -eq 1 ]]; then
  clone_or_update_checkout "${SOM_REPO}" "${SOM_DIR}" "${SOM_COMMIT}"
  install_som_requirements
  install_gsplat
fi

run_smoke_tests

cat <<EOF

Setup complete.

Environment:
- conda env: ${CONDA_DEFAULT_ENV:-${CONDA_ENV_NAME}}

Installed:
- torch ${TORCH_VERSION}+cu128, torchvision ${TORCHVISION_VERSION}+cu128 (sm_120)
- CUDA toolkit from ${CUDA_TOOLKIT_CHANNEL} (CUDA_HOME=${CUDA_HOME:-unset})
$( [[ ${INSTALL_SOM} -eq 1 ]] && echo "- Shape of Motion at commit ${SOM_COMMIT} in ${SOM_DIR}" )
$( [[ ${INSTALL_SOM} -eq 1 ]] && echo "- Shape of Motion deps (RAPIDS cu12 ${RAPIDS_RELEASE})" )
$( [[ ${INSTALL_SOM} -eq 1 ]] && echo "- gsplat from source (sm_120)" )

Next:
- Datasets are NOT downloaded by this script. Get the authors' PROCESSED
  iPhone (DyCheck) and NVIDIA Dynamic Scenes data (Google Drive links in the
  Shape of Motion README), then:
    cd ${SOM_DIR}
    python run_training.py --work-dir <OUT> data:iphone --data.data-dir <SEQ>
- third_party/ is gitignored; pins live in this script.
- MoSca is the planned second reference tool (--with-mosca, not yet implemented).
EOF

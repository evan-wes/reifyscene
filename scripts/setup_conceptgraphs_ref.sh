#!/usr/bin/env bash
set -euo pipefail
# Reproducible installer for the ConceptGraphs reference comparison.
#
# Purpose:
#   Stand up an ISOLATED conda environment running ConceptGraphs, so reifyscene's entity
#   inventory can be compared against a published open-vocabulary 3D scene graph on the
#   same footage, scored by the same DAVIS ground truth. ConceptGraphs is the closest
#   released instance of the choice reifyscene makes -- explicit entities with relations,
#   rather than a feature field -- and it answers the naming question the other way:
#   it CAPTIONS each object with an LVLM instead of scoring it against a candidate list,
#   which is the non-circular approach the vocabulary experiments here kept running into.
#
# Why a separate env, and why not upstream's pins:
#   Upstream recommends torch 2.0.1 + cu118 + a binary pytorch3d built for that exact
#   pair. cu118 ships no sm_120 kernels, so it cannot run on this machine's RTX 5090 at
#   all -- and a separate env does not change what the GPU is. This follows the same
#   pattern setup_reference_tool.sh uses for Shape of Motion instead: an isolated env
#   with a MODERN sm_120-capable torch, and anything that needs compiling built from
#   source against it. Checking the imports, that is viable here: the ConceptGraphs
#   pipeline itself needs open3d + torch + open_clip + ultralytics, and pytorch3d appears
#   in upstream's recipe rather than in gradslam's own requirements.
#
# Known source build:
#   chamferdist -- a CUDA extension, the chamfer distance gradslam uses. Same situation
#   as gsplat in the Shape of Motion setup: no wheel for this arch, build for sm_120.
#
# Not handled here (deliberately):
#   The DAVIS dataloader. ConceptGraphs takes POSED RGB-D and DAVIS ships neither depth
#   nor poses, so the comparison feeds it VGGT's. That adapter is Python, belongs in the
#   repo rather than in an installer, and is written separately.
#
# Usage:
#   scripts/setup_conceptgraphs_ref.sh --create-env          # first run
#   scripts/setup_conceptgraphs_ref.sh                       # re-run into an existing env

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reifyscene_cg_ref}"
PINNED_PYTHON="${PINNED_PYTHON:-3.10}"
CREATE_ENV=0

# Pinned so a rebuild gets the same code. Bump deliberately, not by drift.
CG_REPO="https://github.com/concept-graphs/concept-graphs.git"
CG_COMMIT="93277a02bd89171f8121e84203121cf7af9ebb5d"   # main @ 2026-09-17
CG_DIR="${REPO_ROOT}/third_party/concept-graphs"

# cu128 is the lowest CUDA that ships sm_120 kernels; matches the Shape of Motion env.
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_ARCH_LIST="12.0"
# CUDA 12.8 rejects g++ >= 14 as a host compiler; 13 is modern and under the cap.
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-13}"
# Pinned to torch's CUDA line. A bare `-c nvidia` resolves to the newest toolkit (13.x),
# which torch's extension builder rejects against a cu128 build -- the same channel
# setup_reference_tool.sh pins, for the same reason.
CUDA_TOOLKIT_CHANNEL="${CUDA_TOOLKIT_CHANNEL:-nvidia/label/cuda-12.8.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --create-env) CREATE_ENV=1; shift ;;
    --conda-env)  CONDA_ENV_NAME="$2"; shift 2 ;;
    --python)     PINNED_PYTHON="$2"; shift 2 ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Safety: only ever an isolated reference env. Never the main one.
case "${CONDA_ENV_NAME}" in
  reifyscene|reifyscene_som_ref|base|"")
    echo "Refusing to run against protected env '${CONDA_ENV_NAME}'." >&2
    echo "This script sets up an isolated reference env (e.g. reifyscene_cg_ref)." >&2
    exit 1 ;;
esac

run_conda_relaxed() {
  # conda's hooks are not nounset-safe -- cuda-nvcc's activate.d reads
  # NVCC_PREPEND_FLAGS with no default, which aborts the script under `set -u`. Same
  # wrapper setup_reference_tool.sh uses, for the same reason.
  local had_nounset=0
  case "$-" in *u*) had_nounset=1 ;; esac
  set +u
  conda "$@"
  local rc=$?
  [[ ${had_nounset} -eq 1 ]] && set -u
  return ${rc}
}

conda_base_dir() { conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE}")"; }
command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda_base_dir)/etc/profile.d/conda.sh"

if ! run_conda_relaxed env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  if [[ ${CREATE_ENV} -eq 0 ]]; then
    echo "Conda env '${CONDA_ENV_NAME}' does not exist. Re-run with --create-env." >&2
    exit 1
  fi
  echo "== creating env ${CONDA_ENV_NAME} (python=${PINNED_PYTHON})"
  run_conda_relaxed create -y -n "${CONDA_ENV_NAME}" "python=${PINNED_PYTHON}"
fi

run_conda_relaxed activate "${CONDA_ENV_NAME}"
# Hard-assert before any install: a mis-activated env must never be written to.
[[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]] || {
  echo "Active env is '${CONDA_DEFAULT_ENV:-none}', expected '${CONDA_ENV_NAME}'." >&2
  exit 1
}
set +u   # activation scripts stay in effect for the rest of the run
echo "== active env: ${CONDA_DEFAULT_ENV}  ($(python -V 2>&1))"

echo "== torch ${TORCH_VERSION} (cu128, sm_120)"
pip install --index-url "${TORCH_CUDA_INDEX}" "torch==${TORCH_VERSION}" torchvision

echo "== ConceptGraphs @ ${CG_COMMIT:0:12}"
mkdir -p "$(dirname "${CG_DIR}")"
if [[ -d "${CG_DIR}/.git" ]]; then git -C "${CG_DIR}" fetch --quiet origin
else git clone --quiet "${CG_REPO}" "${CG_DIR}"; fi
git -C "${CG_DIR}" checkout --quiet "${CG_COMMIT}"

echo "== python dependencies"
pip install open3d open_clip_torch ultralytics supervision \
            hydra-core tyro distinctipy natsort imageio kornia plotly \
            opencv-python-headless h5py

# A CUDA extension needs nvcc and a host compiler nvcc will accept, in the env --
# exactly what setup_reference_tool.sh does for gsplat. CUDA 12.8 rejects g++ >= 14, so
# the conda toolchain is pinned below that.
echo "== CUDA toolkit + compilers (for the chamferdist source build)"
# Version-pinned, not just channel-pinned: conda leaves an already-satisfied
# `cuda-toolkit` alone, so a newer toolkit left behind by an interrupted run is never
# downgraded and the build fails much later with a version mismatch.
run_conda_relaxed install -y -c "${CUDA_TOOLKIT_CHANNEL}" "cuda-toolkit=${CUDA_VERSION}"
run_conda_relaxed install -y -c conda-forge "gxx_linux-64=${CONDA_GCC_VERSION}" "gcc_linux-64=${CONDA_GCC_VERSION}"
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not on PATH after toolkit install" >&2; exit 1; }
echo "   nvcc: $(nvcc --version | tail -2 | head -1)"
echo "   CXX:  $(${CXX} --version | head -1)"

# Assert before compiling anything. torch refuses to build an extension whose nvcc
# major.minor differs from the CUDA it was built against, and it only says so after the
# toolkit download and a compile attempt -- this turns that into an immediate, named
# failure with the fix in it.
_torch_cuda="$(python -c 'import torch; print(torch.version.cuda or "none")')"
_nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ "${_nvcc_cuda}" != "${_torch_cuda}" ]]; then
  echo "!! nvcc is CUDA ${_nvcc_cuda} but torch was built against ${_torch_cuda}." >&2
  echo "!! An interrupted earlier run can leave a newer toolkit that conda will not" >&2
  echo "!! downgrade. Remove it and re-run:" >&2
  echo "!!   conda remove -n ${CONDA_ENV_NAME} --force cuda-toolkit cuda-nvcc" >&2
  exit 1
fi
echo "   CUDA ${_nvcc_cuda} matches torch"

echo "== chamferdist (source build for sm_${TORCH_ARCH_LIST/./})"
# --no-build-isolation: its setup.py imports torch, which an isolated build env lacks.
CUDA_HOME="${CUDA_HOME}" CC="${CC}" CXX="${CXX}" \
TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
  pip install --no-build-isolation chamferdist || {
  echo "!! chamferdist failed to build. It is the one CUDA extension here; same" >&2
  echo "!! situation as gsplat in the Shape of Motion setup -- check nvcc and the" >&2
  echo "!! host compiler above, then re-run." >&2
  exit 1
}

# gradslam's own requirements pin open3d==0.10.0.0 -- a 2020 release PyPI no longer
# carries, and one that could not coexist with this stack anyway. Everything it actually
# imports (torch, kornia, imageio, natsort, plotly, opencv, open3d) is installed above at
# modern versions, so it goes in without its dependency list rather than with a broken one.
echo "== gradslam (--no-deps; its open3d pin is from 2020)"
CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
  pip install --no-build-isolation --no-deps \
    "git+https://github.com/gradslam/gradslam.git@main" || {
  echo "!! gradslam failed even with --no-deps." >&2
  exit 1
}

echo
echo "== installed. sanity:"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"arch={torch.cuda.get_arch_list()[-3:] if torch.cuda.is_available() else 'n/a'}")
for m in ("open3d", "open_clip", "ultralytics", "gradslam"):
    try:
        __import__(m); print(f"  {m}: ok")
    except Exception as exc:
        print(f"  {m}: MISSING ({type(exc).__name__})")
PY

#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for the reifyscene reference comparison tool.
#
# Purpose:
#   Stand up an ISOLATED conda environment containing third-party reference 4D
#   reconstruction methods, so reifyscene's own renderer can be benchmarked against
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
#   - conda-only, mirroring scripts/setup_reifyscene.sh (no silent venv fallback).
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

# Deps Shape of Motion imports but does NOT list in its requirements.txt.
# Pinned to the versions verified working on 2026-06-09.
SOM_EXTRA_PIP="tyro==1.0.13 open3d==0.19.0"

# gsplat is built from source for sm_120 at the pinned ref below. Pinned to an
# immutable commit SHA — tags are mutable and a re-pointed tag would execute
# different build code on the next setup run. NOTE: this is the main-branch
# commit that was ACTUALLY built and validated on this machine (2026-06-09,
# CUDA 12.8, g++ 13) and that produced the recorded paper-windmill baseline;
# it reports __version__ 1.5.3 like the v1.5.3 tag but is a few commits ahead.
GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_REF="4e52698e45eaaed929ed3a5065e96a688d085df6"
GSPLAT_ARCH_LIST="12.0"   # TORCH_CUDA_ARCH_LIST for the 5090 (sm_120)

# CUDA 12.8 (and torch's nvcc host-compiler check) rejects g++ >= 14, so the
# reference env's compiler must be <14. Pin to 13 (modern but within the cap).
CONDA_GCC_VERSION="${CONDA_GCC_VERSION:-13}"
CUDA_HOST_GXX_MAX_MAJOR=14   # exclusive upper bound enforced below
PINNED_PYTHON="${PINNED_PYTHON:-3.10}"

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

# Must match DEFAULT_CONDA_ENV in scripts/run_reference_benchmark.py.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-room4d_som_ref}"
INSTALL_SOM=1
INSTALL_MOSCA=0
UNLOCKED=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
SOM_DIR="${THIRD_PARTY_DIR}/shape-of-motion"
LOCK_FILE="${ROOT_DIR}/scripts/locks/room4d_som_ref.lock"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV_NAME="$2"; shift 2 ;;
    --python)
      PINNED_PYTHON="$2"; shift 2 ;;
    --no-shape-of-motion)
      INSTALL_SOM=0; shift ;;
    --unlocked)
      UNLOCKED=1; shift ;;
    --with-mosca)
      INSTALL_MOSCA=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_reference_tool.sh [options]

Sets up an isolated conda env with reference 4D reconstruction methods for
benchmarking reifyscene's renderer.

Options:
  --conda-env NAME        Conda env to create/use (default: room4d_som_ref)
  --python VERSION        Python version if the env must be created (default: 3.10)
  --no-shape-of-motion    Skip the Shape of Motion install
  --unlocked              Resolve dependencies fresh instead of installing from
                          scripts/locks/room4d_som_ref.lock (regenerate locks
                          with scripts/gen_env_locks.sh)
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
# Helpers (mirrors scripts/setup_reifyscene.sh conventions)
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
  # Skip the network fetch when the pinned ref is already resolvable locally —
  # keeps re-runs working offline (the pin is immutable, nothing to update).
  if git -C "${dest_dir}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    echo "Pinned ref ${ref} already present in ${dest_dir} — skipping fetch."
  else
    echo "Fetching updates in ${dest_dir}..."
    git -C "${dest_dir}" fetch origin
  fi
  # Restore tracked files before checkout: this script sed-patches the tree in
  # place, and a pin bump over a dirty tree either fails the checkout or
  # silently carries stale old-pin patches into the new tree. Restoring makes
  # every run deterministic — patches are reapplied fresh afterwards.
  echo "Restoring tracked files in ${dest_dir} (patches are reapplied after checkout)..."
  git -C "${dest_dir}" checkout -- .
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
    reifyscene|base|"")
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
  # Lockfile path (default when the lock exists): replay the exact pip tree of
  # the known-good env — no dependency resolution at all. The lock excludes
  # editable installs and gsplat (built from source separately); torch is
  # already installed at the pinned cu128 build, so its lock line is a no-op.
  # Regenerate with scripts/gen_env_locks.sh after deliberate upgrades.
  if [[ ${UNLOCKED} -eq 0 && -f "${LOCK_FILE}" ]]; then
    echo "Installing reference env from lockfile ${LOCK_FILE} (use --unlocked to re-resolve)..."
    python -m pip install -r "${LOCK_FILE}" --extra-index-url https://pypi.nvidia.com
    return
  fi
  [[ ${UNLOCKED} -eq 0 ]] && echo "No lockfile at ${LOCK_FILE} — falling back to fresh resolution."

  # Fresh-resolution path: generate a patched requirements file from upstream
  # so we track its deps, but (a) drop ALL index lines (torch is installed
  # separately; the NVIDIA index is scoped to the RAPIDS install below so it
  # cannot influence unrelated packages) and (b) split out RAPIDS, moving it
  # from cu11/24.6 to a Blackwell-capable cu12 release.
  local upstream="${SOM_DIR}/requirements.txt"
  local patched="${SOM_DIR}/requirements.reifyscene.txt"
  local rapids="${SOM_DIR}/requirements.reifyscene-rapids.txt"
  [[ -f "${upstream}" ]] || { echo "Missing ${upstream}" >&2; exit 1; }

  sed -E \
    -e '/--extra-index-url/d' \
    -e '/-cu11==24\.6\.\*/d' \
    "${upstream}" > "${patched}"
  sed -nE "s/(.*)-cu11==24\.6\.\*/\1-cu12==${RAPIDS_RELEASE}.*/p" "${upstream}" > "${rapids}"

  echo "Installing Shape of Motion requirements (PyPI only)..."
  python -m pip install -r "${patched}"

  echo "Installing RAPIDS (cu12 ${RAPIDS_RELEASE}) from the NVIDIA index..."
  python -m pip install -r "${rapids}" --extra-index-url https://pypi.nvidia.com

  # Deps that Shape of Motion imports at runtime but forgot to pin in
  # requirements.txt (run_training.py needs tyro; viz/IO needs open3d).
  echo "Installing SoM runtime deps missing from upstream requirements..."
  python -m pip install ${SOM_EXTRA_PIP}
}

require_patched() {
  # sed -i exits 0 even when zero lines match, so every patch must be proven
  # by its sentinel — otherwise a changed upstream (pin bump, dirty checkout)
  # would ship unpatched code that fails late on GPU or, worse for a
  # benchmark tool, runs subtly wrong. min_count matters when several edits
  # share a sentinel string: requiring the exact occurrence count catches the
  # case where only SOME of the edits landed.
  local file="$1" sentinel="$2" what="$3" min_count="${4:-1}"
  local found
  found="$(grep -c "${sentinel}" "${file}" || true)"
  if [[ "${found}" -lt "${min_count}" ]]; then
    echo "PATCH FAILED: ${what} did not (fully) land in ${file} (${found}/${min_count} sentinels)." >&2
    echo "The pinned SoM commit's code no longer matches the sed pattern —" >&2
    echo "update apply_som_patches for the new pin before proceeding." >&2
    exit 1
  fi
}

apply_som_patches() {
  # Shape of Motion targets torch 2.1; torch >=2.11 (needed for sm_120) is
  # stricter about indexing a CUDA tensor with a CPU index. cuML returns cluster
  # labels on CPU, but the points/centers are on CUDA. Put the label tensors on
  # the points' device, which fixes the whole device-mismatch class in
  # motion-basis init (the per-cluster center reduction AND the returned labels
  # consumed by the caller). The sed patterns only match the UNPATCHED form, so
  # re-running is a no-op (idempotent); require_patched proves each patch landed.
  local f="${SOM_DIR}/flow3d/init_utils.py"
  local t="${SOM_DIR}/flow3d/trainer.py"
  for required in "${f}" "${t}"; do
    if [[ ! -f "${required}" ]]; then
      echo "Expected patch target missing: ${required}" >&2
      echo "The SoM checkout at ${SOM_DIR} does not match the pinned commit layout." >&2
      exit 1
    fi
  done

  echo "Applying Shape of Motion torch-device patches to $(basename "${f}")..."
  sed -i \
    -e 's/means_canonical\[torch\.tensor(labels == i)\]/means_canonical[torch.tensor(labels == i, device=means_canonical.device)]/' \
    -e 's/return sampled_centers, num_bases, torch\.tensor(labels)/return sampled_centers, num_bases, torch.tensor(labels, device=means_canonical.device)/' \
    "${f}"
  # Two independent sed edits share this sentinel — require BOTH occurrences
  # so a partial landing (upstream drifted one line but not the other) fails.
  require_patched "${f}" "device=means_canonical.device" "torch device-index patch" 2

  # gsplat >=1.4 returns per-gaussian screen-space radii as (C, G, 2); SoM's
  # densification expects a scalar radius (C, G). Collapse the trailing dim
  # before it's used as a visibility mask.
  if ! grep -q "_current_radii = _current_radii.amax" "${t}"; then
    echo "Patching $(basename "${t}") for gsplat radii shape (C,G,2)->(C,G)..."
    sed -i 's|^\( *\)sel = _current_radii > 0|\1if _current_radii.dim() == 3:  # gsplat>=1.4 radii (C,G,2) -> (C,G)\n\1    _current_radii = _current_radii.amax(dim=-1)\n\1sel = _current_radii > 0|' "${t}"
  fi
  require_patched "${t}" "_current_radii = _current_radii.amax" "gsplat radii patch"
}

install_gsplat() {
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
}

run_smoke_tests() {
  echo
  echo "Running post-install smoke tests..."
  # The cuML and gsplat sections only apply when the SoM stack was installed;
  # with --no-shape-of-motion only the torch check runs (RAPIDS/gsplat are
  # never installed on that path, so requiring them would always fail).
  SMOKE_INSTALL_SOM="${INSTALL_SOM}" python - <<'PY'
import os
import sys
import numpy as np

install_som = os.environ.get("SMOKE_INSTALL_SOM") == "1"

# 1) torch on sm_120 (.item() forces device sync, proving the kernel ran)
import torch
assert torch.cuda.is_available()
(torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")).sum().item()
print(f"[smoke] torch GPU matmul OK ({torch.__version__}, {torch.cuda.get_device_name(0)})")

if not install_som:
    print("[smoke] SoM stack not installed (--no-shape-of-motion) — skipping cuML/gsplat checks.")
    sys.exit(0)

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

# 3) gsplat rasterizes a trivial scene — exercises the sm_120 CUDA kernels the
# source build exists for (an import-only check would pass a broken build).
try:
    import gsplat
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
    assert rgb.shape == (1, 128, 128, 3)
    print(f"[smoke] gsplat rasterization OK ({getattr(gsplat,'__version__','?')}, output {tuple(rgb.shape)})")
except Exception as e:
    print(f"[smoke][FAIL] gsplat rasterization: {type(e).__name__}: {str(e)[:200]}")
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
  apply_som_patches
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
$( [[ ${INSTALL_SOM} -eq 1 ]] && cat <<'SOMEOF'
- Datasets are NOT downloaded by this script. Get the authors' PROCESSED
  iPhone (DyCheck) and NVIDIA Dynamic Scenes data (Google Drive links in the
  Shape of Motion README), validate with scripts/verify_reference_data.sh,
  then benchmark with scripts/run_reference_benchmark.py.
SOMEOF
)
- third_party/ is gitignored; pins live in this script.
- Dependency lock: $( [[ ${UNLOCKED} -eq 0 && -f "${LOCK_FILE}" ]] && echo "installed from ${LOCK_FILE}" || echo "freshly resolved (no lock used)" ).
  Regenerate locks after deliberate upgrades: scripts/gen_env_locks.sh
- MoSca is the planned second reference tool (--with-mosca, not yet implemented).
EOF

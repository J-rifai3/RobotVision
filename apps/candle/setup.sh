#!/usr/bin/env bash
# Setup Grounded-SAM-2 + FoundationPose for the RobotVision candle/cube apps.
#
# CUDA strategy (important):
#   - ONE PyTorch install (default: cu124 wheels). Driver CUDA 13.0 is fine; PyTorch
#     ships its own CUDA 12.4 runtime libraries.
#   - Grounded-SAM-2: pip install only, SAM2_BUILD_CUDA=0 (no nvcc / no SAM2 CUDA ext).
#   - FoundationPose: prebuilt pytorch3d + nvdiffrast wheels matched to torch+cuda tag.
#     Do NOT compile these against /usr/local/cuda-13.0 while torch is cu124.
#   - FoundationPose mycpp: plain C++ cmake build (no CUDA version coupling).
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${APP_ROOT}/../.." && pwd)"
SCRIPTS="${APP_ROOT}/scripts"
GSAM_DIR="${REPO_ROOT}/external/GroundedSAM2"
FP_DIR="${REPO_ROOT}/external/FoundationPose"
SAM2_CKPT="${GSAM_DIR}/checkpoints/sam2.1_hiera_large.pt"
FP_WEIGHTS="${FP_DIR}/weights"
WORK_DIR="${REPO_ROOT}/data/workspaces/candle"
VENDOR_DIR="${REPO_ROOT}/vendor"

# PyTorch wheel index. cu124 works on drivers that report CUDA 12.x/13.x.
# Override before running if you intentionally use a different torch build:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./setup.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "==> App root:  ${APP_ROOT}"
echo "==> Repo root: ${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "candle-scan"; then
  echo "==> Creating conda env candle-scan"
  conda env create -f "${APP_ROOT}/environment.yml"
fi

conda activate candle-scan

echo "==> [1/6] PyTorch (shared by Grounded-SAM-2 and FoundationPose)"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"
fi
python - <<'PY'
import torch
print(f"    torch {torch.__version__}  |  torch.version.cuda = {torch.version.cuda}")
print(f"    cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "${REPO_ROOT}/external"

if [[ ! -d "${GSAM_DIR}/.git" ]]; then
  echo "==> Cloning Grounded-SAM-2"
  git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git "${GSAM_DIR}"
fi

if [[ ! -d "${FP_DIR}/.git" ]]; then
  echo "==> Cloning FoundationPose"
  git clone --depth 1 https://github.com/NVlabs/FoundationPose.git "${FP_DIR}"
fi

if [[ ! -f "${SAM2_CKPT}" ]]; then
  echo "==> Downloading SAM 2.1 checkpoint"
  mkdir -p "${GSAM_DIR}/checkpoints"
  cd "${GSAM_DIR}/checkpoints"
  if [[ -f download_ckpts.sh ]]; then
    bash download_ckpts.sh
  else
    python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_large.pt",
)
dest = "sam2.1_hiera_large.pt"
if not os.path.exists(dest):
    shutil.copy(path, dest)
print("saved", dest)
PY
  fi
  cd "${APP_ROOT}"
fi

echo "==> [2/6] Grounded-SAM-2 (no CUDA extension build)"
cd "${GSAM_DIR}"
# SAM 2's optional CUDA kernel is separate from FoundationPose and often conflicts
# with the system CUDA toolkit version. Image/video segmentation works without it.
SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 pip install -e ".[notebooks]"

echo "==> [3/6] FoundationPose Python requirements"
python -m pip install -r "${FP_DIR}/requirements.txt"

echo "==> [4/6] FoundationPose GPU extensions (prebuilt wheels, matched to torch)"
bash "${SCRIPTS}/install_fp_gpu_extensions.sh"

echo "==> [5/6] FoundationPose C++ extension (mycpp, no CUDA)"
cd "${FP_DIR}"
bash build_all_conda.sh

if [[ ! -d "${FP_WEIGHTS}/2023-10-28-18-33-37" ]] || [[ ! -d "${FP_WEIGHTS}/2024-01-11-20-02-45" ]]; then
  echo "==> FoundationPose weights still missing."
  echo "    Download from:"
  echo "    https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
  echo "    Place refiner (2023-10-28-18-33-37) and scorer (2024-01-11-20-02-45) under:"
  echo "    ${FP_WEIGHTS}/"
fi

echo "==> [6/6] ZED Python API (Stereolabs SDK — NOT pip package 'pyzed')"
# pip install pyzed installs an unrelated package and breaks `import pyzed.sl`.
pip uninstall -y pyzed 2>/dev/null || true
if [[ -f /usr/local/zed/get_python_api.py ]]; then
  python /usr/local/zed/get_python_api.py
  if ! python -c "import pyzed.sl" 2>/dev/null; then
    echo "    ZED API install finished but import failed. Try:"
    echo "    python -m pip install --ignore-installed ${VENDOR_DIR}/pyzed-*.whl"
  fi
else
  echo "    ZED SDK not found at /usr/local/zed/get_python_api.py"
  echo "    Install from https://www.stereolabs.com/developers/release/"
  echo "    then re-run ./setup.sh"
fi

mkdir -p "${WORK_DIR}"

echo "==> Verifying FoundationPose imports"
python "${FP_DIR}/check_env.py" || true

echo ""
echo "==> Setup complete."
echo "    conda activate candle-scan"
echo "    cd ${APP_ROOT} && python scan_candle.py --help"
echo "    cd ${REPO_ROOT}/apps/cube && python scan_cube.py --help"
echo ""
echo "CUDA note:"
echo "  - Grounded-SAM-2 uses PyTorch only (SAM2_BUILD_CUDA=0)."
echo "  - FoundationPose uses prebuilt pytorch3d/nvdiffrast wheels for $(python "${SCRIPTS}/torch_cuda_tag.py")."
echo "  - Do not point CUDA_HOME at /usr/local/cuda-13.0 while torch is cu124."

#!/usr/bin/env bash
# Install FoundationPose GPU extensions WITHOUT compiling against the system CUDA toolkit.
#
# Why: your driver may report CUDA 13.0 while PyTorch wheels are cu124. Compiling
# pytorch3d/nvdiffrast against /usr/local/cuda-13.0 breaks with a version mismatch.
# Grounded-SAM-2 never needs this step (SAM2_BUILD_CUDA=0).
set -euo pipefail

WHEEL_INDEX="${WHEEL_INDEX:-https://miropsota.github.io/torch_packages_builder}"

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: PyTorch must be installed before FoundationPose GPU extensions." >&2
  exit 1
fi

TAG="$(python "$(dirname "$0")/torch_cuda_tag.py")"
echo "==> FoundationPose GPU extensions for ${TAG}"

pick_wheel() {
  local pkg="$1"
  python - <<PY
import json, subprocess, sys, re
pkg = ${pkg@Q}
tag = ${TAG@Q}
index = ${WHEEL_INDEX@Q}
proc = subprocess.run(
    [sys.executable, "-m", "pip", "index", "versions", pkg, "--extra-index-url", index],
    capture_output=True, text=True,
)
text = proc.stdout + proc.stderr
versions = []
for line in text.splitlines():
    if "Available versions:" in line:
        versions = re.findall(r"\\d+\\.\\d+\\.\\d+\\+[^,\\s]+", line)
        break
if not versions:
    m = re.search(r"\\(([^)]+)\\)", text)
    if m:
        versions = [v.strip() for v in m.group(1).split(",")]
matches = [v for v in versions if tag in v]
if not matches:
    print(f"No prebuilt {pkg} wheel for {tag}", file=sys.stderr)
    print("Available:", ", ".join(versions[:8]), file=sys.stderr)
    sys.exit(1)
print(matches[0])
PY
}

install_pkg() {
  local pkg="$1"
  local wheel
  wheel="$(pick_wheel "$pkg")"
  echo "    ${pkg} -> ${wheel}"
  python -m pip install --extra-index-url "${WHEEL_INDEX}" "${pkg}==${wheel}"
}

install_pkg pytorch3d
install_pkg nvdiffrast

python - <<'PY'
import pytorch3d
import nvdiffrast.torch as dr
print("pytorch3d:", pytorch3d.__version__)
print("nvdiffrast: ok")
PY

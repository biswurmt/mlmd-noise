#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_env.sh [--with-medmnist] [--cpu]
# Creates .venv, upgrades pip, installs a CUDA-matched torch build, then installs the package.
WITH_MEDMNIST=0
TORCH_CHANNEL=${TORCH_CHANNEL:-cu126}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.21.0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-medmnist)
      WITH_MEDMNIST=1
      shift
      ;;
    --cpu)
      TORCH_CHANNEL=cpu
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./scripts/setup_env.sh [--with-medmnist] [--cpu]"
      exit 1
      ;;
  esac
done

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CHANNEL}}

echo "Creating virtualenv in ${VENV_DIR} using ${PYTHON}..."
${PYTHON} -m venv "${VENV_DIR}"

echo "Activating virtualenv and upgrading pip..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "Installing torch ${TORCH_VERSION} from ${PYTORCH_INDEX_URL}..."
pip install --index-url "${PYTORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

if [[ ${WITH_MEDMNIST} -eq 1 ]]; then
  echo "Installing torchvision ${TORCHVISION_VERSION} from ${PYTORCH_INDEX_URL}..."
  pip install --index-url "${PYTORCH_INDEX_URL}" "torchvision==${TORCHVISION_VERSION}"
  echo "Installing editable package with [dev,medmnist] extras..."
  pip install -e '.[dev,medmnist]'
else
  echo "Installing editable package with [dev] extras..."
  pip install -e '.[dev]'
  echo "To also install MedMNIST extras, run: ./scripts/setup_env.sh --with-medmnist"
fi

echo "Done. To activate the environment: source ${VENV_DIR}/bin/activate"

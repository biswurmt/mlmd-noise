#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_env.sh [--with-medmnist]
# Creates .venv, upgrades pip, and installs editable package with extras.
WITH_MEDMNIST=0
if [[ ${1:-} == "--with-medmnist" ]]; then
  WITH_MEDMNIST=1
fi

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "Creating virtualenv in ${VENV_DIR} using ${PYTHON}..."
${PYTHON} -m venv "${VENV_DIR}"

echo "Activating virtualenv and upgrading pip..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

if [[ ${WITH_MEDMNIST} -eq 1 ]]; then
  echo "Installing editable package with [dev,medmnist] extras (this will pull large packages like torch)..."
  pip install -e '.[dev,medmnist]'
else
  echo "Installing editable package with [dev] extras..."
  pip install -e '.[dev]'
  echo "To also install MedMNIST (and torch), run: ./scripts/setup_env.sh --with-medmnist"
fi

echo "Done. To activate the environment: source ${VENV_DIR}/bin/activate"

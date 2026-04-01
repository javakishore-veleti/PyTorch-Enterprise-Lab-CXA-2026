#!/usr/bin/env bash
# QuantEdge — Python environment setup
# Creates conda env at: ~/runtime_data/python_venvs/PyTorchMastery/PyTorch-Enterprise-Lab-CXA-2026

set -euo pipefail

REPO_NAME="PyTorch-Enterprise-Lab-CXA-2026"
VENV_BASE="${HOME}/runtime_data/python_venvs/PyTorchMastery"
VENV_PATH="${VENV_BASE}/${REPO_NAME}"
PYTHON_VERSION="3.11"

echo "==> QuantEdge: Setting up Python environment"
echo "    Target path : ${VENV_PATH}"
echo "    Python      : ${PYTHON_VERSION}"

mkdir -p "${VENV_BASE}"

if conda env list | grep -q "${VENV_PATH}"; then
  echo "==> Conda env already exists at ${VENV_PATH} — skipping create"
else
  echo "==> Creating conda env..."
  conda create --prefix "${VENV_PATH}" python="${PYTHON_VERSION}" -y
  echo "==> Conda env created"
fi

echo "==> Installing Python dependencies with uv..."
conda run --prefix "${VENV_PATH}" pip install uv
conda run --prefix "${VENV_PATH}" uv pip install -e ".[dev,gpu,mlops]"

echo ""
echo "✅ Setup complete. To activate:"
echo "   conda activate ${VENV_PATH}"
echo ""

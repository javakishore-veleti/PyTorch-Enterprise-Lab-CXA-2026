#!/usr/bin/env bash
# QuantEdge — Run test suite

set -euo pipefail

VENV_PATH="${HOME}/runtime_data/python_venvs/PyTorchMastery/PyTorch-Enterprise-Lab-CXA-2026"

echo "==> Running QuantEdge test suite..."
conda run --prefix "${VENV_PATH}" \
  pytest tests/ -v --tb=short -m "not gpu and not integration and not slow" "$@"

# Copilot Instructions

## Project Overview

**PyTorch-Enterprise-Lab-CXA-2026** is a hands-on PyTorch lab for Chief Architect-level practitioners, progressing from tensor fundamentals through GPU mastery, transformer design, LLM fine-tuning, and enterprise-grade model serving.

This repository is Python/PyTorch-first. The `.sfdx/` directory is a VS Code Salesforce extension artifact — it is not part of this project and should be ignored.

## Architecture Intent

The lab is structured as a progression:
1. **Tensor & GPU Mastery** — low-level PyTorch primitives, CUDA device management, memory optimization
2. **Transformer Design** — building attention mechanisms and transformer architectures from scratch
3. **LLM Fine-tuning** — parameter-efficient fine-tuning (LoRA, QLoRA, PEFT) on enterprise data
4. **Enterprise Model Serving** — TorchServe, ONNX export, quantization, and production deployment patterns

- **Foundations (Weeks 1–2):** EUR/USD Forex Tick Data 2010–2024 (Kaggle, ~8 GB, 300M+ ticks) for tensor/autograd work; CFPB Consumer Financial Complaints (HuggingFace `cfpb/consumer-finance-complaints`, ~2 GB, 3M+ records) for DataLoader/training loop work

Refer to `PyTorch_Mastery_Plan.xlsx` for the detailed module breakdown and lab sequencing.

## Environment & Tooling

- **Language:** Python 3.10+
- **Core framework:** PyTorch (GPU-accelerated; CUDA expected for GPU labs)
- **Notebooks:** Jupyter / Marimo (`.ipynb_checkpoints/` and `__marimo__/` are gitignored)
- **Package management:** Any of `pip`, `uv`, `poetry`, `pdm`, or `pipenv` — lockfiles are project-specific
- **Linting/formatting:** Ruff (`.ruff_cache/` is gitignored)
- **Type checking:** mypy or pytype

## Conventions to Follow

- Lab modules should be runnable as both standalone Python scripts and Jupyter notebooks where applicable.
- GPU-dependent code must include CPU fallback paths using `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Prefer `torch.compile()` over manual kernel optimization unless CUDA-level control is explicitly required.
- Model checkpoints, large datasets, and generated artifacts should never be committed — add them to `.gitignore`.
- Use type annotations throughout; the project targets mypy-clean code.

## Testing

- Test runner: `pytest`
- Run all tests: `pytest`
- Run a single test file: `pytest path/to/test_file.py`
- Run a single test: `pytest path/to/test_file.py::test_function_name`
- GPU tests should be marked with a custom marker (e.g., `@pytest.mark.gpu`) and skipped when no CUDA device is available.

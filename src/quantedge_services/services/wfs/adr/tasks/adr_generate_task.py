"""ADRGeneratorTask — generates 6 built-in Architecture Decision Records as markdown files."""
from __future__ import annotations
from pathlib import Path
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    ADRGenerateRequest, ADRGenerateResult,
)

_BUILT_IN_ADRS: dict[str, dict] = {
    "ADR-001": {
        "filename": "ADR-001-async-job-pattern.md",
        "title": "Async 202 Job Pattern over Synchronous REST",
        "context": (
            "The QuantEdge platform runs computationally expensive ML workflows "
            "(training, export, quantization) that can take minutes to hours. "
            "Synchronous HTTP responses would time out and leave clients with no feedback."
        ),
        "decision": (
            "All POST endpoints return HTTP 202 Accepted with a `job_id`. "
            "Clients poll `GET /jobs/{job_id}` for status and results. "
            "Background tasks are dispatched via FastAPI `BackgroundTasks`."
        ),
        "positive": [
            "No HTTP timeout issues for long-running ML jobs",
            "Clients can poll at their own cadence or use webhooks",
            "Uniform response contract across all workflows",
        ],
        "negative": [
            "Clients must implement polling logic",
            "In-memory job store is lost on restart (acceptable for lab)",
        ],
    },
    "ADR-002": {
        "filename": "ADR-002-lora-vs-full-finetune.md",
        "title": "LoRA Adapter Fine-Tuning vs Full Model Fine-Tuning",
        "context": (
            "Fine-tuning large language models (7B+ parameters) on domain-specific "
            "data requires enormous GPU memory and compute. "
            "Full fine-tuning updates all parameters, which is prohibitive on a single GPU."
        ),
        "decision": (
            "Use LoRA (Low-Rank Adaptation) via the PEFT library. "
            "Trainable parameters are injected as low-rank matrices into attention layers. "
            "Base model weights are frozen; only adapter weights are updated."
        ),
        "positive": [
            "Reduces trainable parameters by ~99% (e.g., 7B → ~4M params)",
            "Fits on a single 24 GB GPU with 4-bit quantization",
            "Adapters can be merged back into base model for deployment",
        ],
        "negative": [
            "Slight accuracy gap vs full fine-tuning on large datasets",
            "Adapter selection (rank, alpha) requires hyperparameter search",
        ],
    },
    "ADR-003": {
        "filename": "ADR-003-onnx-vs-torchscript.md",
        "title": "ONNX Opset 17 vs TorchScript Trace for Model Export",
        "context": (
            "Production inference requires exporting trained PyTorch models to a "
            "format compatible with optimized runtimes (TensorRT, ONNX Runtime). "
            "Two primary options exist: TorchScript trace/script and ONNX export."
        ),
        "decision": (
            "Primary export target is ONNX opset 17 using `torch.onnx.export`. "
            "TorchScript is retained as a secondary format for PyTorch-native serving. "
            "ONNX Runtime with CUDA EP is used for GPU inference."
        ),
        "positive": [
            "ONNX is runtime-agnostic (TensorRT, OpenVINO, ONNX Runtime)",
            "Opset 17 supports modern operators including scaled dot-product attention",
            "Interoperability with non-PyTorch ecosystems",
        ],
        "negative": [
            "Dynamic control flow requires ONNX script (more complex than trace)",
            "Custom ops may not have ONNX equivalents",
        ],
    },
    "ADR-004": {
        "filename": "ADR-004-fastapi-vs-flask.md",
        "title": "FastAPI Chosen over Flask for REST API Layer",
        "context": (
            "The QuantEdge platform needs a production-grade Python REST API framework "
            "that supports async I/O, automatic schema validation, and OpenAPI docs. "
            "Flask and FastAPI are the two most widely used Python web frameworks."
        ),
        "decision": (
            "FastAPI is chosen as the sole REST framework. "
            "Pydantic v2 DTOs provide request/response validation. "
            "Native `async def` route handlers support non-blocking background tasks."
        ),
        "positive": [
            "Built-in OpenAPI/Swagger UI at /docs with zero config",
            "Pydantic integration eliminates manual validation boilerplate",
            "Native async support enables concurrent request handling",
        ],
        "negative": [
            "Smaller plugin ecosystem compared to Flask",
            "Pydantic v2 migration breaking changes require careful version pinning",
        ],
    },
    "ADR-005": {
        "filename": "ADR-005-quantization-strategy.md",
        "title": "Dynamic INT8 for Inference, QAT for Production",
        "context": (
            "Deploying large PyTorch models requires reducing memory footprint and "
            "increasing inference throughput. Quantization converts FP32 weights to "
            "lower-precision formats (INT8, INT4) with acceptable accuracy tradeoffs."
        ),
        "decision": (
            "Dynamic INT8 quantization (`torch.quantization.quantize_dynamic`) is used "
            "for fast experimentation and CPU inference. "
            "Quantization-Aware Training (QAT) is reserved for production deployments "
            "where accuracy must be preserved."
        ),
        "positive": [
            "Dynamic INT8 requires no calibration dataset — fast to apply",
            "QAT achieves near-FP32 accuracy with INT8 runtime performance",
            "2-4× memory reduction enables larger batch sizes",
        ],
        "negative": [
            "QAT requires re-training which increases pipeline complexity",
            "Not all PyTorch ops support INT8 (fallback to FP32)",
        ],
    },
    "ADR-006": {
        "filename": "ADR-006-canary-deployment.md",
        "title": "Canary Traffic Splitting over Blue-Green Deployment",
        "context": (
            "Deploying new model versions to production carries risk. "
            "Blue-green deployment switches all traffic instantly, while canary "
            "deployment gradually routes a percentage of traffic to the new version."
        ),
        "decision": (
            "Canary deployment with configurable traffic split percentage (default 10%) "
            "is used for all model version promotions. "
            "Metrics are collected during the canary window before full promotion."
        ),
        "positive": [
            "Limits blast radius of a bad model version to canary traffic fraction",
            "Enables A/B metric comparison between model versions in production",
            "Automatic rollback if error rate exceeds threshold",
        ],
        "negative": [
            "More complex routing logic than blue-green",
            "Both model versions must be hosted simultaneously during canary window",
        ],
    },
}


class ADRGeneratorTask:
    """Generates built-in Architecture Decision Records as markdown files."""

    def execute(self, request: ADRGenerateRequest) -> ADRGenerateResult:
        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        targets = _BUILT_IN_ADRS
        if request.adr_ids:
            targets = {k: v for k, v in _BUILT_IN_ADRS.items() if k in request.adr_ids}

        generated: list[str] = []
        for adr_id, meta in targets.items():
            path = out_dir / meta["filename"]
            path.write_text(self._render(adr_id, meta))
            generated.append(str(path))

        return ADRGenerateResult(
            generated_files=generated,
            adr_count=len(generated),
            output_dir=str(out_dir),
            status="completed",
        )

    def _render(self, adr_id: str, meta: dict) -> str:
        positives = "\n".join(f"- {b}" for b in meta["positive"])
        negatives = "\n".join(f"- {t}" for t in meta["negative"])
        return f"""# {adr_id}: {meta["title"]}

**Status:** Accepted  
**Date:** 2026-04-01  
**Deciders:** QuantEdge Architecture Team

## Context
{meta["context"]}

## Decision
{meta["decision"]}

## Consequences
### Positive
{positives}

### Negative
{negatives}
"""

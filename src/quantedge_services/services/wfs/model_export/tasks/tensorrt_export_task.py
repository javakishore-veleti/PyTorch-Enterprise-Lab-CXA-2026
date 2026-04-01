"""TensorRT export task — guarded import, requires GPU + torch_tensorrt."""
from __future__ import annotations
import os
from quantedge_services.api.schemas.foundations.export_schemas import (
    TensorRTExportRequest,
    TensorRTExportResult,
)


class TensorRTExportTask:
    """Compiles a TorchScript model to TensorRT engine (requires torch_tensorrt + CUDA)."""

    def execute(self, request: TensorRTExportRequest) -> TensorRTExportResult:
        try:
            import torch_tensorrt  # noqa: F401
        except ImportError:
            return TensorRTExportResult(
                output_path="",
                precision=request.precision,
                status="torch_tensorrt_not_installed",
            )

        import torch

        if not torch.cuda.is_available():
            return TensorRTExportResult(
                output_path="",
                precision=request.precision,
                status="cuda_not_available",
            )

        os.makedirs(request.output_dir, exist_ok=True)

        scripted_model = torch.jit.load(request.torchscript_path)
        scripted_model = scripted_model.cuda()

        inputs = [
            torch_tensorrt.Input(
                shape=[request.batch_size, request.seq_len, request.input_size]
            )
        ]

        enabled_precisions = {torch.float16} if request.precision == "fp16" else {torch.float32}

        trt_model = torch_tensorrt.compile(
            scripted_model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
        )

        output_path = os.path.join(request.output_dir, f"forex_transformer_trt_{request.precision}.ts")
        torch.jit.save(trt_model, output_path)

        return TensorRTExportResult(
            output_path=output_path,
            precision=request.precision,
            status="success",
        )

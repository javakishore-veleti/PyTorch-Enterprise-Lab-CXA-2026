"""ONNX export task — wraps ForexTransformer to strip attn_weights list."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
from quantedge_services.api.schemas.foundations.export_schemas import (
    ONNXExportRequest,
    ONNXExportResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class ONNXExportTask:
    """Exports ForexTransformer to ONNX format using ONNXWrapper to strip list output."""

    def execute(self, request: ONNXExportRequest) -> ONNXExportResult:
        os.makedirs(request.output_dir, exist_ok=True)

        class ONNXWrapper(nn.Module):
            """Wraps ForexTransformer — returns only the scalar output tensor for ONNX export."""

            def __init__(self, inner: ForexTransformer) -> None:
                super().__init__()
                self._inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output, _ = self._inner(x)
                return output

        base_model = ForexTransformer(
            input_size=request.input_size,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_layers,
            dim_feedforward=request.d_model * 4,
        )
        base_model.eval()
        wrapper = ONNXWrapper(base_model)
        wrapper.eval()

        example_input = torch.randn(4, request.seq_len, request.input_size)

        output_path = os.path.join(request.output_dir, "forex_transformer.onnx")

        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if request.dynamic_batch else None

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (example_input,),
                output_path,
                opset_version=request.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

        return ONNXExportResult(
            output_path=output_path,
            opset_version=request.opset_version,
            dynamic_batch=request.dynamic_batch,
            status="success",
        )

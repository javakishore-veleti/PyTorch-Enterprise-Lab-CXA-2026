"""ONNX validation task — compares ONNX Runtime output against PyTorch eager output."""
from __future__ import annotations
import numpy as np
import torch
from quantedge_services.api.schemas.foundations.export_schemas import (
    ONNXValidateRequest,
    ONNXValidateResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class ONNXValidateTask:
    """Validates ONNX model output parity with PyTorch eager inference."""

    def execute(self, request: ONNXValidateRequest) -> ONNXValidateResult:
        try:
            import onnxruntime as ort
        except ImportError:
            return ONNXValidateResult(
                max_abs_diff=-1.0,
                passed=False,
                status="onnxruntime_not_installed",
            )

        input_np = np.random.randn(
            request.batch_size, request.seq_len, request.input_size
        ).astype(np.float32)

        sess = ort.InferenceSession(request.onnx_path)
        onnx_outputs = sess.run(None, {"input": input_np})
        onnx_out = onnx_outputs[0]

        input_tensor = torch.from_numpy(input_np)
        model = ForexTransformer(
            input_size=request.input_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=4,
            dim_feedforward=256,
        )
        model.eval()
        with torch.no_grad():
            torch_out, _ = model(input_tensor)

        max_abs_diff = float(np.abs(torch_out.numpy() - onnx_out).max())
        passed = max_abs_diff < 1e-3

        return ONNXValidateResult(
            max_abs_diff=max_abs_diff,
            passed=passed,
            status="success",
        )

import os
import torch
import torch.nn as nn

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    QuantizeDynamicRequest,
    QuantizeDynamicResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import (
    ForexTransformer,
)


class DynamicQuantTask:
    def execute(self, request: QuantizeDynamicRequest) -> QuantizeDynamicResult:
        os.makedirs(request.output_dir, exist_ok=True)
        torch.backends.quantized.engine = "qnnpack"

        model = ForexTransformer(
            input_size=request.input_size,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_layers,
            dim_feedforward=request.d_model * 4,
            dropout=0.0,
        )
        model.eval()

        original_path = os.path.join(request.output_dir, "fp32_dynamic_model.pt")
        torch.save(model.state_dict(), original_path)
        original_size_mb = os.path.getsize(original_path) / (1024 * 1024)

        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        output_path = os.path.join(request.output_dir, "quantized_dynamic.pt")
        torch.save(quantized_model, output_path)
        quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        size_reduction_pct = (
            (original_size_mb - quantized_size_mb) / original_size_mb * 100
            if original_size_mb > 0
            else 0.0
        )

        return QuantizeDynamicResult(
            output_path=output_path,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            size_reduction_pct=size_reduction_pct,
            status="success",
        )

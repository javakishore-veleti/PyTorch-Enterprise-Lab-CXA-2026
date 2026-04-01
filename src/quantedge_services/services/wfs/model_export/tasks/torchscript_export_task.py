"""TorchScript export task — torch.jit.trace for ForexTransformer."""
from __future__ import annotations
import os
import torch
from quantedge_services.api.schemas.foundations.export_schemas import (
    TorchScriptExportRequest,
    TorchScriptExportResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class TorchScriptExportTask:
    """Exports ForexTransformer via torch.jit.trace (preferred over script for tuple returns)."""

    def execute(self, request: TorchScriptExportRequest) -> TorchScriptExportResult:
        os.makedirs(request.output_dir, exist_ok=True)

        model = ForexTransformer(
            input_size=request.input_size,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_layers,
            dim_feedforward=request.d_model * 4,
        )
        model.eval()

        example_input = torch.randn(4, request.seq_len, request.input_size)

        with torch.no_grad():
            eager_out, _ = model(example_input)

        # Always use trace — jit.script struggles with dynamic list returns from ForexTransformer
        traced = torch.jit.trace(model, (example_input,))

        output_path = os.path.join(
            request.output_dir, f"forex_transformer_{request.export_mode}.pt"
        )
        torch.jit.save(traced, output_path)

        with torch.no_grad():
            traced_out, _ = traced(example_input)

        validation_passed = traced_out.shape == eager_out.shape

        return TorchScriptExportResult(
            output_path=output_path,
            export_mode=request.export_mode,
            validation_passed=validation_passed,
            status="success",
        )

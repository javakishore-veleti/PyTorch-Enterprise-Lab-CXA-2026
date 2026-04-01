import time
import torch

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    ModelInferRequest,
    ModelInferResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import (
    ForexTransformer,
)


class ModelInferTask:
    def execute(self, request: ModelInferRequest) -> ModelInferResult:
        torch.backends.quantized.engine = "qnnpack"

        t0 = time.perf_counter()

        with torch.no_grad():
            if request.model_format == "torchscript":
                model = torch.jit.load(request.model_path)
                model.eval()
                inp = torch.randn(1, request.seq_len, request.input_size)
                output = model(inp)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output_value = float(output.flatten()[0])

            elif request.model_format == "quantized_dynamic":
                model = torch.load(request.model_path, weights_only=False)
                model.eval()
                flat_input = torch.randn(1, request.input_size * request.seq_len)
                output = model(flat_input)
                output_value = float(output.flatten()[0])

            else:  # eager
                model = ForexTransformer(
                    input_size=request.input_size,
                    d_model=request.d_model,
                    nhead=request.nhead,
                    num_encoder_layers=request.num_layers,
                    dim_feedforward=request.d_model * 4,
                    dropout=0.0,
                )
                model.eval()
                inp = torch.randn(1, request.seq_len, request.input_size)
                output, _ = model(inp)
                output_value = float(output.flatten()[0])

        latency_ms = (time.perf_counter() - t0) * 1000

        return ModelInferResult(
            output_value=output_value,
            latency_ms=latency_ms,
            model_format=request.model_format,
            status="success",
        )

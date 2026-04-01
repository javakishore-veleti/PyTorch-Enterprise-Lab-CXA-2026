import time
import torch
import numpy as np

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    ServingBenchmarkRequest,
    ServingBenchmarkResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import (
    ForexTransformer,
)


class ServingBenchmarkTask:
    def execute(self, request: ServingBenchmarkRequest) -> ServingBenchmarkResult:
        torch.backends.quantized.engine = "qnnpack"

        if request.model_format == "torchscript":
            model = torch.jit.load(request.model_path)
            model.eval()
        elif request.model_format == "quantized_dynamic":
            model = torch.load(request.model_path, weights_only=False)
            model.eval()
        else:
            model = ForexTransformer(
                input_size=request.input_size,
                d_model=request.d_model,
                nhead=request.nhead,
                num_encoder_layers=request.num_layers,
                dim_feedforward=request.d_model * 4,
                dropout=0.0,
            )
            model.eval()

        latencies = []
        total_start = time.perf_counter()

        with torch.no_grad():
            for _ in range(request.num_requests):
                if request.model_format == "quantized_dynamic":
                    inp = torch.randn(request.batch_size, request.input_size * request.seq_len)
                else:
                    inp = torch.randn(request.batch_size, request.seq_len, request.input_size)
                t0 = time.perf_counter()
                model(inp)
                latencies.append((time.perf_counter() - t0) * 1000)

        total_elapsed = time.perf_counter() - total_start
        latencies_arr = np.array(latencies)

        return ServingBenchmarkResult(
            p50_latency_ms=float(np.percentile(latencies_arr, 50)),
            p95_latency_ms=float(np.percentile(latencies_arr, 95)),
            p99_latency_ms=float(np.percentile(latencies_arr, 99)),
            mean_latency_ms=float(np.mean(latencies_arr)),
            throughput_qps=float(request.num_requests / total_elapsed),
            status="success",
        )

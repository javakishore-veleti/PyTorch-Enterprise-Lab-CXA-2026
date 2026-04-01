import os
import time
import torch
import torch.nn as nn

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    QuantizeCompareRequest,
    QuantizeCompareResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import (
    ForexTransformer,
)


class QuantCompareTask:
    def execute(self, request: QuantizeCompareRequest) -> QuantizeCompareResult:
        os.makedirs(request.output_dir, exist_ok=True)
        torch.backends.quantized.engine = "qnnpack"

        fp32_model = ForexTransformer(
            input_size=request.input_size,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_layers,
            dim_feedforward=request.d_model * 4,
            dropout=0.0,
        )
        fp32_model.eval()

        quantized_model = torch.quantization.quantize_dynamic(
            fp32_model, {nn.Linear}, dtype=torch.qint8
        )

        fp32_path = os.path.join(request.output_dir, "compare_fp32.pt")
        dyn_path = os.path.join(request.output_dir, "compare_dynamic.pt")
        torch.save(fp32_model.state_dict(), fp32_path)
        torch.save(quantized_model, dyn_path)
        fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
        dynamic_size_mb = os.path.getsize(dyn_path) / (1024 * 1024)

        dummy = torch.randn(request.batch_size, request.seq_len, request.input_size)

        fp32_times = []
        with torch.no_grad():
            for _ in range(request.num_runs):
                t0 = time.perf_counter()
                fp32_model(dummy)
                fp32_times.append((time.perf_counter() - t0) * 1000)

        dyn_times = []
        with torch.no_grad():
            for _ in range(request.num_runs):
                t0 = time.perf_counter()
                quantized_model(dummy)
                dyn_times.append((time.perf_counter() - t0) * 1000)

        fp32_latency_ms = sum(fp32_times) / len(fp32_times)
        dynamic_latency_ms = sum(dyn_times) / len(dyn_times)
        speedup_ratio = fp32_latency_ms / dynamic_latency_ms if dynamic_latency_ms > 0 else 1.0

        return QuantizeCompareResult(
            fp32_latency_ms=fp32_latency_ms,
            dynamic_latency_ms=dynamic_latency_ms,
            dynamic_size_mb=dynamic_size_mb,
            fp32_size_mb=fp32_size_mb,
            speedup_ratio=speedup_ratio,
            status="success",
        )

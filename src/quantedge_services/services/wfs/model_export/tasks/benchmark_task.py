"""Benchmark task — wall-clock latency + throughput QPS across backends."""
from __future__ import annotations
import os
import time
import torch
from quantedge_services.api.schemas.foundations.export_schemas import (
    BenchmarkRequest,
    BenchmarkResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class BenchmarkTask:
    """Measures inference latency (ms) and throughput (QPS) for eager, TorchScript, and ONNX Runtime."""

    def execute(self, request: BenchmarkRequest) -> BenchmarkResult:
        example_input = torch.randn(request.batch_size, request.seq_len, request.input_size)

        # Eager
        model = ForexTransformer(
            input_size=request.input_size,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_layers,
            dim_feedforward=request.d_model * 4,
        )
        model.eval()
        with torch.no_grad():
            model(example_input)  # warmup
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(request.num_runs):
                model(example_input)
        eager_latency_ms = (time.perf_counter() - start) / request.num_runs * 1000

        # TorchScript
        if request.torchscript_path and os.path.exists(request.torchscript_path):
            scripted = torch.jit.load(request.torchscript_path)
            scripted.eval()
            with torch.no_grad():
                scripted(example_input)  # warmup
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(request.num_runs):
                    scripted(example_input)
            ts_latency_ms = (time.perf_counter() - start) / request.num_runs * 1000
        else:
            ts_latency_ms = 0.0

        # ONNX Runtime
        try:
            import onnxruntime as ort
            import numpy as np
            ort_available = True
        except ImportError:
            ort_available = False

        if ort_available and request.onnx_path and os.path.exists(request.onnx_path):
            import numpy as np
            input_np = example_input.numpy()
            sess = ort.InferenceSession(request.onnx_path)
            sess.run(None, {"input": input_np})  # warmup
            start = time.perf_counter()
            for _ in range(request.num_runs):
                sess.run(None, {"input": input_np})
            ort_latency_ms = (time.perf_counter() - start) / request.num_runs * 1000
        else:
            ort_latency_ms = 0.0

        def _throughput(latency_ms: float) -> float:
            if latency_ms <= 0.0:
                return 0.0
            return (1000.0 / latency_ms) * request.batch_size

        status_parts = ["success"]
        if ts_latency_ms == 0.0:
            status_parts.append("torchscript_skipped")
        if ort_latency_ms == 0.0:
            status_parts.append("onnxruntime_skipped")

        return BenchmarkResult(
            eager_latency_ms=eager_latency_ms,
            torchscript_latency_ms=ts_latency_ms,
            onnxruntime_latency_ms=ort_latency_ms,
            eager_throughput_qps=_throughput(eager_latency_ms),
            torchscript_throughput_qps=_throughput(ts_latency_ms),
            onnxruntime_throughput_qps=_throughput(ort_latency_ms),
            status="|".join(status_parts),
        )

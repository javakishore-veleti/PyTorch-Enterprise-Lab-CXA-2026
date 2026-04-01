from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    MemorySummaryRequest, MemorySummaryResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP
from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM


class MemorySummaryTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: MemorySummaryRequest) -> MemorySummaryResponse:
        try:
            df = pd.read_parquet(request.parquet_path)
            feature_cols = [c for c in df.columns if c != "target"]
            n_features = len(feature_cols)

            use_cuda = torch.cuda.is_available()
            device_str = "cuda" if use_cuda else "cpu"
            device = torch.device(device_str)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            if request.model_type == "lstm":
                model = ForexLSTM(input_size=n_features).to(device)
            else:
                model = ForexMLP(input_size=n_features, hidden_sizes=[128, 64]).to(device)
            model.eval()

            batch_size = min(request.batch_size, len(df))
            X = torch.tensor(df[feature_cols].values[:batch_size], dtype=torch.float32).to(device)

            if request.model_type == "lstm":
                seq_len = 10
                n_seqs = batch_size // seq_len
                if n_seqs == 0:
                    n_seqs = 1
                    if X.shape[0] >= seq_len:
                        X = X[:seq_len]
                    else:
                        repeat_n = seq_len // X.shape[0] + 1
                        X = X.repeat(repeat_n, 1)[:seq_len]
                X = X[: n_seqs * seq_len].view(n_seqs, seq_len, n_features)

            with torch.no_grad():
                _ = model(X)

            allocated_mb = torch.cuda.memory_allocated(device) / 1e6 if use_cuda else 0.0
            reserved_mb = torch.cuda.memory_reserved(device) / 1e6 if use_cuda else 0.0
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1e6 if use_cuda else 0.0
            summary_text = torch.cuda.memory_summary(device) if use_cuda else "CPU-only: no CUDA memory summary"

            return MemorySummaryResponse(
                execution_id=request.execution_id,
                device=device_str,
                allocated_mb=allocated_mb,
                reserved_mb=reserved_mb,
                peak_allocated_mb=peak_allocated_mb,
                summary_text=summary_text,
                status="success",
            )
        except Exception as exc:
            self._logger.error("memory_summary_failed", execution_id=request.execution_id, error=str(exc))
            return MemorySummaryResponse(
                execution_id=request.execution_id,
                device="unknown",
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                summary_text="",
                status="failed",
                error=str(exc),
            )

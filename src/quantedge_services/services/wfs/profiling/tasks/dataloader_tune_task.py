from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    DataloaderTuneRequest, DataloaderTuneResponse,
)
from quantedge_services.core.logging import StructuredLogger


class DataloaderTuneTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: DataloaderTuneRequest) -> DataloaderTuneResponse:
        try:
            df = pd.read_parquet(request.parquet_path)
            feature_cols = [c for c in df.columns if c != "target"]
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            y_col = "target" if "target" in df.columns else feature_cols[-1]
            y = torch.tensor(df[y_col].values, dtype=torch.float32)
            dataset = TensorDataset(X, y)

            timing_results: dict[str, float] = {}
            for nw in request.num_workers_sweep:
                try:
                    loader = DataLoader(
                        dataset,
                        batch_size=request.batch_size,
                        num_workers=nw,
                        pin_memory=request.pin_memory,
                    )
                    t0 = time.perf_counter()
                    for _batch in loader:
                        pass
                    t1 = time.perf_counter()
                    timing_results[str(nw)] = t1 - t0
                except Exception as nw_exc:
                    self._logger.info(
                        "dataloader_tune_worker_failed",
                        execution_id=request.execution_id,
                        num_workers=nw,
                        error=str(nw_exc),
                    )
                    timing_results[str(nw)] = float("inf")

            # Find best (lowest time), ignoring inf
            finite = {k: v for k, v in timing_results.items() if v != float("inf")}
            if finite:
                best_key = min(finite, key=lambda k: finite[k])
            else:
                best_key = str(request.num_workers_sweep[0])

            best_num_workers = int(best_key)
            base_time = timing_results.get("0", 1.0)
            best_time = timing_results[best_key]
            # Compute speedup from raw timing dict (before serialization)
            if base_time == float("inf") or base_time <= 0:
                speedup_vs_single = 1.0
            else:
                speedup_vs_single = base_time / best_time if best_time > 0 else 1.0

            # Replace inf with -1.0 sentinel for serialization
            safe_timing = {k: (v if v != float("inf") else -1.0) for k, v in timing_results.items()}

            return DataloaderTuneResponse(
                execution_id=request.execution_id,
                best_num_workers=best_num_workers,
                timing_results=safe_timing,
                speedup_vs_single=speedup_vs_single,
                status="success",
            )
        except Exception as exc:
            self._logger.error("dataloader_tune_failed", execution_id=request.execution_id, error=str(exc))
            return DataloaderTuneResponse(
                execution_id=request.execution_id,
                best_num_workers=0,
                timing_results={},
                speedup_vs_single=1.0,
                status="failed",
                error=str(exc),
            )

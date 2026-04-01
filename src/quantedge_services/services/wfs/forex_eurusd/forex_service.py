"""ForexEURUSDService — orchestrates all Forex EUR/USD workflow tasks."""

from __future__ import annotations

import pandas as pd

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest,
    ForexAutogradResponse,
    ForexIngestionRequest,
    ForexIngestionResponse,
    ForexPreprocessRequest,
    ForexPreprocessResponse,
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_eurusd.tasks.autograd_task import ForexAutogradTask
from quantedge_services.services.wfs.forex_eurusd.tasks.ingest_task import ForexIngestionTask
from quantedge_services.services.wfs.forex_eurusd.tasks.preprocess_task import ForexPreprocessTask
from quantedge_services.services.wfs.forex_eurusd.tasks.tensor_ops_task import ForexTensorOpsTask


class ForexEURUSDService:
    """Domain service for EUR/USD Forex workflows.

    Owns task orchestration and the in-memory session DataFrame.
    Called exclusively by FoundationsServiceFacade.
    """

    def __init__(
        self,
        ingest_task: ForexIngestionTask,
        preprocess_task: ForexPreprocessTask,
        autograd_task: ForexAutogradTask,
        tensor_ops_task: ForexTensorOpsTask,
    ) -> None:
        self._ingest_task = ingest_task
        self._preprocess_task = preprocess_task
        self._autograd_task = autograd_task
        self._tensor_ops_task = tensor_ops_task
        self._logger = StructuredLogger(name=__name__)
        self._session_df: pd.DataFrame | None = None

    def ingest(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        import pandas as pd
        from pathlib import Path
        COLS = ["datetime", "open", "high", "low", "close", "volume"]
        resp = self._ingest_task.execute(request)
        if resp.status == "success":
            # Re-load into memory for subsequent tasks
            frames = []
            for f in sorted(Path(request.data_dir).glob("*.csv")):
                if request.years and not any(str(y) in f.stem for y in request.years):
                    continue
                frames.append(pd.read_csv(f, header=None, names=COLS, parse_dates=["datetime"]))
            df = pd.concat(frames, ignore_index=True).sort_values("datetime").set_index("datetime")
            self._session_df = df
        return resp

    def preprocess(self, request: ForexPreprocessRequest) -> ForexPreprocessResponse:
        if self._session_df is None:
            return ForexPreprocessResponse(
                execution_id=request.execution_id, rows_after=0,
                nan_filled=0, status="failed", error="No data loaded. Call ingest first.",
            )
        resp, tensor = self._preprocess_task.execute(request, self._session_df)
        if resp.status == "success":
            self._session_tensor = tensor
        return resp

    def run_autograd(self, request: ForexAutogradRequest) -> ForexAutogradResponse:
        if not hasattr(self, "_session_tensor"):
            return ForexAutogradResponse(
                execution_id=request.execution_id,
                manual_loss=0.0, autograd_loss=0.0, max_grad_diff=0.0,
                status="failed", error="No tensor. Call preprocess first.",
            )
        return self._autograd_task.execute(request, self._session_tensor)

    def run_tensor_ops(self, request: ForexTensorOpsRequest) -> ForexTensorOpsResponse:
        if not hasattr(self, "_session_tensor"):
            return ForexTensorOpsResponse(
                execution_id=request.execution_id,
                volatility_points=0, momentum_points=0,
                nan_injected=0, nan_remaining=0,
                status="failed", error="No tensor. Call preprocess first.",
            )
        return self._tensor_ops_task.execute(request, self._session_tensor)

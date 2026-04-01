"""ForexEURUSDService — orchestrates all Forex EUR/USD workflow tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest,
    ForexAutogradResponse,
    ForexDownloadRequest,
    ForexDownloadResponse,
    ForexIngestionRequest,
    ForexIngestionResponse,
    ForexPreprocessRequest,
    ForexPreprocessResponse,
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_eurusd.tasks.autograd_task import ForexAutogradTask
from quantedge_services.services.wfs.forex_eurusd.tasks.download_task import ForexDataDownloadTask
from quantedge_services.services.wfs.forex_eurusd.tasks.ingest_task import ForexIngestionTask
from quantedge_services.services.wfs.forex_eurusd.tasks.preprocess_task import ForexPreprocessTask
from quantedge_services.services.wfs.forex_eurusd.tasks.tensor_ops_task import ForexTensorOpsTask


class ForexEURUSDService:
    """Domain service for EUR/USD Forex workflows.

    Owns task orchestration and in-memory session state (DataFrame + Tensor).
    Called exclusively by FoundationsServiceFacade.

    Pipeline order:  download → ingest → preprocess → autograd / tensor_ops
    """

    def __init__(
        self,
        download_task: ForexDataDownloadTask,
        ingest_task: ForexIngestionTask,
        preprocess_task: ForexPreprocessTask,
        autograd_task: ForexAutogradTask,
        tensor_ops_task: ForexTensorOpsTask,
    ) -> None:
        self._download_task   = download_task
        self._ingest_task     = ingest_task
        self._preprocess_task = preprocess_task
        self._autograd_task   = autograd_task
        self._tensor_ops_task = tensor_ops_task
        self._logger = StructuredLogger(name=__name__)

        self._session_df:     pd.DataFrame | None  = None
        self._session_tensor: "torch.Tensor | None" = None  # noqa: F821
        self._parquet_path:   str = ""

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def download(self, request: ForexDownloadRequest) -> ForexDownloadResponse:
        return self._download_task.execute(request)

    def ingest(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        resp = self._ingest_task.execute(request)
        if resp.status == "success":
            # Read from parquet the task already saved — no double CSV parse.
            self._session_df = pd.read_parquet(resp.parquet_path)
            self._parquet_path = resp.parquet_path
            self._session_tensor = None  # invalidate downstream state
        return resp

    def preprocess(self, request: ForexPreprocessRequest) -> ForexPreprocessResponse:
        if self._session_df is None:
            return ForexPreprocessResponse(
                execution_id=request.execution_id,
                total_bars=0,
                train_bars=0,
                val_bars=0,
                test_bars=0,
                nan_filled=0,
                features=[],
                processed_parquet_path="",
                status="failed",
                error="No data loaded. Call /admin/foundations/forex/ingest first.",
            )
        resp, tensor = self._preprocess_task.execute(request, self._session_df)
        if resp.status == "success":
            self._session_tensor = tensor
        return resp

    def run_autograd(self, request: ForexAutogradRequest) -> ForexAutogradResponse:
        if self._session_tensor is None:
            return ForexAutogradResponse(
                execution_id=request.execution_id,
                manual_loss=0.0,
                autograd_loss=0.0,
                max_grad_diff=0.0,
                status="failed",
                error="No tensor available. Call /admin/foundations/forex/preprocess first.",
            )
        return self._autograd_task.execute(request, self._session_tensor)

    def run_tensor_ops(self, request: ForexTensorOpsRequest) -> ForexTensorOpsResponse:
        if self._session_tensor is None:
            return ForexTensorOpsResponse(
                execution_id=request.execution_id,
                volatility_points=0,
                momentum_points=0,
                nan_injected=0,
                nan_remaining=0,
                status="failed",
                error="No tensor available. Call /admin/foundations/forex/preprocess first.",
            )
        return self._tensor_ops_task.execute(request, self._session_tensor)

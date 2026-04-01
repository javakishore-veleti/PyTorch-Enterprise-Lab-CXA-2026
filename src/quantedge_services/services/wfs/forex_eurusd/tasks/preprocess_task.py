"""ForexPreprocessTask — normalize OHLCV, fill market-closed NaN gaps."""

from __future__ import annotations

import pandas as pd
import torch

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexPreprocessRequest,
    ForexPreprocessResponse,
)
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.logging import StructuredLogger


class ForexPreprocessTask:
    """Stateless preprocessing task. Receives a DataFrame via the session store
    (injected by ForexEURUSDService) and returns a response DTO."""

    def __init__(self, device_manager: DeviceManager) -> None:
        self._device_manager = device_manager
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: ForexPreprocessRequest,
        dataframe: pd.DataFrame,
    ) -> tuple[ForexPreprocessResponse, torch.Tensor]:
        nan_filled = 0
        try:
            df = dataframe.copy()
            if request.fill_gaps:
                before = int(df.isna().sum().sum())
                df = df.ffill()
                nan_filled = before - int(df.isna().sum().sum())

            if request.normalize:
                price_cols = ["open", "high", "low", "close"]
                df[price_cols] = (df[price_cols] - df[price_cols].min()) / (
                    df[price_cols].max() - df[price_cols].min()
                )
                vol_min, vol_max = df["volume"].min(), df["volume"].max()
                df["volume"] = (df["volume"] - vol_min) / (vol_max - vol_min)

            tensor = torch.tensor(
                df[["open", "high", "low", "close", "volume"]].values,
                dtype=torch.float32,
                device=self._device_manager.device,
            )
            self._logger.info(
                "forex_preprocessed",
                execution_id=request.execution_id,
                rows=len(df),
                nan_filled=nan_filled,
                tensor_shape=list(tensor.shape),
            )
            resp = ForexPreprocessResponse(
                execution_id=request.execution_id,
                rows_after=len(df),
                nan_filled=nan_filled,
                status="success",
            )
            return resp, tensor

        except Exception as exc:
            self._logger.error("forex_preprocess_failed", error=str(exc))
            return ForexPreprocessResponse(
                execution_id=request.execution_id, rows_after=0,
                nan_filled=0, status="failed", error=str(exc),
            ), torch.empty(0)

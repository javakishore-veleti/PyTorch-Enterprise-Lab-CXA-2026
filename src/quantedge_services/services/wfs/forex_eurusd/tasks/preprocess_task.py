"""ForexPreprocessTask — normalise OHLCV bars, split into train/val/test, return tensors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexPreprocessRequest,
    ForexPreprocessResponse,
)
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.logging import StructuredLogger

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "spread"]


class ForexPreprocessTask:
    """Stateless preprocessing task.

    Receives a DataFrame (OHLCV+spread bars from parquet) via the service
    session store and returns a rich response DTO + the full normalised tensor.

    Split strategy: chronological (no shuffle) to preserve time ordering.
    """

    def __init__(self, device_manager: DeviceManager) -> None:
        self._device_manager = device_manager
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: ForexPreprocessRequest,
        dataframe: pd.DataFrame,
    ) -> tuple[ForexPreprocessResponse, torch.Tensor]:
        try:
            df = dataframe.copy()
            available_cols = [c for c in _FEATURE_COLS if c in df.columns]

            nan_filled = 0
            if request.fill_gaps:
                before = int(df[available_cols].isna().sum().sum())
                df[available_cols] = df[available_cols].ffill().bfill()
                nan_filled = before - int(df[available_cols].isna().sum().sum())

            if request.normalize:
                df = self._normalise(df, available_cols, request.scaler_type)

            df = df[available_cols].dropna()
            total_bars = len(df)

            train_end = int(total_bars * request.train_ratio)
            val_end   = train_end + int(total_bars * request.val_ratio)

            train_df = df.iloc[:train_end]
            val_df   = df.iloc[train_end:val_end]
            test_df  = df.iloc[val_end:]

            processed_dir = Path("data/forex/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / f"eurusd_processed_{request.execution_id[:8]}.parquet"
            df.to_parquet(processed_path, index=True)

            tensor = torch.tensor(
                df.values,
                dtype=torch.float32,
                device=self._device_manager.device,
            )

            self._logger.info(
                "forex_preprocessed",
                execution_id=request.execution_id,
                total_bars=total_bars,
                train=len(train_df),
                val=len(val_df),
                test=len(test_df),
                nan_filled=nan_filled,
                tensor_shape=list(tensor.shape),
            )
            return ForexPreprocessResponse(
                execution_id=request.execution_id,
                total_bars=total_bars,
                train_bars=len(train_df),
                val_bars=len(val_df),
                test_bars=len(test_df),
                nan_filled=nan_filled,
                features=available_cols,
                processed_parquet_path=str(processed_path),
                status="success",
            ), tensor

        except Exception as exc:
            self._logger.error("forex_preprocess_failed", error=str(exc))
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
                error=str(exc),
            ), torch.empty(0)

    # ── Normalisation helpers ───────────────────────────────────────────────

    def _normalise(
        self,
        df: pd.DataFrame,
        cols: list[str],
        scaler_type: str,
    ) -> pd.DataFrame:
        if scaler_type == "minmax":
            col_min = df[cols].min()
            col_max = df[cols].max()
            denom = (col_max - col_min).replace(0, 1)  # avoid div-by-zero for flat columns
            df[cols] = (df[cols] - col_min) / denom
        elif scaler_type == "zscore":
            mean = df[cols].mean()
            std  = df[cols].std().replace(0, 1)
            df[cols] = (df[cols] - mean) / std
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'. Choose 'minmax' or 'zscore'.")
        return df

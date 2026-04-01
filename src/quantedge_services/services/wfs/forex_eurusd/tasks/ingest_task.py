"""ForexIngestionTask — parses HistData EUR/USD tick CSVs and resamples to OHLCV bars."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexIngestionRequest,
    ForexIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger

# HistData raw tick CSV — no header, three columns.
# Format: YYYYMMDD HHmmss[fff],bid,ask
# Example row:  20190103 220103,1.13490,1.13510
_TICK_COLS = ["datetime_str", "bid", "ask"]

# Some Kaggle-sourced EUR/USD CSVs ship as pre-built OHLCV (5 or 6 cols).
# We detect by column count and handle both.
_OHLCV_COLS = ["datetime", "open", "high", "low", "close", "volume"]


class ForexIngestionTask:
    """Reads raw HistData tick CSVs, resamples to OHLCV bars, persists as Parquet.

    Accepted CSV layouts (auto-detected by column count):
    - **Tick format** (3 cols, no header): ``YYYYMMDD HHmmss, bid, ask``
    - **OHLCV format** (5–6 cols, no header): ``datetime, open, high, low, close[, volume]``

    The output Parquet file has columns: ``open, high, low, close, volume``
    with a ``DatetimeIndex`` at the requested ``resample_freq``.
    """

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        execution_id = str(uuid.uuid4())
        try:
            data_dir = Path(request.data_dir)
            parquet_dir = Path(request.parquet_dir)
            parquet_dir.mkdir(parents=True, exist_ok=True)

            csv_files = sorted(data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in '{data_dir}'. "
                    "Run the download step first: POST /admin/foundations/forex/download"
                )

            frames: list[pd.DataFrame] = []
            for f in csv_files:
                if request.years and not any(str(y) in f.stem for y in request.years):
                    continue
                raw = self._read_csv(f, nrows=request.nrows)
                frames.append(raw)

            if not frames:
                raise ValueError(
                    f"No CSV files matched years filter {request.years} in '{data_dir}'."
                )

            ticks = pd.concat(frames, ignore_index=True).sort_values("datetime")
            ticks_loaded = len(ticks)

            ohlcv = self._resample_to_ohlcv(ticks, freq=request.resample_freq)
            bars_resampled = len(ohlcv)

            parquet_path = parquet_dir / f"eurusd_{request.resample_freq}.parquet"
            ohlcv.to_parquet(parquet_path, index=True)

            self._logger.info(
                "forex_ingested",
                execution_id=execution_id,
                ticks=ticks_loaded,
                bars=bars_resampled,
                parquet=str(parquet_path),
            )
            return ForexIngestionResponse(
                execution_id=execution_id,
                ticks_loaded=ticks_loaded,
                bars_resampled=bars_resampled,
                files_loaded=len(frames),
                parquet_path=str(parquet_path),
                status="success",
            )
        except Exception as exc:
            self._logger.error("forex_ingestion_failed", execution_id=execution_id, error=str(exc))
            return ForexIngestionResponse(
                execution_id=execution_id,
                ticks_loaded=0,
                bars_resampled=0,
                files_loaded=0,
                parquet_path="",
                status="failed",
                error=str(exc),
            )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _read_csv(self, path: Path, nrows: int | None) -> pd.DataFrame:
        """Read a CSV and normalise to a DataFrame with columns: datetime, bid, ask (or OHLCV)."""
        # Peek at the first row to detect column count
        sample = pd.read_csv(path, header=None, nrows=1)
        ncols = sample.shape[1]

        if ncols == 3:
            # Raw tick format: YYYYMMDD HHmmss, bid, ask
            df = pd.read_csv(path, header=None, names=_TICK_COLS, nrows=nrows)
            df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S")
            df = df.drop(columns=["datetime_str"])
        elif ncols >= 5:
            # Pre-built OHLCV format
            cols = _OHLCV_COLS[: ncols]
            df = pd.read_csv(path, header=None, names=cols, nrows=nrows)
            df["datetime"] = pd.to_datetime(df["datetime"])
            # Normalise to tick-like mid/ask for unified resampling path
            df["bid"] = df["open"]
            df["ask"] = df["open"]
        else:
            raise ValueError(
                f"Unexpected column count {ncols} in {path}. "
                "Expected 3-col tick format or 5/6-col OHLCV format."
            )

        return df[["datetime", "bid", "ask"]]

    def _resample_to_ohlcv(self, ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample tick-level bid/ask data to OHLCV bars at the given frequency."""
        ticks = ticks.set_index("datetime")
        mid = (ticks["bid"] + ticks["ask"]) / 2.0
        spread = ticks["ask"] - ticks["bid"]

        ohlcv = pd.DataFrame(
            {
                "open":   mid.resample(freq).first(),
                "high":   mid.resample(freq).max(),
                "low":    mid.resample(freq).min(),
                "close":  mid.resample(freq).last(),
                "volume": mid.resample(freq).count(),   # tick volume
                "spread": spread.resample(freq).mean(),
            }
        ).dropna(subset=["open"])  # drop bars with no ticks (market-closed gaps)

        return ohlcv


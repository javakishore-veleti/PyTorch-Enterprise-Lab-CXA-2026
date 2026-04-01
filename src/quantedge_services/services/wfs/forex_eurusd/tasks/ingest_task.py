"""ForexIngestionTask — loads EUR/USD tick CSVs into a DataFrame."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexIngestionRequest,
    ForexIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger

HISTDATA_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


class ForexIngestionTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        execution_id = str(uuid.uuid4())
        try:
            data_dir = Path(request.data_dir)
            csv_files = sorted(data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files in {data_dir}. "
                    "Download EUR/USD data from https://www.histdata.com"
                )
            frames = []
            for f in csv_files:
                if request.years and not any(str(y) in f.stem for y in request.years):
                    continue
                df = pd.read_csv(
                    f, header=None, names=HISTDATA_COLUMNS,
                    parse_dates=["datetime"], nrows=request.nrows,
                )
                frames.append(df)

            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values("datetime").set_index("datetime")
            self._logger.info("forex_ingested", execution_id=execution_id, rows=len(combined))

            return ForexIngestionResponse(
                execution_id=execution_id,
                rows_loaded=len(combined),
                files_loaded=len(frames),
                status="success",
            )
        except Exception as exc:
            self._logger.error("forex_ingestion_failed", execution_id=execution_id, error=str(exc))
            return ForexIngestionResponse(
                execution_id=execution_id, rows_loaded=0, files_loaded=0,
                status="failed", error=str(exc),
            )

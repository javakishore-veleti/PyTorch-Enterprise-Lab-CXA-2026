"""CFPBIngestionTask — reads HuggingFace parquet shards, cleans, saves filtered parquet."""
from __future__ import annotations
import uuid
from pathlib import Path
import pandas as pd
from quantedge_services.api.schemas.foundations.cfpb_schemas import CFPBIngestionRequest, CFPBIngestionResponse
from quantedge_services.core.logging import StructuredLogger

REQUIRED_COLS = ["Consumer complaint narrative", "Product"]


class CFPBIngestionTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        execution_id = str(uuid.uuid4())
        try:
            parquet_dir = Path(request.parquet_dir)
            parquet_files = sorted(parquet_dir.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files in '{parquet_dir}'. "
                    "Run POST /admin/foundations/cfpb/download first."
                )

            frames = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(frames, ignore_index=True)
            rows_raw = len(df)
            self._logger.info("cfpb_raw_loaded", execution_id=execution_id, rows=rows_raw)

            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"Required columns missing from parquet: {missing}. Available: {list(df.columns)}")

            df = df[df[request.narrative_col].notna() & df[request.product_col].notna()]
            df = df[df[request.narrative_col].str.strip().str.len() >= request.min_narrative_len]
            keep_cols = [
                request.narrative_col, request.product_col,
                "Date received", "Company", "State",
            ]
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols].reset_index(drop=True)

            output_dir = Path(request.output_parquet_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"cfpb_cleaned_{execution_id[:8]}.parquet"
            df.to_parquet(output_path, index=False)

            self._logger.info(
                "cfpb_ingested",
                execution_id=execution_id,
                rows_raw=rows_raw,
                rows_after=len(df),
                parquet=str(output_path),
            )
            return CFPBIngestionResponse(
                execution_id=execution_id,
                rows_raw=rows_raw,
                rows_after_filter=len(df),
                parquet_path=str(output_path),
                status="success",
            )
        except Exception as exc:
            self._logger.error("cfpb_ingestion_failed", error=str(exc))
            return CFPBIngestionResponse(
                execution_id=execution_id,
                rows_raw=0, rows_after_filter=0,
                parquet_path="", status="failed", error=str(exc),
            )

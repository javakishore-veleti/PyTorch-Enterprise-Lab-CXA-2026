from __future__ import annotations
from pathlib import Path
import pandas as pd
from quantedge_services.api.schemas.foundations.lora_schemas import (
    OAsst1IngestionRequest, OAsst1IngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger


class OAsst1IngestionTask:
    """Reads oasst1 parquet files from raw_dir, filters by lang + min_text_len,
    counts assistant/human turns, and saves filtered parquet to parquet_dir."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: OAsst1IngestionRequest) -> OAsst1IngestionResponse:
        raw_dir = Path(request.raw_dir)
        parquet_files = list(raw_dir.rglob("*.parquet"))

        if not parquet_files:
            self._logger.info("oasst1_ingest_skipped_no_files", raw_dir=str(raw_dir))
            return OAsst1IngestionResponse(
                execution_id=request.execution_id,
                rows_ingested=0,
                parquet_path="",
                assistant_turns=0,
                human_turns=0,
                status="skipped",
                error=f"No parquet files found in {raw_dir}",
            )

        try:
            frames = [pd.read_parquet(p) for p in parquet_files]
            df = pd.concat(frames, ignore_index=True)

            if "lang" in df.columns:
                df = df[df["lang"] == request.lang]
            if "text" in df.columns:
                df = df[df["text"].str.len() >= request.min_text_len]

            if request.max_rows is not None:
                df = df.iloc[: request.max_rows]

            assistant_turns = int((df["role"] == "assistant").sum()) if "role" in df.columns else 0
            human_turns = int((df["role"] == "prompter").sum()) if "role" in df.columns else 0

            out_dir = Path(request.parquet_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{request.execution_id}_oasst1.parquet"
            df.to_parquet(out_path, index=False)

            self._logger.info(
                "oasst1_ingest_done",
                rows=len(df),
                assistant=assistant_turns,
                human=human_turns,
            )
            return OAsst1IngestionResponse(
                execution_id=request.execution_id,
                rows_ingested=len(df),
                parquet_path=str(out_path),
                assistant_turns=assistant_turns,
                human_turns=human_turns,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("oasst1_ingest_failed", error=str(exc))
            return OAsst1IngestionResponse(
                execution_id=request.execution_id,
                rows_ingested=0,
                parquet_path="",
                assistant_turns=0,
                human_turns=0,
                status="failed",
                error=str(exc),
            )

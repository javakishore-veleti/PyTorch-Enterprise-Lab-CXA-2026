"""CFPBIngestionTask — downloads CFPB complaints dataset from HuggingFace."""

from __future__ import annotations

import uuid
from pathlib import Path

from datasets import load_dataset

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBIngestionRequest,
    CFPBIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger

HF_DATASET_ID = "cfpb/consumer-finance-complaints"


class CFPBIngestionTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        execution_id = str(uuid.uuid4())
        try:
            self._logger.info("cfpb_ingestion_start", execution_id=execution_id)
            ds = load_dataset(
                HF_DATASET_ID,
                split=request.split,
                cache_dir=str(Path(request.cache_dir)),
                streaming=request.streaming,
                trust_remote_code=True,
            )
            rows = len(ds) if not request.streaming else -1  # type: ignore[arg-type]
            self._logger.info("cfpb_ingested", execution_id=execution_id, rows=rows)
            return CFPBIngestionResponse(
                execution_id=execution_id, rows_loaded=rows, status="success"
            )
        except Exception as exc:
            self._logger.error("cfpb_ingestion_failed", error=str(exc))
            return CFPBIngestionResponse(
                execution_id=execution_id, rows_loaded=0,
                status="failed", error=str(exc),
            )

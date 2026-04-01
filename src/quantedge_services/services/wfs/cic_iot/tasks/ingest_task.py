from __future__ import annotations
from pathlib import Path
import pandas as pd
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    CICIoTIngestionRequest, CICIoTIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger


class CICIoTIngestionTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CICIoTIngestionRequest) -> CICIoTIngestionResponse:
        raw_dir = Path(request.raw_dir)
        parquet_dir = Path(request.parquet_dir)
        try:
            csv_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
            if not csv_files:
                return CICIoTIngestionResponse(
                    execution_id=request.execution_id,
                    rows_loaded=0,
                    parquet_path="",
                    num_classes=0,
                    class_distribution={},
                    status="skipped",
                    error="No CSV files found in raw_dir — run download step first",
                )

            frames = []
            for f in csv_files:
                df = pd.read_csv(f, nrows=request.nrows)
                frames.append(df)

            combined = pd.concat(frames, ignore_index=True)
            rows_loaded = len(combined)

            label_col = request.label_col
            if label_col in combined.columns:
                dist = combined[label_col].value_counts().to_dict()
                class_distribution = {str(k): int(v) for k, v in dist.items()}
                num_classes = len(class_distribution)
            else:
                class_distribution = {}
                num_classes = 0

            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / "cic_iot.parquet"
            combined.to_parquet(parquet_path, index=False)

            self._logger.info(
                "cic_iot_ingested",
                execution_id=request.execution_id,
                rows=rows_loaded,
                classes=num_classes,
            )
            return CICIoTIngestionResponse(
                execution_id=request.execution_id,
                rows_loaded=rows_loaded,
                parquet_path=str(parquet_path),
                num_classes=num_classes,
                class_distribution=class_distribution,
                status="success",
            )
        except Exception as exc:
            self._logger.error("cic_iot_ingest_failed", execution_id=request.execution_id, error=str(exc))
            return CICIoTIngestionResponse(
                execution_id=request.execution_id,
                rows_loaded=0,
                parquet_path="",
                num_classes=0,
                class_distribution={},
                status="failed",
                error=str(exc),
            )

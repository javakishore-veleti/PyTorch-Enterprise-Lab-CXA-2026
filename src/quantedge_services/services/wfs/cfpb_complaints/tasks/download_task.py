"""CFPBDataDownloadTask — downloads CFPB complaints from HuggingFace Hub as parquet."""
from __future__ import annotations
import uuid
from pathlib import Path
from quantedge_services.api.schemas.foundations.cfpb_schemas import CFPBDownloadRequest, CFPBDownloadResponse
from quantedge_services.core.logging import StructuredLogger


class CFPBDataDownloadTask:
    """Downloads cfpb/consumer-finance-complaints dataset from HuggingFace.
    
    Uses huggingface_hub.snapshot_download to get the parquet shards.
    Idempotent: skips if parquet files already present.
    """
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CFPBDownloadRequest) -> CFPBDownloadResponse:
        execution_id = str(uuid.uuid4())
        dest = Path(request.destination_dir)
        dest.mkdir(parents=True, exist_ok=True)

        existing = list(dest.rglob("*.parquet"))
        if existing:
            total_mb = sum(f.stat().st_size for f in existing) / 1024 / 1024
            self._logger.info("cfpb_download_skipped", files=len(existing))
            return CFPBDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=0,
                total_size_mb=round(total_mb, 1),
                parquet_path=str(dest),
                status="skipped",
                message=f"{len(existing)} parquet shard(s) already present. Delete to re-download.",
            )

        try:
            from huggingface_hub import snapshot_download
            self._logger.info("cfpb_download_start", dataset_id=request.hf_dataset_id)
            local_dir = snapshot_download(
                repo_id=request.hf_dataset_id,
                repo_type="dataset",
                local_dir=str(dest),
            )
            parquet_files = list(Path(local_dir).rglob("*.parquet"))
            total_mb = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
            self._logger.info("cfpb_download_done", files=len(parquet_files), mb=round(total_mb, 1))
            return CFPBDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=len(parquet_files),
                total_size_mb=round(total_mb, 1),
                parquet_path=str(dest),
                status="success",
                message=f"Downloaded {len(parquet_files)} parquet shard(s) → {dest}",
            )
        except ImportError:
            msg = "huggingface-hub not installed. Run: pip install huggingface-hub"
            return CFPBDownloadResponse(
                execution_id=execution_id, destination_dir=str(dest),
                files_downloaded=0, total_size_mb=0.0, parquet_path="",
                status="failed", error=msg,
            )
        except Exception as exc:
            self._logger.error("cfpb_download_failed", error=str(exc))
            return CFPBDownloadResponse(
                execution_id=execution_id, destination_dir=str(dest),
                files_downloaded=0, total_size_mb=0.0, parquet_path="",
                status="failed", error=str(exc),
            )

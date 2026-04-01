from __future__ import annotations
from pathlib import Path
from quantedge_services.api.schemas.foundations.lora_schemas import (
    OAsst1DownloadRequest, OAsst1DownloadResponse,
)
from quantedge_services.core.logging import StructuredLogger


class OAsst1DownloadTask:
    """Downloads the OpenAssistant oasst1 dataset from HuggingFace Hub.

    Idempotent: skips if files already present and force_redownload is False.
    Gracefully handles missing HF credentials or hub library.
    """

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: OAsst1DownloadRequest) -> OAsst1DownloadResponse:
        dest = Path(request.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        existing = list(dest.rglob("*.parquet")) + list(dest.rglob("*.jsonl"))
        if existing and not request.force_redownload:
            self._logger.info("oasst1_download_skipped", files=len(existing))
            return OAsst1DownloadResponse(
                execution_id=request.execution_id,
                dest_dir=str(dest),
                files_downloaded=0,
                status="download_skipped",
                error=f"{len(existing)} file(s) already present; set force_redownload=True to re-download.",
            )

        try:
            from huggingface_hub import snapshot_download
            self._logger.info("oasst1_download_start", repo_id=request.hf_repo_id)
            local_dir = snapshot_download(
                repo_id=request.hf_repo_id,
                repo_type="dataset",
                local_dir=str(dest),
            )
            downloaded = list(Path(local_dir).rglob("*.parquet")) + list(Path(local_dir).rglob("*.jsonl"))
            self._logger.info("oasst1_download_done", files=len(downloaded))
            return OAsst1DownloadResponse(
                execution_id=request.execution_id,
                dest_dir=str(dest),
                files_downloaded=len(downloaded),
                status="success",
                error=None,
            )
        except ImportError:
            msg = "huggingface-hub not installed. Run: pip install huggingface-hub"
            self._logger.warning("oasst1_download_skipped_no_hub")
            return OAsst1DownloadResponse(
                execution_id=request.execution_id,
                dest_dir=str(dest),
                files_downloaded=0,
                status="download_skipped",
                error=msg,
            )
        except Exception as exc:
            self._logger.error("oasst1_download_failed", error=str(exc))
            return OAsst1DownloadResponse(
                execution_id=request.execution_id,
                dest_dir=str(dest),
                files_downloaded=0,
                status="failed",
                error=str(exc),
            )

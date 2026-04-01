from __future__ import annotations
from pathlib import Path
from quantedge_services.api.schemas.foundations.attention_schemas import (
    CMAPSSDownloadRequest, CMAPSSDownloadResponse,
)
from quantedge_services.core.logging import StructuredLogger


class CMAPSSDownloadTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CMAPSSDownloadRequest) -> CMAPSSDownloadResponse:
        dest = Path(request.dest_dir)
        try:
            if dest.exists() and any(dest.iterdir()) and not request.force_redownload:
                existing = list(dest.iterdir())
                return CMAPSSDownloadResponse(
                    execution_id=request.execution_id,
                    dest_dir=str(dest),
                    files_downloaded=len(existing),
                    status="success",
                    error=None,
                )
            dest.mkdir(parents=True, exist_ok=True)
            try:
                import kaggle
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(
                    request.kaggle_dataset, path=str(dest), unzip=True
                )
                files = list(dest.iterdir())
                return CMAPSSDownloadResponse(
                    execution_id=request.execution_id,
                    dest_dir=str(dest),
                    files_downloaded=len(files),
                    status="success",
                    error=None,
                )
            except Exception as kaggle_exc:
                self._logger.info(
                    "cmapss_download_skipped",
                    execution_id=request.execution_id,
                    reason=str(kaggle_exc),
                )
                return CMAPSSDownloadResponse(
                    execution_id=request.execution_id,
                    dest_dir=str(dest),
                    files_downloaded=0,
                    status="download_skipped",
                    error=str(kaggle_exc),
                )
        except Exception as exc:
            return CMAPSSDownloadResponse(
                execution_id=request.execution_id,
                dest_dir=str(dest),
                files_downloaded=0,
                status="download_skipped",
                error=str(exc),
            )

"""ForexDataDownloadTask — downloads EUR/USD HistData ticks from Kaggle or HuggingFace."""

from __future__ import annotations

import uuid
from pathlib import Path

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexDownloadRequest,
    ForexDownloadResponse,
)
from quantedge_services.core.logging import StructuredLogger


class ForexDataDownloadTask:
    """Downloads raw HistData EUR/USD tick CSVs.

    Supports two backends controlled by ``ForexDownloadRequest.use_huggingface``:
    - ``False``  → Kaggle  (requires ~/.kaggle/kaggle.json or KAGGLE_* env vars)
    - ``True``   → HuggingFace Hub (requires HF_TOKEN env var for gated repos)

    If the destination directory already contains CSV files the download is
    skipped and a "skipped" status is returned — safe to call idempotently.
    """

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: ForexDownloadRequest) -> ForexDownloadResponse:
        execution_id = str(uuid.uuid4())
        dest = Path(request.destination_dir)
        dest.mkdir(parents=True, exist_ok=True)

        existing = list(dest.glob("*.csv"))
        if existing:
            self._logger.info(
                "forex_download_skipped",
                execution_id=execution_id,
                reason="CSV files already present",
                count=len(existing),
            )
            return ForexDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=0,
                total_size_mb=sum(f.stat().st_size for f in existing) / 1024 / 1024,
                status="skipped",
                message=f"{len(existing)} CSV file(s) already in {dest}. Delete to re-download.",
            )

        if request.use_huggingface:
            return self._download_huggingface(execution_id, request, dest)
        return self._download_kaggle(execution_id, request, dest)

    # ── Kaggle backend ──────────────────────────────────────────────────────

    def _download_kaggle(
        self,
        execution_id: str,
        request: ForexDownloadRequest,
        dest: Path,
    ) -> ForexDownloadResponse:
        try:
            import kagglehub  # pip install kagglehub

            self._logger.info(
                "forex_kaggle_download_start",
                execution_id=execution_id,
                dataset=request.kaggle_dataset,
            )
            download_path = kagglehub.dataset_download(request.kaggle_dataset)
            src = Path(download_path)
            csv_files = list(src.rglob("*.csv"))
            for f in csv_files:
                target = dest / f.name
                target.write_bytes(f.read_bytes())

            total_mb = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
            self._logger.info(
                "forex_kaggle_download_done",
                execution_id=execution_id,
                files=len(csv_files),
                total_mb=round(total_mb, 1),
            )
            return ForexDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=len(csv_files),
                total_size_mb=round(total_mb, 1),
                status="success",
                message=f"Downloaded {len(csv_files)} file(s) from Kaggle → {dest}",
            )
        except ImportError:
            return self._missing_package_response(
                execution_id, dest, "kagglehub",
                "pip install kagglehub  # then set KAGGLE_USERNAME + KAGGLE_KEY env vars",
            )
        except Exception as exc:
            self._logger.error("forex_kaggle_download_failed", error=str(exc))
            return ForexDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=0,
                total_size_mb=0.0,
                status="failed",
                error=str(exc),
            )

    # ── HuggingFace backend ─────────────────────────────────────────────────

    def _download_huggingface(
        self,
        execution_id: str,
        request: ForexDownloadRequest,
        dest: Path,
    ) -> ForexDownloadResponse:
        try:
            from huggingface_hub import snapshot_download  # pip install huggingface-hub

            self._logger.info(
                "forex_hf_download_start",
                execution_id=execution_id,
                dataset_id=request.hf_dataset_id,
            )
            local_dir = snapshot_download(
                repo_id=request.hf_dataset_id,
                repo_type="dataset",
                local_dir=str(dest),
            )
            csv_files = list(Path(local_dir).rglob("*.csv"))
            total_mb = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
            self._logger.info(
                "forex_hf_download_done",
                execution_id=execution_id,
                files=len(csv_files),
                total_mb=round(total_mb, 1),
            )
            return ForexDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=len(csv_files),
                total_size_mb=round(total_mb, 1),
                status="success",
                message=f"Downloaded {len(csv_files)} file(s) from HuggingFace → {dest}",
            )
        except ImportError:
            return self._missing_package_response(
                execution_id, dest, "huggingface-hub",
                "pip install huggingface-hub  # then set HF_TOKEN env var",
            )
        except Exception as exc:
            self._logger.error("forex_hf_download_failed", error=str(exc))
            return ForexDownloadResponse(
                execution_id=execution_id,
                destination_dir=str(dest),
                files_downloaded=0,
                total_size_mb=0.0,
                status="failed",
                error=str(exc),
            )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _missing_package_response(
        self,
        execution_id: str,
        dest: Path,
        package: str,
        install_hint: str,
    ) -> ForexDownloadResponse:
        msg = f"Package '{package}' not installed. Run: {install_hint}"
        self._logger.warning("forex_download_missing_package", package=package)
        return ForexDownloadResponse(
            execution_id=execution_id,
            destination_dir=str(dest),
            files_downloaded=0,
            total_size_mb=0.0,
            status="failed",
            error=msg,
        )

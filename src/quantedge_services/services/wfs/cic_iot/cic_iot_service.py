from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    CICIoTDownloadRequest, CICIoTDownloadResponse,
    CICIoTIngestionRequest, CICIoTIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.cic_iot.tasks.download_task import CICIoTDownloadTask
from quantedge_services.services.wfs.cic_iot.tasks.ingest_task import CICIoTIngestionTask


class CICIoTService:
    def __init__(
        self,
        download_task: CICIoTDownloadTask,
        ingest_task: CICIoTIngestionTask,
    ) -> None:
        self._download_task = download_task
        self._ingest_task = ingest_task
        self._logger = StructuredLogger(name=__name__)

    async def download(self, request: CICIoTDownloadRequest) -> CICIoTDownloadResponse:
        return await asyncio.to_thread(self._download_task.execute, request)

    async def ingest(self, request: CICIoTIngestionRequest) -> CICIoTIngestionResponse:
        return await asyncio.to_thread(self._ingest_task.execute, request)

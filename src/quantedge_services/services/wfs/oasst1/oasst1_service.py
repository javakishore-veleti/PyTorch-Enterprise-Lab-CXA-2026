from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.lora_schemas import (
    OAsst1DownloadRequest, OAsst1DownloadResponse,
    OAsst1IngestionRequest, OAsst1IngestionResponse,
)
from quantedge_services.services.wfs.oasst1.tasks.download_task import OAsst1DownloadTask
from quantedge_services.services.wfs.oasst1.tasks.ingest_task import OAsst1IngestionTask


class OAsst1Service:
    def __init__(
        self,
        download_task: OAsst1DownloadTask,
        ingest_task: OAsst1IngestionTask,
    ) -> None:
        self._download = download_task
        self._ingest = ingest_task

    async def download(self, request: OAsst1DownloadRequest) -> OAsst1DownloadResponse:
        return await asyncio.to_thread(self._download.execute, request)

    async def ingest(self, request: OAsst1IngestionRequest) -> OAsst1IngestionResponse:
        return await asyncio.to_thread(self._ingest.execute, request)

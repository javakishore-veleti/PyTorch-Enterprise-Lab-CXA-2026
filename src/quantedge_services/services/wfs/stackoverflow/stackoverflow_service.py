"""StackOverflowService — async wrapper around download and ingestion tasks."""
from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest,
    StackOverflowDownloadResult,
    StackOverflowIngestionRequest,
    StackOverflowIngestionResult,
)
from quantedge_services.services.wfs.stackoverflow.tasks.download_task import StackOverflowDownloadTask
from quantedge_services.services.wfs.stackoverflow.tasks.ingest_task import StackOverflowIngestionTask


class StackOverflowService:
    def __init__(
        self,
        download_task: StackOverflowDownloadTask,
        ingest_task: StackOverflowIngestionTask,
    ) -> None:
        self._download = download_task
        self._ingest = ingest_task

    async def download(self, request: StackOverflowDownloadRequest) -> StackOverflowDownloadResult:
        return await asyncio.to_thread(self._download.execute, request)

    async def ingest(self, request: StackOverflowIngestionRequest) -> StackOverflowIngestionResult:
        return await asyncio.to_thread(self._ingest.execute, request)

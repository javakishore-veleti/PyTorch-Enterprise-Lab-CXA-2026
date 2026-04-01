from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.attention_schemas import (
    CMAPSSDownloadRequest, CMAPSSDownloadResponse,
    CMAPSSIngestionRequest, CMAPSSIngestionResponse,
)
from quantedge_services.services.wfs.cmapss.tasks.download_task import CMAPSSDownloadTask
from quantedge_services.services.wfs.cmapss.tasks.ingest_task import CMAPSSIngestionTask


class CMAPSSService:
    def __init__(self, download_task: CMAPSSDownloadTask, ingest_task: CMAPSSIngestionTask) -> None:
        self._download = download_task
        self._ingest = ingest_task

    async def download(self, request: CMAPSSDownloadRequest) -> CMAPSSDownloadResponse:
        return await asyncio.to_thread(self._download.execute, request)

    async def ingest(self, request: CMAPSSIngestionRequest) -> CMAPSSIngestionResponse:
        return await asyncio.to_thread(self._ingest.execute, request)

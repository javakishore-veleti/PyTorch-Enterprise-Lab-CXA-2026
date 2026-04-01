from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    DataloaderTuneRequest, DataloaderTuneResponse,
    MemorySummaryRequest, MemorySummaryResponse,
    ProfilerRunRequest, ProfilerRunResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.profiling.tasks.dataloader_tune_task import DataloaderTuneTask
from quantedge_services.services.wfs.profiling.tasks.memory_summary_task import MemorySummaryTask
from quantedge_services.services.wfs.profiling.tasks.profiler_run_task import ProfilerRunTask


class ProfilingService:
    def __init__(
        self,
        profiler_task: ProfilerRunTask,
        memory_task: MemorySummaryTask,
        dataloader_task: DataloaderTuneTask,
    ) -> None:
        self._profiler_task = profiler_task
        self._memory_task = memory_task
        self._dataloader_task = dataloader_task
        self._logger = StructuredLogger(name=__name__)

    async def run_profiler(self, request: ProfilerRunRequest) -> ProfilerRunResponse:
        return await asyncio.to_thread(self._profiler_task.execute, request)

    async def memory_summary(self, request: MemorySummaryRequest) -> MemorySummaryResponse:
        return await asyncio.to_thread(self._memory_task.execute, request)

    async def tune_dataloader(self, request: DataloaderTuneRequest) -> DataloaderTuneResponse:
        return await asyncio.to_thread(self._dataloader_task.execute, request)

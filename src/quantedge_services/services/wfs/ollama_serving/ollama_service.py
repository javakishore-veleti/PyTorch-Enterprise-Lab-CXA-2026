"""OllamaService — async wrapper around Ollama inference and merge tasks."""
from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    OllamaInferRequest,
    OllamaInferResult,
    OllamaMergeRequest,
    OllamaMergeResult,
)
from quantedge_services.services.wfs.ollama_serving.tasks.ollama_infer_task import OllamaInferTask
from quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task import OllamaMergeTask


class OllamaService:
    def __init__(
        self,
        infer_task: OllamaInferTask,
        merge_task: OllamaMergeTask,
    ) -> None:
        self._infer = infer_task
        self._merge = merge_task

    async def infer(self, request: OllamaInferRequest) -> OllamaInferResult:
        return await asyncio.to_thread(self._infer.execute, request)

    async def merge(self, request: OllamaMergeRequest) -> OllamaMergeResult:
        return await asyncio.to_thread(self._merge.execute, request)

from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionExtractRequest, AttentionExtractResponse,
    AttentionHeatmapRequest, AttentionHeatmapResponse,
    ArchDecisionRequest, ArchDecisionResponse,
)
from quantedge_services.services.wfs.attention_viz.tasks.attention_extract_task import AttentionExtractTask
from quantedge_services.services.wfs.attention_viz.tasks.attention_heatmap_task import AttentionHeatmapTask
from quantedge_services.services.wfs.attention_viz.tasks.arch_decision_task import ArchDecisionTask
from quantedge_services.core.logging import StructuredLogger


class AttentionVizService:
    def __init__(
        self,
        extract_task: AttentionExtractTask,
        heatmap_task: AttentionHeatmapTask,
        arch_decision_task: ArchDecisionTask,
    ) -> None:
        self._extract = extract_task
        self._heatmap = heatmap_task
        self._arch = arch_decision_task
        self._logger = StructuredLogger(name=__name__)

    async def extract_weights(self, request: AttentionExtractRequest) -> AttentionExtractResponse:
        return await asyncio.to_thread(self._extract.execute, request)

    async def generate_heatmaps(self, request: AttentionHeatmapRequest) -> AttentionHeatmapResponse:
        return await asyncio.to_thread(self._heatmap.execute, request)

    async def architecture_decision(self, request: ArchDecisionRequest) -> ArchDecisionResponse:
        return await asyncio.to_thread(self._arch.execute, request)

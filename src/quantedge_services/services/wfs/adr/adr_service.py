"""ADRService — async wrapper around ADR generation and listing."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    ADRGenerateRequest, ADRGenerateResult,
    ADRListRequest, ADRListResult,
)
from quantedge_services.services.wfs.adr.tasks.adr_generate_task import ADRGeneratorTask
from quantedge_services.services.wfs.adr.tasks.adr_list_task import ADRListTask


class ADRService:
    """Orchestrates Architecture Decision Record generation and listing."""

    def __init__(self, generate_task: ADRGeneratorTask, list_task: ADRListTask) -> None:
        self._generate = generate_task
        self._list = list_task

    async def generate_adrs(self, request: ADRGenerateRequest) -> ADRGenerateResult:
        return self._generate.execute(request)

    async def list_adrs(self, request: ADRListRequest) -> ADRListResult:
        return self._list.execute(request)

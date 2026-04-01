"""ModelRegistryService — async wrapper for model registry list and promote tasks."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    ModelRegistryListRequest, ModelRegistryListResult,
    ModelRegistryPromoteRequest, ModelRegistryPromoteResult,
)
from quantedge_services.services.wfs.model_registry.tasks.registry_list_task import ModelRegistryListTask
from quantedge_services.services.wfs.model_registry.tasks.registry_promote_task import ModelRegistryPromoteTask


class ModelRegistryService:
    """Wraps model registry tasks as async methods."""

    def __init__(self, list_task: ModelRegistryListTask, promote_task: ModelRegistryPromoteTask) -> None:
        self._list_task = list_task
        self._promote_task = promote_task

    async def list_models(self, request: ModelRegistryListRequest) -> ModelRegistryListResult:
        return self._list_task.execute(request)

    async def promote(self, request: ModelRegistryPromoteRequest) -> ModelRegistryPromoteResult:
        return self._promote_task.execute(request)

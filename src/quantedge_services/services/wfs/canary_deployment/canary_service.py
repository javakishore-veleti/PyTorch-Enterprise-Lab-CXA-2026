"""CanaryService — async wrapper for canary deploy and eval tasks."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    CanaryDeployRequest, CanaryDeployResult,
    CanaryEvalRequest, CanaryEvalResult,
)
from quantedge_services.services.wfs.canary_deployment.tasks.canary_deploy_task import CanaryDeployTask
from quantedge_services.services.wfs.canary_deployment.tasks.canary_eval_task import CanaryEvalTask


class CanaryService:
    """Wraps canary deploy and eval tasks as async methods."""

    def __init__(self, deploy_task: CanaryDeployTask, eval_task: CanaryEvalTask) -> None:
        self._deploy_task = deploy_task
        self._eval_task = eval_task

    async def deploy(self, request: CanaryDeployRequest) -> CanaryDeployResult:
        return self._deploy_task.execute(request)

    async def evaluate(self, request: CanaryEvalRequest) -> CanaryEvalResult:
        return self._eval_task.execute(request)

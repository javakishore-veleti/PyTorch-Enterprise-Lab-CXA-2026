"""ExperimentTrackingService — async wrapper around MLflow tasks."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    MLflowLogRequest, MLflowLogResult,
    MLflowRegisterRequest, MLflowRegisterResult,
)
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_log_task import MLflowLogTask
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_register_task import MLflowRegisterTask


class ExperimentTrackingService:
    """Wraps MLflow log and register tasks as async methods."""

    def __init__(self, log_task: MLflowLogTask, register_task: MLflowRegisterTask) -> None:
        self._log_task = log_task
        self._register_task = register_task

    async def log_run(self, request: MLflowLogRequest) -> MLflowLogResult:
        return self._log_task.execute(request)

    async def register_model(self, request: MLflowRegisterRequest) -> MLflowRegisterResult:
        return self._register_task.execute(request)

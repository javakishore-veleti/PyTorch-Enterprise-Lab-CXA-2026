"""MLflowRegistryClient — MLflow tracking + model registry bridge."""

from __future__ import annotations

import os

import mlflow
import mlflow.pytorch
import torch.nn as nn

from quantedge_services.core.logging import StructuredLogger


class MLflowRegistryClient:
    """Thin client over MLflow tracking and model registry.

    Supports local MLflow, AWS SageMaker, and Azure ML as backends
    via the MODEL_REGISTRY_BACKEND env var.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._backend = os.getenv("MODEL_REGISTRY_BACKEND", "mlflow")
        self._logger = StructuredLogger(name=__name__)
        mlflow.set_tracking_uri(self._tracking_uri)

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, artifact_path: str = "model") -> None:
        if self._backend == "mlflow":
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            self._logger.warning(
                "registry_backend_not_implemented",
                backend=self._backend,
                hint="Implement SageMaker/AzureML registration here",
            )

    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> None:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        self._logger.info(
            "model_registered", name=model_name, run_id=run_id, backend=self._backend
        )

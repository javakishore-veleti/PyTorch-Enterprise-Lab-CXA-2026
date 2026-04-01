"""ModelRegistryPromoteTask — transitions a model version to a new stage."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    ModelRegistryPromoteRequest, ModelRegistryPromoteResult,
)


class ModelRegistryPromoteTask:
    """Promotes a model version to a target stage; mock fallback."""

    def execute(self, request: ModelRegistryPromoteRequest) -> ModelRegistryPromoteResult:
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415
        except ImportError:
            return ModelRegistryPromoteResult(
                model_name=request.model_name,
                version=request.version,
                previous_stage="Staging",
                new_stage=request.target_stage,
                status="mlflow_not_installed",
            )

        try:
            client = MlflowClient()
            mv = client.get_model_version(request.model_name, request.version)
            previous_stage = mv.current_stage
            client.transition_model_version_stage(
                name=request.model_name,
                version=request.version,
                stage=request.target_stage,
            )
            return ModelRegistryPromoteResult(
                model_name=request.model_name,
                version=request.version,
                previous_stage=previous_stage,
                new_stage=request.target_stage,
                status="success",
            )
        except Exception:  # noqa: BLE001
            return ModelRegistryPromoteResult(
                model_name=request.model_name,
                version=request.version,
                previous_stage="Staging",
                new_stage=request.target_stage,
                status="mlflow_error",
            )

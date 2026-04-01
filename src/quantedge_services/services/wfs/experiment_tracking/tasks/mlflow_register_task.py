"""MLflowRegisterTask — registers a model version in the MLflow model registry."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    MLflowRegisterRequest, MLflowRegisterResult,
)


class MLflowRegisterTask:
    """Registers a trained model in the MLflow model registry; mock fallback."""

    def execute(self, request: MLflowRegisterRequest) -> MLflowRegisterResult:
        try:
            import mlflow  # noqa: PLC0415
            from mlflow.tracking import MlflowClient  # noqa: PLC0415
        except ImportError:
            return MLflowRegisterResult(
                model_name=request.model_name,
                version="1",
                stage=request.stage,
                status="mlflow_not_installed",
            )

        try:
            model_uri = f"runs:/{request.run_id}/{request.artifact_path}"
            mv = mlflow.register_model(model_uri, request.model_name)
            client = MlflowClient()
            client.transition_model_version_stage(
                name=request.model_name,
                version=mv.version,
                stage=request.stage,
            )
            return MLflowRegisterResult(
                model_name=mv.name,
                version=str(mv.version),
                stage=request.stage,
                status="success",
            )
        except Exception:  # noqa: BLE001
            return MLflowRegisterResult(
                model_name=request.model_name,
                version="1",
                stage=request.stage,
                status="mlflow_error",
            )

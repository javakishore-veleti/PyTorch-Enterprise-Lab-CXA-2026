"""MLflowLogTask — logs experiment parameters, metrics, and artifacts."""
from __future__ import annotations
import os
from uuid import uuid4
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    MLflowLogRequest, MLflowLogResult,
)


class MLflowLogTask:
    """Logs a run to MLflow; falls back to mock if mlflow is not installed."""

    def execute(self, request: MLflowLogRequest) -> MLflowLogResult:
        try:
            import mlflow  # noqa: PLC0415
        except ImportError:
            short_id = str(uuid4()).replace("-", "")[:8]
            return MLflowLogResult(
                run_id=f"mock-run-{short_id}",
                experiment_id="0",
                artifact_uri="./mlruns",
                status="mlflow_not_installed",
            )

        mlflow.set_experiment(request.experiment_name)
        with mlflow.start_run(run_name=request.run_name) as run:
            if request.params:
                mlflow.log_params(request.params)
            if request.metrics:
                mlflow.log_metrics(request.metrics)
            if request.tags:
                mlflow.set_tags(request.tags)
            for path in request.artifact_paths:
                if os.path.exists(path):
                    mlflow.log_artifact(path)
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            artifact_uri = run.info.artifact_uri

        return MLflowLogResult(
            run_id=run_id,
            experiment_id=str(experiment_id),
            artifact_uri=artifact_uri,
            status="success",
        )

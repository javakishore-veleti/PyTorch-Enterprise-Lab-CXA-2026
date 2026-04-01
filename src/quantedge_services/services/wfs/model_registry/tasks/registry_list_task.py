"""ModelRegistryListTask — lists registered models from MLflow or mock."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    ModelRegistryListRequest, ModelRegistryListResult,
)


class ModelRegistryListTask:
    """Lists models in the MLflow model registry; returns mock if not installed."""

    _MOCK_MODELS = [
        {"name": "ForexTransformer-v1", "latest_versions": [{"version": "1", "stage": "Staging"}]},
        {"name": "LoRAAdapter-v1", "latest_versions": [{"version": "2", "stage": "Production"}]},
        {"name": "QuantizedMLP-v1", "latest_versions": [{"version": "1", "stage": "Archived"}]},
    ]

    def execute(self, request: ModelRegistryListRequest) -> ModelRegistryListResult:
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415
        except ImportError:
            models = self._MOCK_MODELS
            if request.filter_name:
                models = [m for m in models if request.filter_name.lower() in m["name"].lower()]
            models = models[: request.max_results]
            return ModelRegistryListResult(
                models=models,
                total_count=len(models),
                status="mlflow_not_installed",
            )

        client = MlflowClient()
        filter_str = f"name LIKE '%{request.filter_name}%'" if request.filter_name else ""
        registered = client.search_registered_models(
            filter_string=filter_str,
            max_results=request.max_results,
        )
        models = [
            {"name": m.name, "latest_versions": [
                {"version": v.version, "stage": v.current_stage}
                for v in m.latest_versions
            ]}
            for m in registered
        ]
        return ModelRegistryListResult(
            models=models,
            total_count=len(models),
            status="success",
        )

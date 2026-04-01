"""Tests for Week 11 — Experiment Tracking tasks, services, and facade."""
from __future__ import annotations
import pytest
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    MLflowLogRequest, MLflowLogResult,
    MLflowRegisterRequest, MLflowRegisterResult,
    ModelRegistryListRequest, ModelRegistryListResult,
    ModelRegistryPromoteRequest, ModelRegistryPromoteResult,
)
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_log_task import MLflowLogTask
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_register_task import MLflowRegisterTask
from quantedge_services.services.wfs.experiment_tracking.tracking_service import ExperimentTrackingService
from quantedge_services.services.wfs.model_registry.tasks.registry_list_task import ModelRegistryListTask
from quantedge_services.services.wfs.model_registry.tasks.registry_promote_task import ModelRegistryPromoteTask
from quantedge_services.services.wfs.model_registry.registry_service import ModelRegistryService


def test_mlflow_log_task_no_mlflow_returns_mock():
    task = MLflowLogTask()
    req = MLflowLogRequest(experiment_name="test-exp", run_name="run-1")
    result = task.execute(req)
    assert isinstance(result, MLflowLogResult)


def test_mlflow_log_task_result_is_dto():
    task = MLflowLogTask()
    req = MLflowLogRequest(experiment_name="test-exp", run_name="run-1", params={"lr": "0.001"})
    result = task.execute(req)
    assert hasattr(result, "run_id")
    assert hasattr(result, "status")


def test_mlflow_log_task_run_id_not_empty():
    task = MLflowLogTask()
    req = MLflowLogRequest(experiment_name="test-exp", run_name="run-1")
    result = task.execute(req)
    assert result.run_id != ""


def test_mlflow_register_task_no_mlflow_returns_mock():
    task = MLflowRegisterTask()
    req = MLflowRegisterRequest(run_id="abc123", model_name="ForexTransformer")
    result = task.execute(req)
    assert isinstance(result, MLflowRegisterResult)


def test_mlflow_register_task_result_fields():
    task = MLflowRegisterTask()
    req = MLflowRegisterRequest(run_id="abc123", model_name="ForexTransformer", stage="Production")
    result = task.execute(req)
    assert result.model_name == "ForexTransformer"
    assert result.version != ""
    assert result.stage in ("Staging", "Production", "Archived")


def test_registry_list_task_returns_mock_models():
    task = ModelRegistryListTask()
    req = ModelRegistryListRequest()
    result = task.execute(req)
    assert isinstance(result, ModelRegistryListResult)
    assert len(result.models) > 0


def test_registry_list_task_count_matches_models():
    task = ModelRegistryListTask()
    req = ModelRegistryListRequest()
    result = task.execute(req)
    assert result.total_count == len(result.models)


def test_registry_promote_task_returns_dto():
    task = ModelRegistryPromoteTask()
    req = ModelRegistryPromoteRequest(model_name="ForexTransformer-v1", version="1", target_stage="Production")
    result = task.execute(req)
    assert isinstance(result, ModelRegistryPromoteResult)
    assert result.new_stage == "Production"


def test_tracking_service_has_log_method():
    svc = ExperimentTrackingService(
        log_task=MLflowLogTask(),
        register_task=MLflowRegisterTask(),
    )
    assert hasattr(svc, "log_run")


def test_tracking_service_has_register_method():
    svc = ExperimentTrackingService(
        log_task=MLflowLogTask(),
        register_task=MLflowRegisterTask(),
    )
    assert hasattr(svc, "register_model")


def test_registry_service_has_list_method():
    svc = ModelRegistryService(
        list_task=ModelRegistryListTask(),
        promote_task=ModelRegistryPromoteTask(),
    )
    assert hasattr(svc, "list_models")


def test_registry_service_has_promote_method():
    svc = ModelRegistryService(
        list_task=ModelRegistryListTask(),
        promote_task=ModelRegistryPromoteTask(),
    )
    assert hasattr(svc, "promote")


def test_facade_has_submit_mlflow_log():
    from quantedge_services.api.dependencies import get_container
    container = get_container()
    assert hasattr(container.foundations_facade, "submit_mlflow_log")


def test_facade_has_submit_registry_promote():
    from quantedge_services.api.dependencies import get_container
    container = get_container()
    assert hasattr(container.foundations_facade, "submit_registry_promote")

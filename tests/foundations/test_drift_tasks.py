"""Tests for drift detection tasks and service — Week 12."""
from __future__ import annotations
import pytest
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    DataDriftRequest, DataDriftResult,
    ConceptDriftRequest, ConceptDriftResult,
)
from quantedge_services.services.wfs.drift_detection.tasks.data_drift_task import DataDriftTask
from quantedge_services.services.wfs.drift_detection.tasks.concept_drift_task import ConceptDriftTask
from quantedge_services.services.wfs.drift_detection.drift_service import DriftDetectionService


# ── DataDriftTask ──────────────────────────────────────────────────────────


def test_data_drift_task_returns_dto():
    task = DataDriftTask()
    req = DataDriftRequest(reference_data_path="/nonexistent/ref.parquet", current_data_path="/nonexistent/cur.parquet")
    result = task.execute(req)
    assert isinstance(result, DataDriftResult)


def test_data_drift_task_synthetic_data_fallback():
    task = DataDriftTask()
    req = DataDriftRequest(reference_data_path="/nonexistent/ref.parquet", current_data_path="/nonexistent/cur.parquet")
    result = task.execute(req)
    assert result.status == "completed"
    assert len(result.feature_psi_scores) > 0


def test_data_drift_psi_scores_nonnegative():
    task = DataDriftTask()
    req = DataDriftRequest(reference_data_path="/nonexistent/ref.parquet", current_data_path="/nonexistent/cur.parquet")
    result = task.execute(req)
    for score in result.feature_psi_scores.values():
        assert score >= 0.0


def test_data_drift_drifted_features_is_list():
    task = DataDriftTask()
    req = DataDriftRequest(reference_data_path="/nonexistent/ref.parquet", current_data_path="/nonexistent/cur.parquet")
    result = task.execute(req)
    assert isinstance(result.drifted_features, list)


# ── ConceptDriftTask ───────────────────────────────────────────────────────


def test_concept_drift_task_returns_dto():
    task = ConceptDriftTask()
    req = ConceptDriftRequest(predictions_log_path="/nonexistent/preds.parquet")
    result = task.execute(req)
    assert isinstance(result, ConceptDriftResult)


def test_concept_drift_synthetic_fallback():
    task = ConceptDriftTask()
    req = ConceptDriftRequest(predictions_log_path="/nonexistent/preds.parquet")
    result = task.execute(req)
    assert result.status == "completed"
    assert result.window_size > 0


def test_concept_drift_relative_change_nonnegative():
    task = ConceptDriftTask()
    req = ConceptDriftRequest(predictions_log_path="/nonexistent/preds.parquet")
    result = task.execute(req)
    assert result.relative_change >= 0.0


# ── DriftDetectionService ──────────────────────────────────────────────────


def test_drift_service_has_data_drift_method():
    service = DriftDetectionService(DataDriftTask(), ConceptDriftTask())
    assert hasattr(service, "detect_data_drift")


def test_drift_service_has_concept_drift_method():
    service = DriftDetectionService(DataDriftTask(), ConceptDriftTask())
    assert hasattr(service, "detect_concept_drift")


# ── Facade ─────────────────────────────────────────────────────────────────


def test_facade_has_submit_data_drift():
    from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
    assert hasattr(FoundationsServiceFacade, "submit_data_drift")


def test_facade_has_submit_concept_drift():
    from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
    assert hasattr(FoundationsServiceFacade, "submit_concept_drift")


# ── Admin Router ───────────────────────────────────────────────────────────


def test_admin_router_has_drift_data_endpoint():
    from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
    assert hasattr(FoundationsAdminRouter, "drift_data")

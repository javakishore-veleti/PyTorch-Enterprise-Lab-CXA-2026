"""Tests for monitoring tasks and service — Week 12."""
from __future__ import annotations
import json
import pytest
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    PrometheusMetricsRequest, PrometheusMetricsResult,
    AuditLogRequest, AuditLogResult,
)
from quantedge_services.core.jobs import JobRegistry
from quantedge_services.services.wfs.monitoring.tasks.prometheus_metrics_task import PrometheusMetricsTask
from quantedge_services.services.wfs.monitoring.tasks.audit_log_task import AuditLogTask
from quantedge_services.services.wfs.monitoring.monitoring_service import MonitoringService


# ── PrometheusMetricsTask ──────────────────────────────────────────────────


def test_prometheus_metrics_task_creates_file(tmp_path):
    registry = JobRegistry()
    task = PrometheusMetricsTask(job_registry=registry)
    out = str(tmp_path / "metrics.json")
    req = PrometheusMetricsRequest(output_path=out)
    result = task.execute(req)
    assert (tmp_path / "metrics.json").exists()


def test_prometheus_metrics_result_fields(tmp_path):
    registry = JobRegistry()
    task = PrometheusMetricsTask(job_registry=registry)
    out = str(tmp_path / "metrics.json")
    req = PrometheusMetricsRequest(output_path=out)
    result = task.execute(req)
    assert isinstance(result, PrometheusMetricsResult)
    assert result.status == "completed"


def test_prometheus_metrics_total_jobs_nonnegative(tmp_path):
    registry = JobRegistry()
    task = PrometheusMetricsTask(job_registry=registry)
    out = str(tmp_path / "metrics.json")
    req = PrometheusMetricsRequest(output_path=out)
    result = task.execute(req)
    assert result.total_jobs >= 0
    assert result.completed_jobs >= 0
    assert result.failed_jobs >= 0


# ── AuditLogTask ───────────────────────────────────────────────────────────


def test_audit_log_task_creates_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = AuditLogTask()
    req = AuditLogRequest(event_type="model_deployed", actor="test-service", resource="model-v1")
    result = task.execute(req)
    log_file = tmp_path / "data" / "audit" / "audit_log.jsonl"
    assert log_file.exists()


def test_audit_log_task_result_has_log_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = AuditLogTask()
    req = AuditLogRequest(event_type="drift_detected", actor="drift-service", resource="feature_1")
    result = task.execute(req)
    assert isinstance(result, AuditLogResult)
    assert len(result.log_id) > 0
    assert result.status == "completed"


def test_audit_log_task_appends_multiple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = AuditLogTask()
    for i in range(3):
        req = AuditLogRequest(event_type=f"event_{i}", actor="svc", resource=f"res_{i}")
        task.execute(req)
    log_file = tmp_path / "data" / "audit" / "audit_log.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        assert json.loads(line)  # valid JSON


# ── MonitoringService ──────────────────────────────────────────────────────


def test_monitoring_service_has_methods():
    registry = JobRegistry()
    service = MonitoringService(PrometheusMetricsTask(job_registry=registry), AuditLogTask())
    assert hasattr(service, "collect_metrics")
    assert hasattr(service, "log_audit_event")


# ── Facade ─────────────────────────────────────────────────────────────────


def test_facade_has_submit_prometheus_metrics():
    from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
    assert hasattr(FoundationsServiceFacade, "submit_prometheus_metrics")


def test_facade_has_submit_audit_log():
    from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
    assert hasattr(FoundationsServiceFacade, "submit_audit_log")


# ── Admin Router ───────────────────────────────────────────────────────────


def test_admin_router_has_monitoring_metrics_endpoint():
    from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
    assert hasattr(FoundationsAdminRouter, "monitoring_metrics")

"""MonitoringService — async wrapper around Prometheus metrics and audit logging."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    PrometheusMetricsRequest, PrometheusMetricsResult,
    AuditLogRequest, AuditLogResult,
)
from quantedge_services.services.wfs.monitoring.tasks.prometheus_metrics_task import PrometheusMetricsTask
from quantedge_services.services.wfs.monitoring.tasks.audit_log_task import AuditLogTask


class MonitoringService:
    """Orchestrates Prometheus metrics collection and audit logging."""

    def __init__(self, metrics_task: PrometheusMetricsTask, audit_task: AuditLogTask) -> None:
        self._metrics = metrics_task
        self._audit = audit_task

    async def collect_metrics(self, request: PrometheusMetricsRequest) -> PrometheusMetricsResult:
        return self._metrics.execute(request)

    async def log_audit_event(self, request: AuditLogRequest) -> AuditLogResult:
        return self._audit.execute(request)

    def prometheus_text(self) -> str:
        return self._metrics.prometheus_text()

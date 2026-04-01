"""PrometheusMetricsTask — snapshot JobRegistry stats to JSON + Prometheus text format."""
from __future__ import annotations
import json
from pathlib import Path
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    PrometheusMetricsRequest, PrometheusMetricsResult,
)
from quantedge_services.core.jobs import JobRegistry, JobStatus


class PrometheusMetricsTask:
    """Reads JobRegistry stats and writes metrics snapshot."""

    def __init__(self, job_registry: JobRegistry) -> None:
        self._registry = job_registry

    def execute(self, request: PrometheusMetricsRequest) -> PrometheusMetricsResult:
        jobs = self._registry.list_all()
        total = len(jobs)
        completed = sum(1 for j in jobs if j.status == JobStatus.SUCCESS)
        failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
        pending = sum(1 for j in jobs if j.status == JobStatus.PENDING)

        latencies: list[float] = []
        for j in jobs:
            if j.started_at and j.completed_at:
                ms = (j.completed_at - j.started_at).total_seconds() * 1000
                latencies.append(ms)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        metrics = {
            "total_jobs": total,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "pending_jobs": pending,
            "avg_latency_ms": round(avg_latency, 3),
        }

        out_path = Path(request.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))

        return PrometheusMetricsResult(
            total_jobs=total,
            completed_jobs=completed,
            failed_jobs=failed,
            avg_latency_ms=round(avg_latency, 3),
            metrics_path=str(out_path),
            status="completed",
        )

    def prometheus_text(self) -> str:
        """Generate Prometheus exposition text format."""
        jobs = self._registry.list_all()
        completed = sum(1 for j in jobs if j.status == JobStatus.SUCCESS)
        failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
        pending = sum(1 for j in jobs if j.status == JobStatus.PENDING)
        running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)

        lines = [
            "# HELP quantedge_jobs_total Total jobs submitted",
            "# TYPE quantedge_jobs_total counter",
            f'quantedge_jobs_total{{status="completed"}} {completed}',
            f'quantedge_jobs_total{{status="failed"}} {failed}',
            f'quantedge_jobs_total{{status="pending"}} {pending}',
            f'quantedge_jobs_total{{status="running"}} {running}',
        ]
        return "\n".join(lines) + "\n"

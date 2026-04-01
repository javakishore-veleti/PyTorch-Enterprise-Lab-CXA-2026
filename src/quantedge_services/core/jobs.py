"""JobRegistry — tracks background workflow job state (thread-safe in-memory store)."""
from __future__ import annotations
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"


@dataclass
class JobRecord:
    id:           str
    task_name:    str
    status:       JobStatus            = JobStatus.PENDING
    submitted_at: datetime             = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at:   datetime | None     = None
    completed_at: datetime | None     = None
    result:       dict[str, Any] | None = None
    error:        str | None           = None


class JobRegistry:
    """Thread-safe in-memory job store. Singleton via DependencyContainer."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, task_name: str) -> JobRecord:
        job = JobRecord(id=str(uuid.uuid4()), task_name=task_name)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def set_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status     = JobStatus.RUNNING
            self._jobs[job_id].started_at = datetime.now(timezone.utc)

    def set_success(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id].status       = JobStatus.SUCCESS
            self._jobs[job_id].completed_at = datetime.now(timezone.utc)
            self._jobs[job_id].result       = result

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            self._jobs[job_id].status       = JobStatus.FAILED
            self._jobs[job_id].completed_at = datetime.now(timezone.utc)
            self._jobs[job_id].error        = error

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list_all(
        self,
        task_name: str | None = None,
        status: "JobStatus | None" = None,
    ) -> list[JobRecord]:
        """Return all jobs, newest first. Optionally filter by task_name and/or status."""
        with self._lock:
            jobs = list(self._jobs.values())
        if task_name:
            jobs = [j for j in jobs if j.task_name == task_name]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.submitted_at, reverse=True)

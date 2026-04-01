"""Job lifecycle DTOs — returned by async 202 endpoints."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel
from quantedge_services.core.jobs import JobStatus


class JobSubmittedResponse(BaseModel):
    job_id:    str
    task_name: str
    status:    Literal["pending"] = "pending"
    message:   str = "Job queued. Poll /jobs/{job_id} for status."


class JobStatusResponse(BaseModel):
    job_id:       str
    task_name:    str
    status:       JobStatus
    submitted_at: datetime
    started_at:   datetime | None      = None
    completed_at: datetime | None      = None
    result:       dict[str, Any] | None = None
    error:        str | None            = None


class JobListResponse(BaseModel):
    jobs:  list[JobStatusResponse]
    total: int

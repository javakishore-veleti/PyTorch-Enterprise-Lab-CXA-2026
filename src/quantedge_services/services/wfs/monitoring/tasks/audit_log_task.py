"""AuditLogTask — structured JSON audit entries appended to JSONL file."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    AuditLogRequest, AuditLogResult,
)


class AuditLogTask:
    """Appends structured audit log entries to a JSONL file."""

    def execute(self, request: AuditLogRequest) -> AuditLogResult:
        log_id = str(uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        log_path = Path("data/audit/audit_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "event_type": request.event_type,
            "actor": request.actor,
            "resource": request.resource,
            "metadata": request.metadata,
            "severity": request.severity,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

        return AuditLogResult(
            log_id=log_id,
            timestamp=timestamp,
            event_type=request.event_type,
            severity=request.severity,
            log_path=str(log_path),
            status="completed",
        )

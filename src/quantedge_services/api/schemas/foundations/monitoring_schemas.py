"""Monitoring, Drift Detection, and ADR DTOs."""
from __future__ import annotations
from pydantic import BaseModel


class DataDriftRequest(BaseModel):
    reference_data_path: str
    current_data_path: str
    feature_columns: list[str] = []
    psi_threshold: float = 0.2
    ks_threshold: float = 0.05


class DataDriftResult(BaseModel):
    feature_psi_scores: dict[str, float]
    feature_ks_pvalues: dict[str, float]
    drifted_features: list[str]
    drift_detected: bool
    status: str


class ConceptDriftRequest(BaseModel):
    predictions_log_path: str
    window_size: int = 100
    drift_threshold: float = 0.1


class ConceptDriftResult(BaseModel):
    baseline_mean: float
    current_mean: float
    relative_change: float
    drift_detected: bool
    window_size: int
    status: str


class PrometheusMetricsRequest(BaseModel):
    output_path: str
    include_job_stats: bool = True
    include_model_stats: bool = True


class PrometheusMetricsResult(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_latency_ms: float
    metrics_path: str
    status: str


class AuditLogRequest(BaseModel):
    event_type: str
    actor: str
    resource: str
    metadata: dict[str, str] = {}
    severity: str = "INFO"


class AuditLogResult(BaseModel):
    log_id: str
    timestamp: str
    event_type: str
    severity: str
    log_path: str
    status: str


class ADRGenerateRequest(BaseModel):
    output_dir: str = "docs/adr"
    adr_ids: list[str] = []


class ADRGenerateResult(BaseModel):
    generated_files: list[str]
    adr_count: int
    output_dir: str
    status: str


class ADRListRequest(BaseModel):
    adr_dir: str = "docs/adr"


class ADRListResult(BaseModel):
    adrs: list[dict]
    total_count: int
    status: str

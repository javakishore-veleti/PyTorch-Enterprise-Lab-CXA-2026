"""Tracking & Canary deployment DTOs for Week 11."""
from __future__ import annotations
from pydantic import BaseModel


class MLflowLogRequest(BaseModel):
    experiment_name: str
    run_name: str
    params: dict[str, str] = {}
    metrics: dict[str, float] = {}
    artifact_paths: list[str] = []
    tags: dict[str, str] = {}

class MLflowLogResult(BaseModel):
    run_id: str
    experiment_id: str
    artifact_uri: str
    status: str

class MLflowRegisterRequest(BaseModel):
    run_id: str
    model_name: str
    artifact_path: str = "model"
    stage: str = "Staging"

class MLflowRegisterResult(BaseModel):
    model_name: str
    version: str
    stage: str
    status: str

class CanaryDeployRequest(BaseModel):
    baseline_model_path: str
    candidate_model_path: str
    canary_traffic_pct: float = 10.0
    deployment_id: str
    input_size: int = 14
    seq_len: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    state_dir: str = "data/canary"

class CanaryDeployResult(BaseModel):
    deployment_id: str
    canary_traffic_pct: float
    baseline_routed: int
    candidate_routed: int
    status: str

class CanaryEvalRequest(BaseModel):
    deployment_id: str
    num_eval_requests: int = 100
    success_threshold: float = 0.95
    input_size: int = 14
    seq_len: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    state_dir: str = "data/canary"

class CanaryEvalResult(BaseModel):
    deployment_id: str
    baseline_mean_loss: float
    candidate_mean_loss: float
    promotion_decision: str  # "promote", "rollback", "inconclusive"
    status: str

class ModelRegistryListRequest(BaseModel):
    filter_name: str = ""
    max_results: int = 20

class ModelRegistryListResult(BaseModel):
    models: list[dict]
    total_count: int
    status: str

class ModelRegistryPromoteRequest(BaseModel):
    model_name: str
    version: str
    target_stage: str  # "Staging", "Production", "Archived"

class ModelRegistryPromoteResult(BaseModel):
    model_name: str
    version: str
    previous_stage: str
    new_stage: str
    status: str

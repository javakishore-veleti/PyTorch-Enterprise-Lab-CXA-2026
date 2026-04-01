from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class AttentionExtractRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    sequences: list[list[list[float]]]
    seq_len: int = 30
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0


class AttentionWeightsLayer(BaseModel):
    layer_idx: int
    weights: list[list[list[list[float]]]]


class AttentionExtractResponse(BaseModel):
    execution_id: str
    num_layers: int
    num_heads: int
    seq_len: int
    layers: list[AttentionWeightsLayer]
    status: str
    error: Optional[str] = None


class AttentionHeatmapRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    sequences: list[list[list[float]]]
    seq_len: int = 30
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    output_dir: str = "data/attention_viz"
    colormap: str = "viridis"
    figure_dpi: int = 100
    sensor_labels: list[str] = []


class AttentionHeatmapResponse(BaseModel):
    execution_id: str
    output_dir: str
    files_created: list[str]
    num_layers: int
    num_heads: int
    status: str
    error: Optional[str] = None


class ArchDecisionRequest(BaseModel):
    execution_id: str
    task_type: str = "time_series_regression"
    seq_len: int = 30
    input_size: int = 14
    output_type: str = "scalar"


class ArchDecisionOption(BaseModel):
    architecture: str
    suitability_score: int
    pros: list[str]
    cons: list[str]
    recommended_when: str


class ArchDecisionResponse(BaseModel):
    execution_id: str
    task_type: str
    recommended_architecture: str
    rationale: str
    options: list[ArchDecisionOption]
    status: str
    error: Optional[str] = None

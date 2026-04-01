"""Domain Adaptation & Ollama Serving DTOs — Week 8."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── StackOverflow ──────────────────────────────────────────────────────────────

class StackOverflowDownloadRequest(BaseModel):
    dataset_source: str = "kaggle"
    output_dir: str = "data/stackoverflow"


class StackOverflowDownloadResult(BaseModel):
    output_path: str
    record_count: int
    status: str
    error: Optional[str] = None


class StackOverflowIngestionRequest(BaseModel):
    input_path: str
    output_dir: str
    tags_filter: list[str] = ["java", "python"]


class StackOverflowIngestionResult(BaseModel):
    output_path: str
    filtered_count: int
    status: str
    error: Optional[str] = None


# ── Domain Adaptation ─────────────────────────────────────────────────────────

class DomainAdaptTrainRequest(BaseModel):
    data_path: str
    model_name: str = "gpt2"
    output_dir: str = "data/domain_adapt/checkpoints"
    lora_rank: int = 16
    lora_alpha: int = 32
    max_steps: int = 100
    batch_size: int = 4


class DomainAdaptTrainResult(BaseModel):
    checkpoint_path: str
    train_loss: float
    steps: int
    status: str
    error: Optional[str] = None


class DomainAdaptEvalRequest(BaseModel):
    data_path: str
    checkpoint_path: str
    model_name: str = "gpt2"
    lora_rank: int = 16
    lora_alpha: int = 32


class DomainAdaptEvalResult(BaseModel):
    eval_loss: float
    perplexity: float
    status: str
    error: Optional[str] = None


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaInferRequest(BaseModel):
    model_name: str = "llama3"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    ollama_base_url: str = "http://localhost:11434"


class OllamaInferResult(BaseModel):
    response: str
    status: str
    latency_ms: float
    error: Optional[str] = None


class OllamaMergeRequest(BaseModel):
    adapter_checkpoint_path: str
    base_model_name: str = "gpt2"
    output_dir: str = "data/ollama/merged"
    ollama_model_name: str = "quantedge-adapted"


class OllamaMergeResult(BaseModel):
    merged_model_path: str
    modelfile_path: str
    status: str
    error: Optional[str] = None

from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class ProfilerRunRequest(BaseModel):
    execution_id: str
    parquet_path: str = "data/forex/processed/train.parquet"
    model_type: str = "mlp"
    checkpoint_path: str = ""
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 5
    trace_dir: str = "data/profiler/traces"
    with_stack: bool = False


class ProfilerRunResponse(BaseModel):
    execution_id: str
    trace_path: str
    steps_profiled: int
    avg_step_ms: float
    peak_memory_mb: float
    status: str
    error: Optional[str] = None


class MemorySummaryRequest(BaseModel):
    execution_id: str
    parquet_path: str = "data/forex/processed/train.parquet"
    model_type: str = "mlp"
    batch_size: int = 256


class MemorySummaryResponse(BaseModel):
    execution_id: str
    device: str
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    summary_text: str
    status: str
    error: Optional[str] = None


class DataloaderTuneRequest(BaseModel):
    execution_id: str
    parquet_path: str = "data/forex/processed/train.parquet"
    batch_size: int = 256
    num_workers_sweep: list[int] = [0, 1, 2, 4]
    pin_memory: bool = False


class DataloaderTuneResponse(BaseModel):
    execution_id: str
    best_num_workers: int
    timing_results: dict[str, float]
    speedup_vs_single: float
    status: str
    error: Optional[str] = None


class CICIoTDownloadRequest(BaseModel):
    execution_id: str
    kaggle_dataset: str = "dhoogla/ciciot2023"
    dest_dir: str = "data/cic_iot/raw"
    force_redownload: bool = False


class CICIoTDownloadResponse(BaseModel):
    execution_id: str
    dest_dir: str
    files_downloaded: int
    status: str
    error: Optional[str] = None


class CICIoTIngestionRequest(BaseModel):
    execution_id: str
    raw_dir: str = "data/cic_iot/raw"
    parquet_dir: str = "data/cic_iot/parquet"
    nrows: Optional[int] = None
    label_col: str = "label"


class CICIoTIngestionResponse(BaseModel):
    execution_id: str
    rows_loaded: int
    parquet_path: str
    num_classes: int
    class_distribution: dict[str, int]
    status: str
    error: Optional[str] = None

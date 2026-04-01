from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class OAsst1DownloadRequest(BaseModel):
    execution_id: str
    hf_repo_id: str = "OpenAssistant/oasst1"
    dest_dir: str = "data/oasst1/raw"
    force_redownload: bool = False


class OAsst1DownloadResponse(BaseModel):
    execution_id: str
    dest_dir: str
    files_downloaded: int
    status: str
    error: Optional[str] = None


class OAsst1IngestionRequest(BaseModel):
    execution_id: str
    raw_dir: str = "data/oasst1/raw"
    parquet_dir: str = "data/oasst1/parquet"
    lang: str = "en"
    min_text_len: int = 20
    max_rows: Optional[int] = None


class OAsst1IngestionResponse(BaseModel):
    execution_id: str
    rows_ingested: int
    parquet_path: str
    assistant_turns: int
    human_turns: int
    status: str
    error: Optional[str] = None


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: list[str] = ["w_q", "w_k", "w_v", "w_o"]


class LoRATrainRequest(BaseModel):
    execution_id: str
    base_checkpoint_path: str
    parquet_path: str = "data/cmapss/parquet/FD001_seq30.parquet"
    seq_len: int = 30
    input_size: int = 14
    lora_config: LoRAConfig = LoRAConfig()
    epochs: int = 10
    learning_rate: float = 0.0005
    batch_size: int = 128
    patience: int = 5
    checkpoint_dir: str = "data/lora/checkpoints"


class LoRATrainResponse(BaseModel):
    execution_id: str
    epochs_trained: int
    best_val_loss: float
    lora_checkpoint_path: str
    trainable_params: int
    frozen_params: int
    trainable_ratio: float
    status: str
    error: Optional[str] = None


class LoRAEvalRequest(BaseModel):
    execution_id: str
    base_checkpoint_path: str
    lora_checkpoint_path: str
    parquet_path: str = "data/cmapss/parquet/FD001_seq30.parquet"
    seq_len: int = 30
    input_size: int = 14
    lora_config: LoRAConfig = LoRAConfig()


class LoRAEvalResponse(BaseModel):
    execution_id: str
    rmse: float
    mae: float
    score: float
    trainable_params: int
    status: str
    error: Optional[str] = None


class LoRAPredictRequest(BaseModel):
    execution_id: str
    base_checkpoint_path: str
    lora_checkpoint_path: str
    sequences: list[list[list[float]]]
    seq_len: int = 30
    input_size: int = 14
    lora_config: LoRAConfig = LoRAConfig()


class LoRAPredictResponse(BaseModel):
    execution_id: str
    rul_predictions: list[float]
    status: str
    error: Optional[str] = None


class LoRAMergeRequest(BaseModel):
    execution_id: str
    base_checkpoint_path: str
    lora_checkpoint_path: str
    merged_checkpoint_path: str = "data/lora/merged"
    seq_len: int = 30
    input_size: int = 14
    lora_config: LoRAConfig = LoRAConfig()


class LoRAMergeResponse(BaseModel):
    execution_id: str
    merged_checkpoint_path: str
    total_params: int
    status: str
    error: Optional[str] = None

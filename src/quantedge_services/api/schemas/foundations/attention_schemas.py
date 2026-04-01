from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class CMAPSSDownloadRequest(BaseModel):
    execution_id: str
    kaggle_dataset: str = "behrad3d/nasa-cmaps"
    dest_dir: str = "data/cmapss/raw"
    force_redownload: bool = False


class CMAPSSDownloadResponse(BaseModel):
    execution_id: str
    dest_dir: str
    files_downloaded: int
    status: str
    error: Optional[str] = None


class CMAPSSIngestionRequest(BaseModel):
    execution_id: str
    raw_dir: str = "data/cmapss/raw"
    parquet_dir: str = "data/cmapss/parquet"
    subset: str = "FD001"
    seq_len: int = 30


class CMAPSSIngestionResponse(BaseModel):
    execution_id: str
    sequences_created: int
    parquet_path: str
    num_sensors: int
    max_rul: int
    status: str
    error: Optional[str] = None


class AttentionTrainRequest(BaseModel):
    execution_id: str
    parquet_path: str = "data/cmapss/parquet/FD001_seq30.parquet"
    seq_len: int = 30
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 128
    patience: int = 5
    checkpoint_dir: str = "data/cmapss/checkpoints"


class AttentionTrainResponse(BaseModel):
    execution_id: str
    epochs_trained: int
    best_val_loss: float
    checkpoint_path: str
    status: str
    error: Optional[str] = None


class AttentionEvalRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    parquet_path: str = "data/cmapss/parquet/FD001_seq30.parquet"
    seq_len: int = 30
    input_size: int = 14


class AttentionEvalResponse(BaseModel):
    execution_id: str
    rmse: float
    mae: float
    score: float
    status: str
    error: Optional[str] = None


class AttentionPredictRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    sequences: list[list[list[float]]]


class AttentionPredictResponse(BaseModel):
    execution_id: str
    rul_predictions: list[float]
    status: str
    error: Optional[str] = None

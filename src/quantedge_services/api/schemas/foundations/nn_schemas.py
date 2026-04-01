"""DTOs for Week 3 Neural Network workflows on Forex data."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NNTrainRequest(BaseModel):
    execution_id: str
    model_type: Literal["mlp", "lstm"] = "mlp"
    parquet_path: str = "data/forex/processed/train.parquet"
    hidden_sizes: list[int] = [128, 64, 32]
    dropout_rate: float = 0.3
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 256
    patience: int = 5


class NNTrainResponse(BaseModel):
    execution_id: str
    model_type: str
    epochs_trained: int
    best_val_loss: float
    checkpoint_path: str
    status: str
    error: str | None = None


class NNEvalRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    parquet_path: str = "data/forex/processed/val.parquet"


class NNEvalResponse(BaseModel):
    execution_id: str
    mse: float
    mae: float
    rmse: float
    direction_accuracy: float
    status: str
    error: str | None = None


class NNPredictRequest(BaseModel):
    execution_id: str
    checkpoint_path: str
    input_features: list[list[float]]


class NNPredictResponse(BaseModel):
    execution_id: str
    predictions: list[float]
    status: str
    error: str | None = None

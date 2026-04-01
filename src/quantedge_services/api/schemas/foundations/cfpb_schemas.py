"""DTOs for CFPB Complaints workflow — request and response contracts."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class CFPBDownloadRequest(BaseModel):
    destination_dir: str = Field(default="data/cfpb/raw", description="Directory for raw HuggingFace parquet files")
    hf_dataset_id: str = Field(default="cfpb/consumer-finance-complaints")
    split: str = Field(default="train")


class CFPBDownloadResponse(BaseModel):
    execution_id: str
    destination_dir: str
    files_downloaded: int
    total_size_mb: float
    parquet_path: str
    status: Literal["success", "failed", "skipped"]
    message: str = ""
    error: str | None = None


class CFPBIngestionRequest(BaseModel):
    parquet_dir: str = Field(default="data/cfpb/raw", description="Directory containing downloaded HuggingFace parquet")
    output_parquet_dir: str = Field(default="data/cfpb/processed")
    narrative_col: str = Field(default="Consumer complaint narrative")
    product_col: str = Field(default="Product")
    min_narrative_len: int = Field(default=50, description="Drop narratives shorter than this")


class CFPBIngestionResponse(BaseModel):
    execution_id: str
    rows_raw: int
    rows_after_filter: int
    parquet_path: str
    status: Literal["success", "failed"]
    error: str | None = None


class CFPBPreprocessRequest(BaseModel):
    execution_id: str
    model_name: str = Field(default="distilbert-base-uncased")
    max_length: int = Field(default=256, ge=32, le=512)
    batch_size: int = Field(default=1000, ge=1)


class CFPBPreprocessResponse(BaseModel):
    execution_id: str
    rows_after_filter: int
    n_classes: int
    class_weights: dict[str, float]
    status: Literal["success", "failed"]
    error: str | None = None


class CFPBDatasetRequest(BaseModel):
    execution_id: str
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)


class CFPBDatasetResponse(BaseModel):
    execution_id: str
    train_samples: int
    val_samples: int
    train_batches: int
    val_batches: int
    status: Literal["success", "failed"]
    error: str | None = None


class CFPBTrainRequest(BaseModel):
    execution_id: str
    model_name: str = Field(default="distilbert-base-uncased")
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    seed: int = Field(default=42)
    checkpoint_dir: str = Field(default="data/checkpoints/cfpb")
    resume_from: str | None = None


class CFPBTrainResponse(BaseModel):
    execution_id: str
    epochs_completed: int
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    checkpoint_path: str
    status: Literal["success", "failed"]
    error: str | None = None


class CFPBPredictRequest(BaseModel):
    execution_id: str
    text: str = Field(min_length=10, description="Raw complaint narrative text")
    checkpoint_path: str


class CFPBPredictResponse(BaseModel):
    execution_id: str
    predicted_product: str
    confidence: float
    status: Literal["success", "failed"]
    error: str | None = None

"""DTOs for Forex EUR/USD workflow — request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ForexDownloadRequest(BaseModel):
    destination_dir: str = Field(
        default="data/forex/raw",
        description="Directory to download raw HistData CSV files into",
    )
    kaggle_dataset: str = Field(
        default="kianindeed/eurusd-histdata-tick-data",
        description="Kaggle dataset slug (owner/dataset)",
    )
    use_huggingface: bool = Field(
        default=False,
        description="Pull from HuggingFace instead of Kaggle",
    )
    hf_dataset_id: str = Field(
        default="nlp-creator/eurusd-historical-forex",
        description="HuggingFace dataset repository ID",
    )


class ForexDownloadResponse(BaseModel):
    execution_id: str
    destination_dir: str
    files_downloaded: int
    total_size_mb: float
    status: Literal["success", "failed", "skipped"]
    message: str = ""
    error: str | None = None


class ForexIngestionRequest(BaseModel):
    data_dir: str = Field(
        default="data/forex/raw",
        description="Directory containing raw HistData CSV files",
    )
    parquet_dir: str = Field(
        default="data/forex/parquet",
        description="Directory to persist resampled OHLCV parquet files",
    )
    resample_freq: str = Field(
        default="1min",
        description="Pandas offset alias for OHLCV bar resampling (e.g. '1min', '5min', '1h')",
    )
    years: list[int] = Field(
        default_factory=list,
        description="Filter to specific years; empty = all found",
    )
    nrows: int | None = Field(
        default=None,
        description="Cap tick rows per file for dev/testing",
    )


class ForexIngestionResponse(BaseModel):
    execution_id: str
    ticks_loaded: int
    bars_resampled: int
    files_loaded: int
    parquet_path: str
    status: Literal["success", "failed"]
    error: str | None = None


class ForexPreprocessRequest(BaseModel):
    execution_id: str
    fill_gaps: bool = True
    normalize: bool = True
    scaler_type: Literal["minmax", "zscore"] = "minmax"
    train_ratio: float = Field(default=0.70, ge=0.05, le=0.95)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.50)


class ForexPreprocessResponse(BaseModel):
    execution_id: str
    total_bars: int
    train_bars: int
    val_bars: int
    test_bars: int
    nan_filled: int
    features: list[str]
    processed_parquet_path: str
    status: Literal["success", "failed"]
    error: str | None = None


class ForexAutogradRequest(BaseModel):
    execution_id: str
    window_size: int = Field(default=20, ge=2, description="Rolling window for tensor ops")


class ForexAutogradResponse(BaseModel):
    execution_id: str
    manual_loss: float
    autograd_loss: float
    max_grad_diff: float
    status: Literal["success", "failed"]
    error: str | None = None


class ForexTensorOpsRequest(BaseModel):
    execution_id: str
    volatility_window: int = Field(default=20, ge=2)
    momentum_window: int = Field(default=14, ge=2)
    inject_nan_fraction: float = Field(default=0.01, ge=0.0, le=0.5)


class ForexTensorOpsResponse(BaseModel):
    execution_id: str
    volatility_points: int
    momentum_points: int
    nan_injected: int
    nan_remaining: int
    status: Literal["success", "failed"]
    error: str | None = None

"""DTOs for Forex EUR/USD workflow — request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ForexIngestionRequest(BaseModel):
    data_dir: str = Field(default="data/forex", description="Path to HistData CSV files")
    years: list[int] = Field(default_factory=list, description="Filter to specific years; empty = all")
    nrows: int | None = Field(default=None, description="Cap rows per file (dev/testing only)")


class ForexIngestionResponse(BaseModel):
    execution_id: str
    rows_loaded: int
    files_loaded: int
    status: Literal["success", "failed"]
    error: str | None = None


class ForexPreprocessRequest(BaseModel):
    execution_id: str
    fill_gaps: bool = True
    normalize: bool = True


class ForexPreprocessResponse(BaseModel):
    execution_id: str
    rows_after: int
    nan_filled: int
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

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from quantedge_services.api.schemas.foundations.attention_schemas import (
    AttentionEvalRequest, AttentionEvalResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class AttentionEvalTask:
    """Loads checkpoint + parquet, runs inference, computes RMSE/MAE/NASA-score."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: AttentionEvalRequest) -> AttentionEvalResponse:
        parquet_path = Path(request.parquet_path)
        if not parquet_path.exists():
            return AttentionEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0,
                mae=0.0,
                score=0.0,
                status="skipped",
                error=f"Parquet not found: {parquet_path}",
            )

        checkpoint_path = Path(request.checkpoint_path)
        if not checkpoint_path.exists():
            return AttentionEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0,
                mae=0.0,
                score=0.0,
                status="skipped",
                error=f"Checkpoint not found: {checkpoint_path}",
            )

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = ForexTransformer(
                input_size=ckpt["input_size"],
                d_model=ckpt["d_model"],
                nhead=ckpt["nhead"],
                num_encoder_layers=ckpt["num_encoder_layers"],
                dim_feedforward=ckpt["dim_feedforward"],
                dropout=ckpt.get("dropout", 0.1),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            df = pd.read_parquet(parquet_path)
            feature_cols = [c for c in df.columns if c.startswith("f")]
            X_raw = df[feature_cols].values.astype(np.float32)
            y_true = df["rul"].values.astype(np.float32)
            X = X_raw.reshape(-1, request.seq_len, request.input_size)

            with torch.no_grad():
                x_tensor = torch.from_numpy(X)
                pred_tensor, _ = model(x_tensor)
                y_pred = pred_tensor.squeeze(-1).numpy()

            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae = float(np.mean(np.abs(y_pred - y_true)))
            score = self._cmaps_score(y_pred, y_true)

            return AttentionEvalResponse(
                execution_id=request.execution_id,
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                score=round(score, 4),
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("attention_eval_failed", error=str(exc))
            return AttentionEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0,
                mae=0.0,
                score=0.0,
                status="failed",
                error=str(exc),
            )

    @staticmethod
    def _cmaps_score(pred: np.ndarray, true: np.ndarray) -> float:
        diff = pred - true
        return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))

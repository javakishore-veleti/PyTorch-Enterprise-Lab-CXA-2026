from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRAEvalRequest, LoRAEvalResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.services.wfs.lora_finetuning.models.lora_transformer import LoRATransformer


class LoRAEvalTask:
    """Loads base + LoRA checkpoint, runs inference, computes RMSE/MAE/NASA score."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: LoRAEvalRequest) -> LoRAEvalResponse:
        base_ckpt_path = Path(request.base_checkpoint_path)
        lora_ckpt_path = Path(request.lora_checkpoint_path)

        if not base_ckpt_path.exists():
            return LoRAEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0, mae=0.0, score=0.0, trainable_params=0,
                status="skipped",
                error=f"Base checkpoint not found: {base_ckpt_path}",
            )
        if not lora_ckpt_path.exists():
            return LoRAEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0, mae=0.0, score=0.0, trainable_params=0,
                status="skipped",
                error=f"LoRA checkpoint not found: {lora_ckpt_path}",
            )
        parquet_path = Path(request.parquet_path)
        if not parquet_path.exists():
            return LoRAEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0, mae=0.0, score=0.0, trainable_params=0,
                status="skipped",
                error=f"Parquet not found: {parquet_path}",
            )

        try:
            base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            base_model = ForexTransformer(
                input_size=base_ckpt.get("input_size", request.input_size),
                d_model=base_ckpt["d_model"],
                nhead=base_ckpt["nhead"],
                num_encoder_layers=base_ckpt["num_encoder_layers"],
                dim_feedforward=base_ckpt["dim_feedforward"],
                dropout=base_ckpt.get("dropout", 0.1),
            )
            base_model.load_state_dict(base_ckpt["model_state_dict"])

            cfg = request.lora_config
            model = LoRATransformer(
                base_model=base_model,
                rank=cfg.rank,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
                target_modules=cfg.target_modules,
            )

            lora_ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
            model.load_lora_state_dict(lora_ckpt["lora_state_dict"])
            model.eval()

            param_counts = model.count_parameters()
            trainable_params = param_counts["trainable"]

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

            return LoRAEvalResponse(
                execution_id=request.execution_id,
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                score=round(score, 4),
                trainable_params=trainable_params,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("lora_eval_failed", error=str(exc))
            return LoRAEvalResponse(
                execution_id=request.execution_id,
                rmse=0.0, mae=0.0, score=0.0, trainable_params=0,
                status="failed",
                error=str(exc),
            )

    @staticmethod
    def _cmaps_score(pred: np.ndarray, true: np.ndarray) -> float:
        diff = pred - true
        return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))

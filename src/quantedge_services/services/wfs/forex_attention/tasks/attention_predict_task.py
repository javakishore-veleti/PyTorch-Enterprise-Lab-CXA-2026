from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from quantedge_services.api.schemas.foundations.attention_schemas import (
    AttentionPredictRequest, AttentionPredictResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class AttentionPredictTask:
    """Loads checkpoint, runs inference on provided sequences."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: AttentionPredictRequest) -> AttentionPredictResponse:
        checkpoint_path = Path(request.checkpoint_path)
        if not checkpoint_path.exists():
            return AttentionPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=[],
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

            x = torch.tensor(request.sequences, dtype=torch.float32)
            with torch.no_grad():
                pred, _ = model(x)
            predictions = pred.squeeze(-1).tolist()
            if isinstance(predictions, float):
                predictions = [predictions]

            return AttentionPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=predictions,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("attention_predict_failed", error=str(exc))
            return AttentionPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=[],
                status="failed",
                error=str(exc),
            )

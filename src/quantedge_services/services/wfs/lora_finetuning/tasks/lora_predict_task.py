from __future__ import annotations
from pathlib import Path
import torch
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRAPredictRequest, LoRAPredictResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.services.wfs.lora_finetuning.models.lora_transformer import LoRATransformer


class LoRAPredictTask:
    """Loads base + LoRA checkpoint, runs inference on provided sequences."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: LoRAPredictRequest) -> LoRAPredictResponse:
        base_ckpt_path = Path(request.base_checkpoint_path)
        lora_ckpt_path = Path(request.lora_checkpoint_path)

        if not base_ckpt_path.exists():
            return LoRAPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=[],
                status="skipped",
                error=f"Base checkpoint not found: {base_ckpt_path}",
            )
        if not lora_ckpt_path.exists():
            return LoRAPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=[],
                status="skipped",
                error=f"LoRA checkpoint not found: {lora_ckpt_path}",
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

            x = torch.tensor(request.sequences, dtype=torch.float32)
            with torch.no_grad():
                pred, _ = model(x)
            predictions = pred.squeeze(-1).tolist()
            if isinstance(predictions, float):
                predictions = [predictions]

            return LoRAPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=predictions,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("lora_predict_failed", error=str(exc))
            return LoRAPredictResponse(
                execution_id=request.execution_id,
                rul_predictions=[],
                status="failed",
                error=str(exc),
            )

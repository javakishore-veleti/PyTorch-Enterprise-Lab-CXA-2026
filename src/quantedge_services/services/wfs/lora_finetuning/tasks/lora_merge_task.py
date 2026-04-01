from __future__ import annotations
from pathlib import Path
import torch
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRAMergeRequest, LoRAMergeResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.services.wfs.lora_finetuning.models.lora_transformer import LoRATransformer


class LoRAMergeTask:
    """Loads base + LoRA checkpoint, merges weights, saves merged ForexTransformer."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: LoRAMergeRequest) -> LoRAMergeResponse:
        base_ckpt_path = Path(request.base_checkpoint_path)
        lora_ckpt_path = Path(request.lora_checkpoint_path)

        if not base_ckpt_path.exists():
            return LoRAMergeResponse(
                execution_id=request.execution_id,
                merged_checkpoint_path="",
                total_params=0,
                status="skipped",
                error=f"Base checkpoint not found: {base_ckpt_path}",
            )
        if not lora_ckpt_path.exists():
            return LoRAMergeResponse(
                execution_id=request.execution_id,
                merged_checkpoint_path="",
                total_params=0,
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

            merged_model = model.merge_and_export()
            total_params = sum(p.numel() for p in merged_model.parameters())

            out_dir = Path(request.merged_checkpoint_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            merged_path = out_dir / f"{request.execution_id}_merged.pt"

            torch.save(
                {
                    "input_size": base_ckpt.get("input_size", request.input_size),
                    "d_model": base_ckpt["d_model"],
                    "nhead": base_ckpt["nhead"],
                    "num_encoder_layers": base_ckpt["num_encoder_layers"],
                    "dim_feedforward": base_ckpt["dim_feedforward"],
                    "dropout": base_ckpt.get("dropout", 0.1),
                    "seq_len": request.seq_len,
                    "model_state_dict": merged_model.state_dict(),
                },
                merged_path,
            )

            return LoRAMergeResponse(
                execution_id=request.execution_id,
                merged_checkpoint_path=str(merged_path),
                total_params=total_params,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("lora_merge_failed", error=str(exc))
            return LoRAMergeResponse(
                execution_id=request.execution_id,
                merged_checkpoint_path="",
                total_params=0,
                status="failed",
                error=str(exc),
            )

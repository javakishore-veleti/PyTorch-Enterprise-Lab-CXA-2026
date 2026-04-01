from __future__ import annotations
from pathlib import Path
import torch
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionExtractRequest, AttentionExtractResponse, AttentionWeightsLayer,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.core.logging import StructuredLogger


class AttentionExtractTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: AttentionExtractRequest) -> AttentionExtractResponse:
        ckpt_path = Path(request.checkpoint_path)
        if not ckpt_path.exists():
            return AttentionExtractResponse(
                execution_id=request.execution_id,
                num_layers=0,
                num_heads=0,
                seq_len=request.seq_len,
                layers=[],
                status="skipped",
                error=f"Checkpoint not found: {ckpt_path}",
            )
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            input_size = ckpt.get("input_size", request.input_size)
            d_model = ckpt.get("d_model", request.d_model)
            nhead = ckpt.get("nhead", request.nhead)
            num_encoder_layers = ckpt.get("num_encoder_layers", request.num_encoder_layers)
            dim_feedforward = ckpt.get("dim_feedforward", request.dim_feedforward)
            dropout = 0.0  # deterministic inference

            model = ForexTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            x = torch.tensor(request.sequences, dtype=torch.float32)
            with torch.no_grad():
                _, attn_weights_list = model(x)

            layers = [
                AttentionWeightsLayer(
                    layer_idx=i,
                    weights=w.tolist(),
                )
                for i, w in enumerate(attn_weights_list)
            ]

            return AttentionExtractResponse(
                execution_id=request.execution_id,
                num_layers=len(layers),
                num_heads=nhead,
                seq_len=x.shape[1],
                layers=layers,
                status="success",
            )
        except Exception as exc:
            self._logger.error("attention_extract_failed", error=str(exc))
            return AttentionExtractResponse(
                execution_id=request.execution_id,
                num_layers=0,
                num_heads=0,
                seq_len=request.seq_len,
                layers=[],
                status="failed",
                error=str(exc),
            )

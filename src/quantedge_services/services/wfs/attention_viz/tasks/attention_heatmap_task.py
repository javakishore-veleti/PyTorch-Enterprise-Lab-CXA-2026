from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionHeatmapRequest, AttentionHeatmapResponse,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.core.logging import StructuredLogger


class AttentionHeatmapTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: AttentionHeatmapRequest) -> AttentionHeatmapResponse:
        ckpt_path = Path(request.checkpoint_path)
        if not ckpt_path.exists():
            return AttentionHeatmapResponse(
                execution_id=request.execution_id,
                output_dir=request.output_dir,
                files_created=[],
                num_layers=0,
                num_heads=0,
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

            model = ForexTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Use only first sequence, batch=1
            first_seq = [request.sequences[0]]
            x = torch.tensor(first_seq, dtype=torch.float32)

            with torch.no_grad():
                _, attn_weights_list = model(x)

            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            files_created: list[str] = []
            for l_idx, w in enumerate(attn_weights_list):
                # w shape: [1, nhead, seq_len, seq_len]
                w_np = w[0].cpu().numpy()  # [nhead, seq_len, seq_len]
                for h_idx in range(nhead):
                    weights = w_np[h_idx]  # [seq_len, seq_len]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(weights, cmap=request.colormap, aspect='auto', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, label='Attention weight')
                    ax.set_xlabel('Key timestep')
                    ax.set_ylabel('Query timestep')
                    ax.set_title(f'Layer {l_idx + 1} · Head {h_idx + 1}')
                    if request.sensor_labels:
                        ax.set_xticks(range(len(request.sensor_labels)))
                        ax.set_xticklabels(request.sensor_labels, rotation=90)
                        ax.set_yticks(range(len(request.sensor_labels)))
                        ax.set_yticklabels(request.sensor_labels)
                    png_path = output_dir / f"layer{l_idx + 1:02d}_head{h_idx + 1:02d}.png"
                    fig.savefig(str(png_path), dpi=request.figure_dpi, bbox_inches='tight')
                    plt.close(fig)
                    files_created.append(str(png_path))

            return AttentionHeatmapResponse(
                execution_id=request.execution_id,
                output_dir=str(output_dir),
                files_created=files_created,
                num_layers=len(attn_weights_list),
                num_heads=nhead,
                status="success",
            )
        except Exception as exc:
            self._logger.error("attention_heatmap_failed", error=str(exc))
            return AttentionHeatmapResponse(
                execution_id=request.execution_id,
                output_dir=request.output_dir,
                files_created=[],
                num_layers=0,
                num_heads=0,
                status="failed",
                error=str(exc),
            )

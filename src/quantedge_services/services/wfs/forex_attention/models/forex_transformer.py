from __future__ import annotations
import torch
import torch.nn as nn
from quantedge_services.services.wfs.forex_attention.models.attention_components import (
    PositionalEncoding,
    TransformerEncoderBlock,
)


class ForexTransformer(nn.Module):
    """Transformer encoder for time-series regression (RUL / forex forecasting).

    Input:  [batch, seq_len, input_size]
    Output: ([batch, 1], list[attn_weights per block])
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len=512, dropout=dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        attn_weights_list: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x, w = block(x, mask)
            attn_weights_list.append(w)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x), attn_weights_list

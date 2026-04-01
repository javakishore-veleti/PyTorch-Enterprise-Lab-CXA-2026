"""ForexLSTM — stacked 2-layer LSTM with FC head for Forex regression."""
from __future__ import annotations

import torch
import torch.nn as nn


class ForexLSTM(nn.Module):
    """Stacked LSTM with FC head. Input: [batch, seq_len, input_size]."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # take the last time step
        last = out[:, -1, :]
        return self.fc(self.dropout(last))

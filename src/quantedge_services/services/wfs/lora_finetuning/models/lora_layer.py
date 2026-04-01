from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear layer.

    Forward: W x + (B A x) * (alpha/rank)

    Where:
    - W = frozen base weight (requires_grad=False)
    - A = trainable [rank, in_features], initialized with kaiming_uniform
    - B = trainable [out_features, rank], initialized with zeros
    - scaling = alpha / rank

    The base Linear's weight is frozen; only A and B are updated during training.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.base_weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.base_bias = (
            nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
            if base_linear.bias is not None
            else None
        )

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.base_weight, self.base_bias)
        lora_out = nn.functional.linear(self.dropout(x), self.lora_A)
        lora_out = nn.functional.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Return a standard nn.Linear with LoRA weights merged into base weight."""
        merged_weight = self.base_weight + (self.lora_B @ self.lora_A) * self.scaling
        out = nn.Linear(
            self.lora_A.size(1),
            self.lora_B.size(0),
            bias=self.base_bias is not None,
        )
        out.weight.data = merged_weight.clone()
        if self.base_bias is not None:
            out.bias.data = self.base_bias.clone()
        return out

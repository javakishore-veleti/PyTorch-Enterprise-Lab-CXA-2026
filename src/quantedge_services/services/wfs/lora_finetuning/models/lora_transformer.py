from __future__ import annotations
import copy
import torch
import torch.nn as nn
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.services.wfs.lora_finetuning.models.lora_layer import LoRALinear


class LoRATransformer(nn.Module):
    """Wraps a pre-trained ForexTransformer, replacing target Linear modules with LoRALinear.

    Steps in __init__:
    1. Deep-copy the base ForexTransformer
    2. Freeze ALL parameters of the copy
    3. For each encoder block, replace target_modules (w_q, w_k, w_v, w_o) in MultiHeadAttention
       with LoRALinear wrappers

    Only LoRA A and B matrices have requires_grad=True.
    """

    def __init__(
        self,
        base_model: ForexTransformer,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        if target_modules is None:
            target_modules = ["w_q", "w_k", "w_v", "w_o"]

        self.model = copy.deepcopy(base_model)

        for param in self.model.parameters():
            param.requires_grad = False

        for block in self.model.encoder_blocks:
            attn = block.attn
            for module_name in target_modules:
                if hasattr(attn, module_name):
                    base_linear = getattr(attn, module_name)
                    if isinstance(base_linear, nn.Linear):
                        setattr(
                            attn,
                            module_name,
                            LoRALinear(base_linear, rank=rank, alpha=alpha, dropout=dropout),
                        )

    def forward(self, x: torch.Tensor, mask=None):
        return self.model(x, mask)

    def count_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def get_lora_state_dict(self) -> dict:
        """Return only LoRA adapter weights for efficient checkpoint saving."""
        return {k: v for k, v in self.state_dict().items() if "lora_A" in k or "lora_B" in k}

    def load_lora_state_dict(self, lora_state: dict) -> None:
        """Load LoRA adapter weights into model."""
        current = self.state_dict()
        current.update(lora_state)
        self.load_state_dict(current)

    def merge_and_export(self) -> ForexTransformer:
        """Merge all LoRA weights into base weights and return a standard ForexTransformer."""
        merged = copy.deepcopy(self.model)
        for block in merged.encoder_blocks:
            attn = block.attn
            for module_name in ["w_q", "w_k", "w_v", "w_o"]:
                layer = getattr(attn, module_name, None)
                if isinstance(layer, LoRALinear):
                    setattr(attn, module_name, layer.merge_weights())
        return merged

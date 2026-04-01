"""PEFTModelWrapper — LoRA-adapter wrapper around a HuggingFace causal LM."""
from __future__ import annotations
from typing import TYPE_CHECKING

try:
    import torch.nn as nn
    from transformers import AutoModelForCausalLM  # type: ignore
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

if TYPE_CHECKING:
    import torch.nn as nn


class PEFTModelWrapper:
    """Wraps a HuggingFace causal LM with a PEFT LoRA adapter.

    Raises ``RuntimeError`` at construction time if the ``peft`` package is
    not installed so callers receive a clear error instead of an AttributeError.
    """

    def __init__(
        self,
        base_model_name: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: list[str],
    ) -> None:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft not installed")

        self._base_model_name = base_model_name
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._target_modules = target_modules

        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(base, lora_cfg)

    def get_trainable_param_count(self) -> int:
        """Returns the number of trainable (adapter) parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def merge_and_unload(self) -> "nn.Module":
        """Merges LoRA weights into the base model and returns a plain nn.Module."""
        return self.model.merge_and_unload()

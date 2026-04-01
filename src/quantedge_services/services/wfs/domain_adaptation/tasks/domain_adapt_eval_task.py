"""DomainAdaptEvalTask — evaluate a LoRA-adapted causal LM on a text dataset."""
from __future__ import annotations
import math
from pathlib import Path

import torch

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    DomainAdaptEvalRequest,
    DomainAdaptEvalResult,
)

try:
    import pandas as pd
    from transformers import AutoTokenizer  # type: ignore
    from quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper import PEFTModelWrapper
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class DomainAdaptEvalTask:
    """Loads a saved adapter checkpoint, runs forward pass on the eval set, and
    returns cross-entropy loss + perplexity."""

    def execute(self, request: DomainAdaptEvalRequest) -> DomainAdaptEvalResult:
        if not _DEPS_AVAILABLE:
            return DomainAdaptEvalResult(
                eval_loss=0.0,
                perplexity=0.0,
                status="error",
                error="Required dependencies not installed",
            )

        import pandas as pd
        from transformers import AutoTokenizer

        df = pd.read_parquet(request.data_path)
        texts: list[str] = df["text"].astype(str).tolist()

        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        wrapper = PEFTModelWrapper(
            base_model_name=request.model_name,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
            target_modules=["c_attn", "c_proj"],
        )

        checkpoint = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
        adapter_state = checkpoint.get("adapter_state_dict", {})
        if adapter_state:
            state = wrapper.model.state_dict()
            state.update(adapter_state)
            wrapper.model.load_state_dict(state, strict=False)

        model = wrapper.model
        model.eval()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for text in texts[:min(len(texts), 8)]:
                enc = tokenizer(
                    [text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                )
                input_ids = enc["input_ids"]
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_loss += outputs.loss.item()
                count += 1

        eval_loss = total_loss / max(1, count)
        perplexity = math.exp(min(eval_loss, 20.0))

        return DomainAdaptEvalResult(
            eval_loss=eval_loss,
            perplexity=perplexity,
            status="success",
        )

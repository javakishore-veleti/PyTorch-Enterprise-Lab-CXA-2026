"""DomainAdaptTrainTask — fine-tunes a causal LM with LoRA via PEFT."""
from __future__ import annotations
import math
from pathlib import Path

import torch

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    DomainAdaptTrainRequest,
    DomainAdaptTrainResult,
)

try:
    import pandas as pd
    from transformers import AutoTokenizer  # type: ignore
    from quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper import PEFTModelWrapper
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class DomainAdaptTrainTask:
    """Loads tokenizer + PEFTModelWrapper, trains for *max_steps* with AdamW,
    and saves an adapter checkpoint."""

    def execute(self, request: DomainAdaptTrainRequest) -> DomainAdaptTrainResult:
        if not _DEPS_AVAILABLE:
            return DomainAdaptTrainResult(
                checkpoint_path="",
                train_loss=0.0,
                steps=0,
                status="error",
                error="Required dependencies (transformers/peft/pandas) not installed",
            )

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        model = wrapper.model
        model.train()

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=2e-4,
        )

        total_loss = 0.0
        steps_done = 0
        batch_size = request.batch_size

        for step in range(request.max_steps):
            idx = step % max(1, (len(texts) // batch_size))
            batch_texts = texts[idx * batch_size : (idx + 1) * batch_size] or texts[:1]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss: torch.Tensor = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps_done += 1

        avg_loss = total_loss / max(1, steps_done)

        checkpoint_path = output_dir / "adapter_checkpoint.pt"
        torch.save(
            {
                "adapter_state_dict": {
                    k: v for k, v in model.state_dict().items() if "lora" in k
                },
                "config": {
                    "model_name": request.model_name,
                    "lora_rank": request.lora_rank,
                    "lora_alpha": request.lora_alpha,
                },
                "train_loss": avg_loss,
                "steps": steps_done,
            },
            checkpoint_path,
        )

        return DomainAdaptTrainResult(
            checkpoint_path=str(checkpoint_path),
            train_loss=avg_loss,
            steps=steps_done,
            status="success",
        )

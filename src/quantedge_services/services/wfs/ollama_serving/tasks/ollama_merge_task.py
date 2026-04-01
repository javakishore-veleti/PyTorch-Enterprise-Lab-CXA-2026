"""OllamaMergeTask — merge a LoRA adapter into its base model and write an Ollama Modelfile."""
from __future__ import annotations
from pathlib import Path

import torch

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    OllamaMergeRequest,
    OllamaMergeResult,
)

try:
    from quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper import PEFTModelWrapper
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

_MODELFILE_TEMPLATE = "FROM ./merged_model\nPARAMETER temperature 0.7\n"


class OllamaMergeTask:
    """Loads adapter checkpoint, merges weights via ``PEFTModelWrapper.merge_and_unload``,
    saves the merged model state dict, and writes an Ollama Modelfile."""

    def execute(self, request: OllamaMergeRequest) -> OllamaMergeResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_model_path = output_dir / "merged_model"
        merged_model_path.mkdir(parents=True, exist_ok=True)
        modelfile_path = output_dir / "ollama_modelfile.txt"

        if not _PEFT_AVAILABLE:
            modelfile_path.write_text(_MODELFILE_TEMPLATE)
            return OllamaMergeResult(
                merged_model_path=str(merged_model_path),
                modelfile_path=str(modelfile_path),
                status="error",
                error="peft not installed",
            )

        checkpoint = torch.load(
            request.adapter_checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint.get("config", {})
        lora_rank = config.get("lora_rank", 16)
        lora_alpha = config.get("lora_alpha", 32)

        wrapper = PEFTModelWrapper(
            base_model_name=request.base_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj"],
        )

        adapter_state = checkpoint.get("adapter_state_dict", {})
        if adapter_state:
            state = wrapper.model.state_dict()
            state.update(adapter_state)
            wrapper.model.load_state_dict(state, strict=False)

        merged = wrapper.merge_and_unload()
        torch.save({"model_state_dict": merged.state_dict()}, merged_model_path / "pytorch_model.bin")

        modelfile_path.write_text(_MODELFILE_TEMPLATE)

        return OllamaMergeResult(
            merged_model_path=str(merged_model_path),
            modelfile_path=str(modelfile_path),
            status="success",
        )

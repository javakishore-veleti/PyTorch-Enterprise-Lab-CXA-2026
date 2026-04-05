"""Week 07 — LoRA & QLoRA.

Modules:
    quantized_loader   — Load Mistral-7B in 4-bit with BitsAndBytes
    lora_config        — Configure and apply LoRA adapters
    dataset_prep       — Prepare OpenAssistant oasst1 dataset
    qlora_trainer      — Train with QLoRA using HF Trainer
    memory_comparison  — Compare QLoRA vs full fine-tune memory
"""

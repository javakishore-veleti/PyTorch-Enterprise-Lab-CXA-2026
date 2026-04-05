# Hands-On Log — Week 07: LoRA & QLoRA

**Package:** `src/quantedge_services_handson/week07/`
**Goal:** QLoRA fine-tune on RTX 5080 — adapter weights saved independently

---

## Step 1 — Load Model in 4-bit
**File:** `week07/quantized_loader.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `quantized_loader.py`
- [ ] Implement `load_quantized_model(model_name) -> (model, tokenizer)`
- [ ] BitsAndBytesConfig: load_in_4bit, nf4, float16 compute, double quant

### Checkpoint
- Memory footprint: _____ GB (should be ~4-5 GB, not 14+)

### Notes


---

## Step 2 — Configure LoRA Adapters
**File:** `week07/lora_config.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `lora_config.py`
- [ ] Implement `apply_lora(model, rank=16) -> (model, lora_config)`
- [ ] prepare_model_for_kbit_training → LoraConfig → get_peft_model
- [ ] target_modules: q_proj, k_proj, v_proj, o_proj

### Checkpoint
- Trainable params: _____ M (should be ~0.1-0.5% of total)
- `model.print_trainable_parameters()` output: _____

### Notes


---

## Step 3 — Prepare Dataset
**File:** `week07/dataset_prep.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `dataset_prep.py`
- [ ] Implement `prepare_oasst1_dataset(tokenizer, max_length=512)`
- [ ] Load OpenAssistant/oasst1, filter English, format as `[INST]...[/INST]`
- [ ] Tokenize with max_length, truncation, padding

### Checkpoint
- Dataset size: _____ samples
- Token length: 512 per sample

### Notes


---

## Step 4 — Train with QLoRA
**File:** `week07/qlora_trainer.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `qlora_trainer.py`
- [ ] Implement `train_qlora(model, tokenizer, train_dataset, eval_dataset)`
- [ ] TrainingArguments: 3 epochs, batch=4, grad_accum=4, lr=2e-4, fp16
- [ ] Save adapter to `checkpoints/week07/adapter`

### Checkpoint
- Final training loss: _____
- Adapter size on disk: _____ MB (should be ~25 MB, not 14 GB)

### Notes


---

## Step 5 — Compare Memory: QLoRA vs Full
**File:** `week07/memory_comparison.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `memory_comparison.py`
- [ ] Implement `compare_memory_usage()`
- [ ] Load full fp16 → record memory, load QLoRA → record memory

### Checkpoint

| Mode      | GPU Memory | Trainable Params |
|-----------|-----------|------------------|
| Full fp16 |     GB    |                  |
| QLoRA     |     GB    |                  |

### Notes


---

## Week 07 Summary
**All Steps Passed:** `No`
**Memory Savings:** ___x
**Key Learnings:**

**Commit SHA:**

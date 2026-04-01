# PyTorch Mastery Plan — Chief AI Architect Edition

> 12 Weeks · RTX 5080 (Ubuntu) · HuggingFace + Kaggle Datasets · End-to-end from Tensors to MLOps

---

## Overall Progress

| Phase | Weeks | Status |
|---|---|---|
| 🟡 Foundations | 1–2 | `⬜ Not Started` |
| 🔵 GPU Mastery | 3–4 | `⬜ Not Started` |
| 🟣 Transformer | 5–6 | `⬜ Not Started` |
| 🟠 Fine-Tuning | 7–8 | `⬜ Not Started` |
| 🟢 Production | 9–10 | `⬜ Not Started` |
| 🔴 MLOps | 11–12 | `⬜ Not Started` |

> **Status key:** `⬜ Not Started` · `🔄 In Progress` · `✅ Done` · `🚫 Blocked`

---

## Phase 1 — Foundations (Weeks 1–2)

### Week 1 · Tensors & Autograd

| | |
|---|---|
| **Topics** | Tensors, autograd, computation graph, device management (CPU↔GPU) |
| **Libraries** | `torch`, `numpy` |
| **Dataset** | Automobile Telematics · Kaggle · ~300 MB |
| **Input** | CSV: speed, RPM, throttle, acceleration columns |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Write manual backward pass
- [ ] Compare with autograd result
- [ ] Draw computation graph on paper
- [ ] Intentionally cause NaN — then fix it

**Outcome**

> Understand what `.backward()` does in memory — not just that it works

---

### Week 2 · Training Loop & DataLoader

| | |
|---|---|
| **Topics** | Custom Dataset & DataLoader, training loop, loss & metrics, checkpointing, reproducibility |
| **Libraries** | `torch.utils.data`, `datasets` (HF) |
| **Dataset** | OpenAssistant oasst1 · HuggingFace · ~1 GB |
| **Input** | Conversation JSON tokenized into `input_ids` sequences |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Build `Dataset` class from scratch
- [ ] Add `pin_memory` + `num_workers`
- [ ] Save & resume from checkpoint
- [ ] Set all seeds — confirm same loss across runs
- [ ] Add eval loop with accuracy metric

**Outcome**

> Clean training loop that trains, saves, resumes — fully reproducible on your RTX 5080

---

## Phase 2 — GPU Mastery (Weeks 3–4)

### Week 3 · Mixed Precision & OOM Debugging

| | |
|---|---|
| **Topics** | Mixed precision (AMP), GradScaler, gradient accumulation, OOM debugging patterns |
| **Libraries** | `torch.cuda.amp`, `accelerate` |
| **Dataset** | CIC IoT 2023 · Kaggle · ~10 GB |
| **Input** | Network traffic CSVs from 105 real IoT devices |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Baseline training speed without AMP
- [ ] Add `torch.autocast` + `GradScaler`
- [ ] Inflate batch size until OOM
- [ ] Fix OOM with gradient accumulation
- [ ] Log throughput before vs after

**Outcome**

> Same model running 2x faster — proven with timing logs, not guesswork

---

### Week 4 · Profiling & Bottleneck Identification

| | |
|---|---|
| **Topics** | `torch.profiler`, `memory_summary`, dataloader tuning, compute vs I/O bottleneck identification |
| **Libraries** | `torch.profiler`, `tensorboard` |
| **Dataset** | CIC IoT 2023 · Kaggle · ~10 GB |
| **Input** | Same IoT traffic data — now stress-tested at scale |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Run profiler for 5 steps
- [ ] Open TensorBoard trace viewer
- [ ] Read `memory_summary()` output
- [ ] Tune `num_workers` until GPU never waits
- [ ] Document: was bottleneck CPU/IO or GPU compute?

**Outcome**

> Profiler report showing real bottleneck — identified, documented, and fixed

---

## Phase 3 — Transformer (Weeks 5–6)

### Week 5 · Attention from Scratch

| | |
|---|---|
| **Topics** | Multi-head attention from scratch, positional encoding, layer norm, feed-forward blocks |
| **Libraries** | `torch.nn`, `einops` |
| **Dataset** | NASA CMAPSS Engine Degradation · Kaggle · ~50 MB |
| **Input** | 21-channel sensor time-series with RUL (remaining useful life) labels |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Implement scaled dot-product attention using raw `matmul`
- [ ] Add multi-head split/merge
- [ ] Add sinusoidal positional encoding
- [ ] Stack 4 transformer blocks with pre-norm
- [ ] **No `nn.Transformer` allowed**

**Outcome**

> Transformer encoder built entirely from scratch — every line understood

---

### Week 6 · Architecture Decisions & Attention Visualization

| | |
|---|---|
| **Topics** | Encoder vs decoder vs encoder-decoder architecture decisions, attention weight visualization |
| **Libraries** | `transformers`, `matplotlib` |
| **Dataset** | NASA CMAPSS Engine Degradation · Kaggle · ~50 MB |
| **Input** | Same sensor sequences — with attention weight extraction added |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Train model to convergence
- [ ] Extract attention weights from each head
- [ ] Plot heatmap over time-steps and sensor channels
- [ ] Identify which sensors model attends to most
- [ ] Write 1-page architecture decision rationale

**Outcome**

> Trained transformer with visualized attention — every architectural choice explainable in an interview

---

## Phase 4 — Fine-Tuning (Weeks 7–8)

### Week 7 · LoRA & QLoRA

| | |
|---|---|
| **Topics** | Full fine-tuning vs LoRA vs QLoRA, adapter layers, rank selection, target module choice |
| **Libraries** | `peft`, `bitsandbytes`, `transformers` |
| **Dataset** | OpenAssistant oasst1 · HuggingFace · ~1 GB |
| **Input** | Instruction-response pairs formatted in ShareGPT / Alpaca style |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Load Mistral-7B in 4-bit with `BitsAndBytesConfig`
- [ ] Configure `LoraConfig` with `rank=16`
- [ ] Freeze base weights — verify only adapter params train
- [ ] Compare GPU memory: QLoRA vs full fine-tune
- [ ] Log training loss curve

**Outcome**

> QLoRA fine-tune running on RTX 5080 — adapter weights saved and loadable independently

---

### Week 8 · Domain Adaptation & Ollama Serving

| | |
|---|---|
| **Topics** | Inference with loaded adapters, weight merging, Ollama serving, prompt template formatting |
| **Libraries** | `peft`, `ollama`, `accelerate` |
| **Dataset** | StackOverflow Java/ES dump · Kaggle · ~2 GB (filtered) |
| **Input** | Java + Elasticsearch + Spring Boot Q&A pairs after domain filtering |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Filter StackOverflow to your tech domain only
- [ ] Fine-tune on filtered Q&A pairs
- [ ] Merge LoRA adapter into base model
- [ ] Package and run via Ollama
- [ ] Test with 10 real enterprise tech questions

**Outcome**

> Domain-adapted Mistral-7B running locally — answers Spring Boot and Elasticsearch questions correctly

---

## Phase 5 — Production (Weeks 9–10)

### Week 9 · Export Formats (TorchScript · ONNX · TensorRT)

| | |
|---|---|
| **Topics** | TorchScript vs ONNX vs TensorRT — tradeoffs, `torch.onnx.export`, dynamic axes, version risks |
| **Libraries** | `torch.jit`, `onnx`, `optimum` (HF) |
| **Dataset** | NASA CMAPSS Engine Degradation · Kaggle · ~50 MB |
| **Input** | Trained sensor transformer from Week 6 as starting point |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Export via `torch.jit.script`
- [ ] Export same model via `torch.onnx.export` with dynamic batch axis
- [ ] Benchmark inference latency: TorchScript vs ONNX
- [ ] Document when to choose each path
- [ ] Test `onnxruntime` vs `torch` inference output parity

**Outcome**

> Same model exported three ways — latency table with decision guide for edge vs cloud

---

### Week 10 · Quantization, FastAPI Serving & Docker

| | |
|---|---|
| **Topics** | INT8 quantization, `onnxruntime-gpu` inference, FastAPI serving, input validation, timeouts, structured logging |
| **Libraries** | `onnxruntime-gpu`, `fastapi`, `pydantic`, `uvicorn` |
| **Dataset** | Any prior dataset — focus is the serving pipeline, not new data |
| **Input** | ONNX model from Week 9 |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Quantize ONNX model to INT8
- [ ] Measure accuracy delta before/after quantization
- [ ] Build FastAPI endpoint with Pydantic input validation
- [ ] Add timeout middleware (`asyncio`)
- [ ] Add structured JSON logging with correlation ID
- [ ] Dockerize and run locally

**Outcome**

> Production inference service — Dockerized, P50/P99 latency documented, error handling solid

---

## Phase 6 — MLOps (Weeks 11–12)

### Week 11 · Experiment Tracking & Canary Deployment

| | |
|---|---|
| **Topics** | Model versioning, experiment tracking with MLflow, canary deployment pattern, rollback strategy |
| **Libraries** | `mlflow`, `huggingface_hub` |
| **Dataset** | HuggingFace Hub — model cards as governance artifacts |
| **Input** | All models trained in prior weeks as versioning subjects |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Log all prior experiments to MLflow with params + metrics
- [ ] Register models with version tags
- [ ] Write a model card (intended use, limitations, eval results)
- [ ] Define rollback trigger criteria: `latency > Xms` OR `accuracy drop > Y%`
- [ ] Sketch canary deployment pattern for model swap

**Outcome**

> MLflow dashboard with all 10 weeks of experiments tracked — rollback runbook written and defensible

---

### Week 12 · Drift Detection, Monitoring & ADR

| | |
|---|---|
| **Topics** | Data drift detection, GPU utilization monitoring in serving, cost per inference, Architecture Decision Record |
| **Libraries** | `evidently`, `prometheus`, `grafana` |
| **Dataset** | Kaggle competition — one end-to-end notebook of your choice |
| **Input** | Any Kaggle competition dataset — picked for relevance to your target interview role |
| **Status** | `⬜ Not Started` |

**Intermediate Tasks**

- [ ] Add Evidently drift detector to FastAPI serving pipeline
- [ ] Hook GPU utilization metrics to Prometheus
- [ ] Build minimal Grafana dashboard (latency + GPU + drift)
- [ ] Calculate: cost per 1,000 inferences on RTX 5080
- [ ] Write one full Architecture Decision Record (ADR)
- [ ] Publish Kaggle notebook publicly

**Outcome**

> Full system: train → version → deploy → monitor — documented as ADR. Public Kaggle notebook as portfolio proof.

---

## Quick Reference — Libraries by Phase

| Phase | Key Libraries |
|---|---|
| Foundations | `torch` `numpy` `torch.utils.data` `datasets` |
| GPU Mastery | `torch.cuda.amp` `accelerate` `torch.profiler` `tensorboard` |
| Transformer | `torch.nn` `einops` `transformers` `matplotlib` |
| Fine-Tuning | `peft` `bitsandbytes` `transformers` `ollama` `accelerate` |
| Production | `torch.jit` `onnx` `optimum` `onnxruntime-gpu` `fastapi` `pydantic` `uvicorn` |
| MLOps | `mlflow` `huggingface_hub` `evidently` `prometheus` `grafana` |

## Quick Reference — Datasets

| Dataset | Source | Size | Used in |
|---|---|---|---|
| Automobile Telematics | Kaggle | ~300 MB | Week 1 |
| OpenAssistant oasst1 | HuggingFace | ~1 GB | Weeks 2, 7 |
| CIC IoT 2023 | Kaggle | ~10 GB | Weeks 3, 4 |
| NASA CMAPSS Engine Degradation | Kaggle | ~50 MB | Weeks 5, 6, 9 |
| StackOverflow Java/ES dump | Kaggle | ~2 GB | Week 8 |
| Kaggle competition (your choice) | Kaggle | varies | Week 12 |

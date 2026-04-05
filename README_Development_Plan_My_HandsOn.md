# PyTorch Mastery — Hands-On Lab Guide

> 12 Weeks of self-paced, hands-on PyTorch exercises
> All code lives in `src/quantedge_services_handson/`
> Reference implementation: `src/quantedge_services/` (do NOT peek until you finish each exercise)

---

## How to Use This Guide

1. Each week has **numbered steps** — follow them in order
2. Write your code in the specified file path under `src/quantedge_services_handson/weekNN/`
3. Run the provided test/validation commands after each step
4. Only check the reference implementation (`src/quantedge_services/`) if you are stuck for more than 30 minutes
5. Each step has a **Checkpoint** — a concrete output you should see before moving on

---

## Setup

```bash
# From project root
pip install -e ".[dev]"
```

---

## Module Structure

```
src/quantedge_services_handson/
    __init__.py
    week01/
        __init__.py
        tensors.py
        autograd.py
        nan_handling.py
        rolling_ops.py
        device_mgmt.py
    week02/
        __init__.py
        dataset.py
        iterable_dataset.py
        dataloader.py
        classifier.py
        training_loop.py
        checkpointing.py
    week03/
        __init__.py
        mlp_model.py
        lstm_model.py
        baseline_trainer.py
        amp_trainer.py
        grad_accumulation.py
        benchmark.py
    week04/
        __init__.py
        profiler.py
        memory_analysis.py
        worker_tuning.py
        diagnosis.py
    week05/
        __init__.py
        attention.py
        multi_head.py
        positional_encoding.py
        transformer_block.py
        encoder.py
    week06/
        __init__.py
        cmapss_dataset.py
        trainer.py
        attention_extractor.py
        heatmap_viz.py
        architecture_rationale.py
    week07/
        __init__.py
        quantized_loader.py
        lora_config.py
        dataset_prep.py
        qlora_trainer.py
        memory_comparison.py
    week08/
        __init__.py
        data_filter.py
        domain_finetuner.py
        adapter_merger.py
        ollama_packager.py
        domain_tester.py
    week09/
        __init__.py
        torchscript_export.py
        onnx_export.py
        onnx_inference.py
        benchmark.py
        decision_guide.py
    week10/
        __init__.py
        quantization.py
        serving.py
        middleware.py
        Dockerfile
        load_test.py
    week11/
        __init__.py
        mlflow_setup.py
        experiment_logger.py
        model_registry.py
        model_card.py
        canary_strategy.py
    week12/
        __init__.py
        psi.py
        ks_drift.py
        concept_drift.py
        prometheus_metrics.py
        audit_logger.py
        adr_generator.py
```

---

## Phase 1 — Foundations (Weeks 1-2)

### Week 1 · Tensors & Autograd

**Goal:** Understand `.backward()` in memory on real financial data

**Package:** `src/quantedge_services_handson/week01/`

#### Step 1 — Load Forex CSV into Tensors

```python
# src/quantedge_services_handson/week01/tensors.py

import torch
import numpy as np
import pandas as pd

def load_forex_data(csv_path: str) -> torch.Tensor:
    """
    TODO:
    1. Read CSV with columns: datetime, open, high, low, close, volume
    2. Extract 'close' prices into a numpy array
    3. Convert to a torch.float32 tensor
    4. Return the tensor
    """
    pass
```

**Checkpoint:** `print(tensor.shape, tensor.dtype)` should show `torch.Size([N]) torch.float32`

#### Step 2 — Compute Price Deltas with Autograd

```python
# src/quantedge_services_handson/week01/autograd.py

import torch

def compute_price_deltas(prices: torch.Tensor) -> torch.Tensor:
    """
    TODO:
    1. Set requires_grad=True on the prices tensor
    2. Compute deltas: delta[i] = prices[i+1] - prices[i]
    3. Compute a scalar loss = deltas.pow(2).mean()
    4. Call loss.backward()
    5. Return prices.grad
    """
    pass

def manual_backward(prices: torch.Tensor) -> torch.Tensor:
    """
    TODO:
    1. Compute deltas = prices[1:] - prices[:-1]
    2. Compute loss = deltas.pow(2).mean()
    3. Manually compute d(loss)/d(prices) using chain rule:
       - d(loss)/d(delta_i) = 2 * delta_i / N
       - d(delta_i)/d(prices_i) = -1
       - d(delta_i)/d(prices_{i+1}) = +1
    4. Accumulate gradients into a grad tensor
    5. Return the grad tensor
    """
    pass
```

**Checkpoint:** `torch.allclose(manual_grad, autograd_grad, atol=1e-6)` should be `True`

#### Step 3 — Inject NaN and Fix It

```python
# src/quantedge_services_handson/week01/nan_handling.py

import torch

def inject_nan_and_fix(prices: torch.Tensor) -> torch.Tensor:
    """
    TODO:
    1. Clone the prices tensor
    2. Inject NaN at random positions (simulate missing weekend/holiday ticks)
       prices_dirty[::7] = float('nan')
    3. Verify: assert torch.isnan(prices_dirty).any()
    4. Fix it: replace NaN with forward-fill (use the previous valid value)
    5. Verify: assert not torch.isnan(prices_fixed).any()
    6. Return the fixed tensor
    """
    pass
```

**Checkpoint:** Print count of NaNs before and after — should go from N > 0 to 0

#### Step 4 — Rolling Volatility & Momentum

```python
# src/quantedge_services_handson/week01/rolling_ops.py

import torch

def rolling_volatility(prices: torch.Tensor, window: int = 20) -> torch.Tensor:
    """
    TODO:
    1. Compute log returns: log_ret[i] = log(prices[i+1] / prices[i])
    2. Use unfold to create rolling windows of size `window`
    3. Compute std of each window
    4. Return the volatility tensor
    """
    pass

def rolling_momentum(prices: torch.Tensor, window: int = 10) -> torch.Tensor:
    """
    TODO:
    1. Compute momentum: mom[i] = prices[i] - prices[i - window]
    2. Return the momentum tensor
    """
    pass
```

**Checkpoint:** Both return tensors with valid (non-NaN) values; shapes should be `(len(prices) - window,)`

#### Step 5 — Device Management

```python
# src/quantedge_services_handson/week01/device_mgmt.py

import torch

def demonstrate_device_transfer(prices: torch.Tensor) -> dict:
    """
    TODO:
    1. Check if CUDA is available
    2. If yes, move tensor to GPU: prices_gpu = prices.to('cuda')
    3. Perform computation on GPU (e.g., rolling mean)
    4. Move result back to CPU
    5. Return dict with keys: 'device_available', 'cpu_time_ms', 'gpu_time_ms'
       (use torch.cuda.Event for GPU timing)
    """
    pass
```

**Checkpoint:** Print the timing dict — GPU should be faster for large tensors

#### Validation

```bash
python -c "
from quantedge_services_handson.week01.tensors import load_forex_data
from quantedge_services_handson.week01.autograd import compute_price_deltas, manual_backward
from quantedge_services_handson.week01.nan_handling import inject_nan_and_fix
from quantedge_services_handson.week01.rolling_ops import rolling_volatility, rolling_momentum
import torch

prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
grad = compute_price_deltas(prices.clone())
manual = manual_backward(prices.clone())
print('Autograd grad shape:', grad.shape)
print('Manual grad shape:', manual.shape)
print('Match:', torch.allclose(grad, manual, atol=1e-5))

fixed = inject_nan_and_fix(prices.clone())
print('NaN fixed:', not torch.isnan(fixed).any())

vol = rolling_volatility(prices, window=20)
mom = rolling_momentum(prices, window=10)
print('Volatility shape:', vol.shape)
print('Momentum shape:', mom.shape)
print('Week 1 PASSED')
"
```

---

### Week 2 · Training Loop & DataLoader

**Goal:** Build a clean training loop on financial complaint data — fully reproducible

**Package:** `src/quantedge_services_handson/week02/`

#### Step 1 — Custom Map-Style Dataset

```python
# src/quantedge_services_handson/week02/dataset.py

from torch.utils.data import Dataset
import torch

class ComplaintDataset(Dataset):
    """
    TODO:
    1. __init__: Accept a list of complaint texts and product labels
       - Tokenize texts into fixed-length integer sequences (simple char-level or word-index)
       - Convert labels to integer class indices
       - Store as tensors
    2. __len__: Return number of samples
    3. __getitem__: Return (input_tensor, label_tensor) for index i
    """

    def __init__(self, texts: list[str], labels: list[str], max_len: int = 128):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass
```

**Checkpoint:** `dataset[0]` returns `(tensor of shape [128], tensor of shape [])` with integer dtype

#### Step 2 — Iterable-Style Dataset

```python
# src/quantedge_services_handson/week02/iterable_dataset.py

from torch.utils.data import IterableDataset
import torch

class ComplaintIterableDataset(IterableDataset):
    """
    TODO:
    1. __init__: Accept a file path to complaint CSV
    2. __iter__: Yield (input_tensor, label_tensor) one row at a time
       - Read CSV line-by-line (do NOT load entire file into memory)
       - Tokenize and encode on-the-fly
    """

    def __init__(self, csv_path: str, max_len: int = 128):
        pass

    def __iter__(self):
        pass
```

**Checkpoint:** `next(iter(dataset))` returns a valid `(input, label)` tuple

#### Step 3 — DataLoader with pin_memory and num_workers

```python
# src/quantedge_services_handson/week02/dataloader.py

from torch.utils.data import DataLoader

def create_dataloaders(
    train_dataset, val_dataset, batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    """
    TODO:
    1. Create train DataLoader with:
       - shuffle=True
       - num_workers=4  (tune this based on your CPU cores)
       - pin_memory=True (if CUDA available)
       - drop_last=True
    2. Create val DataLoader with:
       - shuffle=False
       - Same num_workers and pin_memory
       - drop_last=False
    3. Return (train_loader, val_loader)
    """
    pass
```

**Checkpoint:** Iterate one batch — `batch_x.shape` should be `[64, 128]`

#### Step 4 — Simple MLP Classifier

```python
# src/quantedge_services_handson/week02/classifier.py

import torch
import torch.nn as nn

class ComplaintClassifier(nn.Module):
    """
    TODO:
    1. __init__: Build a simple MLP:
       - Embedding(vocab_size, embed_dim)
       - Linear(embed_dim, hidden_dim)
       - ReLU
       - Linear(hidden_dim, num_classes)
    2. forward: embed -> mean pool over sequence -> MLP -> logits
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, num_classes: int = 10):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** `model(batch_x).shape` should be `[64, num_classes]`

#### Step 5 — Training Loop with Eval and Checkpointing

```python
# src/quantedge_services_handson/week02/training_loop.py

import torch

def set_all_seeds(seed: int = 42) -> None:
    """
    TODO:
    1. torch.manual_seed(seed)
    2. torch.cuda.manual_seed_all(seed)
    3. np.random.seed(seed)
    4. random.seed(seed)
    5. torch.backends.cudnn.deterministic = True
    6. torch.backends.cudnn.benchmark = False
    """
    pass

def train(
    model, train_loader, val_loader,
    epochs: int = 10, lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints/week02"
) -> dict:
    """
    TODO:
    1. Setup: optimizer (Adam), loss_fn (CrossEntropyLoss), device
    2. Handle class imbalance: compute class weights, pass to CrossEntropyLoss(weight=...)
    3. Training loop per epoch:
       a. model.train()
       b. For each batch: forward -> loss -> backward -> optimizer.step()
       c. Track running loss
    4. Eval loop per epoch:
       a. model.eval()
       b. torch.no_grad()
       c. Compute accuracy = correct / total
    5. Checkpoint: save model.state_dict(), optimizer.state_dict(), epoch, best_val_acc
       - Save best model (highest val accuracy)
       - Save latest model (for resume)
    6. Return dict with 'train_losses', 'val_accuracies', 'best_epoch'
    """
    pass
```

**Checkpoint:** Train for 3 epochs, kill process, resume — loss should continue from where it left off

#### Step 6 — Resume from Checkpoint

```python
# src/quantedge_services_handson/week02/checkpointing.py

import torch

def resume_from_checkpoint(
    model, optimizer, checkpoint_path: str
) -> int:
    """
    TODO:
    1. Load checkpoint dict from file
    2. Load model state dict
    3. Load optimizer state dict
    4. Return the epoch number to resume from
    """
    pass
```

**Checkpoint:** Resume training — loss values continue from saved state

#### Validation

```bash
python -c "
from quantedge_services_handson.week02.dataset import ComplaintDataset
from quantedge_services_handson.week02.dataloader import create_dataloaders
from quantedge_services_handson.week02.classifier import ComplaintClassifier
from quantedge_services_handson.week02.training_loop import set_all_seeds, train
import torch

texts = ['bad service'] * 500 + ['good bank'] * 500
labels = ['complaint'] * 500 + ['praise'] * 500

ds = ComplaintDataset(texts, labels, max_len=32)
print('Dataset size:', len(ds))
print('Sample shape:', ds[0][0].shape, ds[0][1].shape)

train_ds, val_ds = torch.utils.data.random_split(ds, [800, 200])
train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)

set_all_seeds(42)
model = ComplaintClassifier(vocab_size=256, num_classes=2)
result = train(model, train_loader, val_loader, epochs=3)
print('Train losses:', [f'{l:.4f}' for l in result['train_losses']])
print('Val accuracies:', [f'{a:.4f}' for a in result['val_accuracies']])
print('Week 2 PASSED')
"
```

---

## Phase 2 — GPU Mastery (Weeks 3-4)

### Week 3 · Mixed Precision & OOM Debugging

**Goal:** Train IoT anomaly detection model 2x faster with AMP

**Package:** `src/quantedge_services_handson/week03/`

#### Step 1 — IoT Network Traffic MLP

```python
# src/quantedge_services_handson/week03/mlp_model.py

import torch
import torch.nn as nn

class IoTAnomalyMLP(nn.Module):
    """
    TODO:
    1. Build an MLP for IoT traffic classification:
       - Input: 46 network features (packet length, flow duration, etc.)
       - Hidden: 256 -> 128 -> 64 (with BatchNorm + ReLU + Dropout)
       - Output: num_classes (34 IoT attack types + normal)
    2. Initialize weights with kaiming_normal_
    """

    def __init__(self, input_dim: int = 46, num_classes: int = 35):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** Forward pass on random input should produce logits of shape `[batch, 35]`

#### Step 2 — IoT LSTM Model

```python
# src/quantedge_services_handson/week03/lstm_model.py

import torch
import torch.nn as nn

class IoTAnomalyLSTM(nn.Module):
    """
    TODO:
    1. Build an LSTM for sequential traffic analysis:
       - Input: (batch, seq_len, feature_dim)
       - LSTM: 2 layers, hidden_size=128, bidirectional
       - Output head: Linear(256, num_classes)  # 256 because bidirectional
    2. Use the last hidden state for classification
    """

    def __init__(self, feature_dim: int = 46, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 35):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** Input `[32, 100, 46]` -> Output `[32, 35]`

#### Step 3 — Baseline Training (No AMP)

```python
# src/quantedge_services_handson/week03/baseline_trainer.py

import time
import torch

def train_baseline(model, train_loader, epochs: int = 5) -> dict:
    """
    TODO:
    1. Standard training loop (float32 everywhere)
    2. Record:
       - Wall-clock time per epoch
       - Peak GPU memory (torch.cuda.max_memory_allocated())
       - Throughput (samples/sec)
    3. Return dict: {'epoch_times': [...], 'peak_memory_mb': float, 'throughput': float}
    """
    pass
```

**Checkpoint:** Note down baseline throughput (samples/sec) — you'll compare later

#### Step 4 — Add Automatic Mixed Precision

```python
# src/quantedge_services_handson/week03/amp_trainer.py

import torch

def train_with_amp(model, train_loader, epochs: int = 5) -> dict:
    """
    TODO:
    1. Create GradScaler: scaler = torch.amp.GradScaler()
    2. In the training loop:
       a. with torch.autocast(device_type='cuda', dtype=torch.float16):
              output = model(batch_x)
              loss = criterion(output, batch_y)
       b. scaler.scale(loss).backward()
       c. scaler.step(optimizer)
       d. scaler.update()
       e. optimizer.zero_grad()
    3. Record same metrics as baseline
    4. Return dict with metrics
    """
    pass
```

**Checkpoint:** AMP throughput should be ~1.5-2x baseline. Peak memory should be lower.

#### Step 5 — OOM Provocation & Fix with Gradient Accumulation

```python
# src/quantedge_services_handson/week03/grad_accumulation.py

import torch

def find_max_batch_size(model, input_shape: tuple, device: str = 'cuda') -> int:
    """
    TODO:
    1. Start with batch_size = 64
    2. Double it each iteration
    3. Try a forward + backward pass
    4. Catch RuntimeError (CUDA OOM)
    5. Return the last batch_size that worked
    """
    pass

def train_with_gradient_accumulation(
    model, train_loader, accumulation_steps: int = 4, epochs: int = 5
) -> dict:
    """
    TODO:
    1. Combine AMP with gradient accumulation:
       for i, (batch_x, batch_y) in enumerate(train_loader):
           with autocast:
               loss = criterion(model(batch_x), batch_y)
               loss = loss / accumulation_steps
           scaler.scale(loss).backward()

           if (i + 1) % accumulation_steps == 0:
               scaler.step(optimizer)
               scaler.update()
               optimizer.zero_grad()

    2. This simulates a batch size of (actual_batch * accumulation_steps)
       without the memory cost
    3. Return metrics dict
    """
    pass
```

**Checkpoint:** Effective batch size = `batch_size * accumulation_steps` but memory stays at single-batch level

#### Step 6 — Compare & Log Results

```python
# src/quantedge_services_handson/week03/benchmark.py

def compare_training_modes(model_class, train_loader) -> None:
    """
    TODO:
    1. Run train_baseline() and capture metrics
    2. Run train_with_amp() and capture metrics
    3. Run train_with_gradient_accumulation() and capture metrics
    4. Print a comparison table:
       | Mode      | Time/epoch | Peak Mem (MB) | Throughput (samples/s) |
       |-----------|-----------|---------------|----------------------|
       | Baseline  | ...       | ...           | ...                  |
       | AMP       | ...       | ...           | ...                  |
       | AMP+Accum | ...       | ...           | ...                  |
    """
    pass
```

**Checkpoint:** A printed table proving AMP is faster

---

### Week 4 · Profiling & Bottleneck Identification

**Goal:** Find and fix the actual bottleneck in your training pipeline

**Package:** `src/quantedge_services_handson/week04/`

#### Step 1 — PyTorch Profiler Setup

```python
# src/quantedge_services_handson/week04/profiler.py

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training(model, train_loader, num_steps: int = 5) -> str:
    """
    TODO:
    1. Setup profiler:
       with profile(
           activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
           schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
           on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
           record_shapes=True,
           profile_memory=True,
           with_stack=True
       ) as prof:
    2. Run `num_steps` training iterations inside the profiler
    3. Call prof.step() after each iteration
    4. Return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    """
    pass
```

**Checkpoint:** A profiler table showing top-10 CUDA operations by time

#### Step 2 — Memory Summary Analysis

```python
# src/quantedge_services_handson/week04/memory_analysis.py

import torch

def analyze_memory(model, sample_batch) -> str:
    """
    TODO:
    1. Reset peak memory stats: torch.cuda.reset_peak_memory_stats()
    2. Run forward + backward on sample_batch
    3. Capture torch.cuda.memory_summary()
    4. Parse and return key stats:
       - Current allocated
       - Peak allocated
       - Number of allocations
    5. Return the full memory summary string
    """
    pass
```

**Checkpoint:** Memory summary printed — note the peak allocation

#### Step 3 — DataLoader Worker Tuning

```python
# src/quantedge_services_handson/week04/worker_tuning.py

def tune_num_workers(dataset, batch_size: int = 64) -> dict:
    """
    TODO:
    1. For num_workers in [0, 1, 2, 4, 8]:
       a. Create DataLoader with that num_workers
       b. Time how long it takes to iterate through one epoch
       c. Record the time
    2. Return dict: {num_workers: epoch_time_seconds}
    3. Print recommendation: "Optimal num_workers = X"
    """
    pass
```

**Checkpoint:** A table showing epoch time vs num_workers — find the sweet spot

#### Step 4 — Identify Bottleneck & Document

```python
# src/quantedge_services_handson/week04/diagnosis.py

def diagnose_bottleneck(profiler_output: str, memory_summary: str,
                        worker_timings: dict) -> str:
    """
    TODO:
    1. Analyze:
       - If DataLoader time > model time -> I/O bottleneck
       - If CUDA kernel time dominates -> GPU compute bottleneck
       - If CPU time dominates with low GPU util -> CPU bottleneck
    2. Return a diagnosis string:
       "BOTTLENECK: [IO|GPU|CPU] — [explanation] — FIX: [recommendation]"
    """
    pass
```

**Checkpoint:** A clear 1-line diagnosis of your bottleneck

#### Step 5 — TensorBoard Visualization

```bash
# After running profile_training(), launch TensorBoard:
tensorboard --logdir=./logs/profiler --port=6006

# Open http://localhost:6006 in your browser
# Navigate to the "PyTorch Profiler" tab
# Take a screenshot of the trace view for your records
```

**Checkpoint:** TensorBoard trace view open in browser showing GPU kernels timeline

---

## Phase 3 — Transformer (Weeks 5-6)

### Week 5 · Attention from Scratch

**Goal:** Build a Transformer encoder entirely from scratch — NO `nn.Transformer`

**Package:** `src/quantedge_services_handson/week05/`

#### Step 1 — Scaled Dot-Product Attention

```python
# src/quantedge_services_handson/week05/attention.py

import torch
import math

def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TODO:
    1. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
       - Use torch.matmul, NOT nn.functional
    2. If mask is provided, apply it: scores.masked_fill_(mask == 0, -1e9)
    3. Apply softmax: weights = softmax(scores, dim=-1)
    4. Compute output: output = weights @ V
    5. Return (output, weights)

    Shapes: Q,K,V: (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
    """
    pass
```

**Checkpoint:** `output.shape == Q.shape` and `weights.sum(dim=-1)` should be all 1s

#### Step 2 — Multi-Head Attention

```python
# src/quantedge_services_handson/week05/multi_head.py

import torch
import torch.nn as nn
from .attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """
    TODO:
    1. __init__:
       - d_model: 128, num_heads: 8 -> d_k = 16
       - Create W_Q, W_K, W_V, W_O as nn.Linear (no bias for now)
    2. forward(x):
       a. Project: Q = W_Q(x), K = W_K(x), V = W_V(x)
       b. Reshape to (batch, num_heads, seq_len, d_k)
       c. Call scaled_dot_product_attention
       d. Reshape back to (batch, seq_len, d_model)
       e. Project through W_O
       f. Return (output, attention_weights)

    NO nn.MultiheadAttention allowed!
    """

    def __init__(self, d_model: int = 128, num_heads: int = 8):
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
```

**Checkpoint:** Input shape `[2, 50, 128]` -> Output shape `[2, 50, 128]`

#### Step 3 — Positional Encoding

```python
# src/quantedge_services_handson/week05/positional_encoding.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    TODO:
    1. __init__:
       - Precompute PE matrix of shape (max_len, d_model)
       - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
       - Register as buffer (not parameter)
    2. forward(x):
       - Add PE[:seq_len] to x
       - Return x + PE
    """

    def __init__(self, d_model: int = 128, max_len: int = 5000):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** PE values at position 0 should alternate between sin(0)=0 and cos(0)=1

#### Step 4 — Transformer Encoder Block (Pre-Norm)

```python
# src/quantedge_services_handson/week05/transformer_block.py

import torch
import torch.nn as nn
from .multi_head import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    TODO (Pre-Norm architecture):
    1. __init__:
       - MultiHeadAttention (your implementation from Step 2)
       - FeedForward: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
       - Two LayerNorm layers
       - Dropout
    2. forward(x):
       - norm1 -> multi_head_attention -> dropout -> residual add
       - norm2 -> feed_forward -> dropout -> residual add
       - Return output
    """

    def __init__(self, d_model: int = 128, num_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.1):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** `block(x).shape == x.shape` — dimensions preserved

#### Step 5 — Stack 4 Blocks into Full Encoder

```python
# src/quantedge_services_handson/week05/encoder.py

import torch
import torch.nn as nn
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock

class TransformerEncoder(nn.Module):
    """
    TODO:
    1. __init__:
       - SinusoidalPositionalEncoding
       - nn.ModuleList of 4 TransformerBlocks
       - Final LayerNorm
       - Regression head: Linear(d_model, 1) for RUL prediction
    2. forward(x):
       - x: (batch, seq_len, input_dim) — sensor readings
       - Project input_dim -> d_model with a Linear layer
       - Add positional encoding
       - Pass through all 4 transformer blocks
       - Final norm
       - Mean pool over sequence dim
       - Regression head -> scalar RUL prediction
    """

    def __init__(self, input_dim: int = 21, d_model: int = 128,
                 num_heads: int = 8, num_layers: int = 4):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

**Checkpoint:** Input `[32, 50, 21]` (batch=32, seq=50, sensors=21) -> Output `[32, 1]`

#### Validation

```bash
python -c "
from quantedge_services_handson.week05.attention import scaled_dot_product_attention
from quantedge_services_handson.week05.encoder import TransformerEncoder
import torch

Q = K = V = torch.randn(2, 8, 50, 16)
out, weights = scaled_dot_product_attention(Q, K, V)
assert out.shape == (2, 8, 50, 16), f'Bad shape: {out.shape}'
assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8, 50), atol=1e-5)

model = TransformerEncoder(input_dim=21, d_model=128, num_heads=8, num_layers=4)
x = torch.randn(32, 50, 21)
rul_pred = model(x)
assert rul_pred.shape == (32, 1), f'Bad shape: {rul_pred.shape}'
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
print('Week 5 PASSED')
"
```

---

### Week 6 · Architecture Decisions & Attention Visualization

**Goal:** Train the transformer, visualize attention, write architecture rationale

**Package:** `src/quantedge_services_handson/week06/`

#### Step 1 — CMAPSS Dataset Loader

```python
# src/quantedge_services_handson/week06/cmapss_dataset.py

import torch
from torch.utils.data import Dataset

class CMAPSSDataset(Dataset):
    """
    TODO:
    1. Load NASA CMAPSS data (train_FD001.txt)
       - Columns: unit_id, cycle, op_setting1-3, sensor1-21
    2. For each engine unit, create sequences of length `seq_len`
    3. Compute RUL = max_cycle - current_cycle (capped at 125)
    4. Normalize sensor readings (StandardScaler per feature)
    5. Return (sensor_sequence, rul_label)
    """

    def __init__(self, data_path: str, seq_len: int = 50, max_rul: int = 125):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass
```

**Checkpoint:** `dataset[0]` returns tensors of shapes `[50, 21]` and `[1]`

#### Step 2 — Train to Convergence

```python
# src/quantedge_services_handson/week06/trainer.py

def train_cmapss(model, train_loader, val_loader,
                 epochs: int = 50, lr: float = 1e-4) -> dict:
    """
    TODO:
    1. Loss: MSELoss (this is regression)
    2. Optimizer: AdamW with weight_decay=1e-4
    3. Scheduler: CosineAnnealingLR
    4. Train loop with early stopping (patience=10)
    5. Metric: RMSE on validation set
    6. Return: {'train_losses': [...], 'val_rmses': [...], 'best_epoch': int}
    """
    pass
```

**Checkpoint:** Validation RMSE should decrease over epochs; best RMSE < 20 for FD001

#### Step 3 — Extract Attention Weights

```python
# src/quantedge_services_handson/week06/attention_extractor.py

import torch

def extract_attention_weights(model, sample_input: torch.Tensor) -> list[torch.Tensor]:
    """
    TODO:
    1. Modify your TransformerEncoder to store attention weights during forward
       (add a hook or return them)
    2. Run a forward pass on sample_input
    3. Collect attention weights from each layer
    4. Return list of tensors, each of shape (num_heads, seq_len, seq_len)
    """
    pass
```

**Checkpoint:** `len(weights) == 4` (one per layer), each shape `[8, 50, 50]`

#### Step 4 — Attention Heatmap Visualization

```python
# src/quantedge_services_handson/week06/heatmap_viz.py

import matplotlib.pyplot as plt
import torch

def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    layer: int = 0, head: int = 0,
    save_path: str = "attention_heatmap.png"
) -> None:
    """
    TODO:
    1. Select weights for specified layer and head
    2. Create a matplotlib heatmap (imshow or seaborn)
    3. X-axis: time steps (source positions)
    4. Y-axis: time steps (query positions)
    5. Add colorbar
    6. Title: f"Layer {layer}, Head {head}"
    7. Save to save_path
    """
    pass

def plot_sensor_importance(
    attention_weights: list[torch.Tensor],
    sensor_names: list[str],
    save_path: str = "sensor_importance.png"
) -> None:
    """
    TODO:
    1. Average attention across all layers and heads
    2. For each sensor channel, compute mean attention received
    3. Plot a bar chart: sensor name vs attention score
    4. Identify top-5 most attended sensors
    5. Save to save_path
    """
    pass
```

**Checkpoint:** Two saved PNG files showing attention patterns

#### Step 5 — Architecture Decision Rationale

```python
# src/quantedge_services_handson/week06/architecture_rationale.py

def write_architecture_rationale(save_path: str = "docs/week06_architecture_rationale.md") -> None:
    """
    TODO: Write a 1-page markdown document answering:
    1. Why encoder-only (not decoder or encoder-decoder) for RUL prediction?
    2. Why pre-norm instead of post-norm?
    3. Why 4 layers and 8 heads? (What did you try, what worked?)
    4. What did the attention heatmaps reveal about sensor importance?
    5. If you were to redo this, what would you change?

    Write this yourself — it's your interview prep material!
    """
    pass
```

**Checkpoint:** A markdown file you could confidently discuss in a technical interview

---

## Phase 4 — Fine-Tuning (Weeks 7-8)

### Week 7 · LoRA & QLoRA

**Goal:** Fine-tune Mistral-7B with QLoRA on instruction data

**Package:** `src/quantedge_services_handson/week07/`

#### Step 1 — Load Model in 4-bit

```python
# src/quantedge_services_handson/week07/quantized_loader.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_quantized_model(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """
    TODO:
    1. Create BitsAndBytesConfig:
       - load_in_4bit=True
       - bnb_4bit_quant_type="nf4"
       - bnb_4bit_compute_dtype=torch.float16
       - bnb_4bit_use_double_quant=True
    2. Load model with quantization_config
    3. Load tokenizer
    4. Print model memory footprint: model.get_memory_footprint() / 1e9 GB
    5. Return (model, tokenizer)
    """
    pass
```

**Checkpoint:** Memory footprint should be ~4-5 GB (not 14+ GB of full fp16)

#### Step 2 — Configure LoRA Adapters

```python
# src/quantedge_services_handson/week07/lora_config.py

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora(model, rank: int = 16) -> tuple:
    """
    TODO:
    1. prepare_model_for_kbit_training(model)
    2. Create LoraConfig:
       - r=rank
       - lora_alpha=32
       - target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
       - lora_dropout=0.05
       - task_type="CAUSAL_LM"
    3. Apply: model = get_peft_model(model, lora_config)
    4. Print trainable params:
       model.print_trainable_parameters()
       (should be ~0.1-0.5% of total)
    5. Return (model, lora_config)
    """
    pass
```

**Checkpoint:** Output should say something like "trainable params: 6M || all params: 3.7B || 0.16%"

#### Step 3 — Prepare Dataset

```python
# src/quantedge_services_handson/week07/dataset_prep.py

from datasets import load_dataset

def prepare_oasst1_dataset(tokenizer, max_length: int = 512):
    """
    TODO:
    1. Load dataset: load_dataset("OpenAssistant/oasst1")
    2. Filter to English, top-level messages with replies
    3. Format as instruction pairs:
       "<s>[INST] {instruction} [/INST] {response}</s>"
    4. Tokenize with tokenizer(text, max_length=max_length,
                               truncation=True, padding="max_length")
    5. Return tokenized dataset
    """
    pass
```

**Checkpoint:** `dataset[0]['input_ids']` should be a list of token IDs of length 512

#### Step 4 — Train with QLoRA

```python
# src/quantedge_services_handson/week07/qlora_trainer.py

from transformers import TrainingArguments, Trainer

def train_qlora(model, tokenizer, train_dataset, eval_dataset) -> None:
    """
    TODO:
    1. TrainingArguments:
       - output_dir="checkpoints/week07"
       - num_train_epochs=3
       - per_device_train_batch_size=4
       - gradient_accumulation_steps=4
       - learning_rate=2e-4
       - fp16=True
       - logging_steps=10
       - save_strategy="epoch"
       - evaluation_strategy="epoch"
    2. Trainer(model, args, train_dataset, eval_dataset, tokenizer)
    3. trainer.train()
    4. Save adapter: model.save_pretrained("checkpoints/week07/adapter")
    """
    pass
```

**Checkpoint:** Training loss should decrease; adapter saved to disk (~25 MB, not 14 GB)

#### Step 5 — Compare Memory: QLoRA vs Full Fine-Tune

```python
# src/quantedge_services_handson/week07/memory_comparison.py

def compare_memory_usage() -> None:
    """
    TODO:
    1. Load model in full fp16 — record memory
    2. Load model in 4-bit QLoRA — record memory
    3. Print comparison table:
       | Mode      | GPU Memory | Trainable Params |
       |-----------|-----------|------------------|
       | Full fp16 | XX GB     | 7B               |
       | QLoRA     | XX GB     | 6M               |
    """
    pass
```

**Checkpoint:** QLoRA uses ~3-4x less memory

---

### Week 8 · Domain Adaptation & Ollama Serving

**Goal:** Fine-tune on your domain data, merge adapters, serve via Ollama

**Package:** `src/quantedge_services_handson/week08/`

#### Step 1 — Filter StackOverflow Data

```python
# src/quantedge_services_handson/week08/data_filter.py

import pandas as pd

def filter_stackoverflow_data(
    csv_path: str,
    tags: list[str] = ["java", "spring-boot", "elasticsearch"],
    min_score: int = 5
) -> pd.DataFrame:
    """
    TODO:
    1. Load StackOverflow dump CSV
    2. Filter by tags (keep rows where tag column contains any of the target tags)
    3. Filter by score >= min_score
    4. Create Q&A pairs: (question_title + question_body, accepted_answer_body)
    5. Clean HTML tags from text
    6. Return DataFrame with columns: ['question', 'answer']
    """
    pass
```

**Checkpoint:** DataFrame with 5K-50K high-quality domain Q&A pairs

#### Step 2 — Fine-Tune on Domain Data

```python
# src/quantedge_services_handson/week08/domain_finetuner.py

def fine_tune_domain(
    model, tokenizer, qa_pairs, output_dir: str = "checkpoints/week08"
) -> None:
    """
    TODO:
    1. Format as instruction pairs:
       "[INST] {question} [/INST] {answer}"
    2. Tokenize dataset
    3. Train with same QLoRA setup as Week 7
    4. Save adapter weights
    """
    pass
```

**Checkpoint:** Training completes; adapter saved

#### Step 3 — Merge LoRA into Base Model

```python
# src/quantedge_services_handson/week08/adapter_merger.py

from peft import PeftModel

def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str = "models/merged_domain_model"
) -> None:
    """
    TODO:
    1. Load base model in fp16
    2. Load adapter: PeftModel.from_pretrained(base_model, adapter_path)
    3. Merge: merged = model.merge_and_unload()
    4. Save merged model: merged.save_pretrained(output_path)
    5. Save tokenizer: tokenizer.save_pretrained(output_path)
    """
    pass
```

**Checkpoint:** Full merged model saved (14+ GB) — adapter baked into weights

#### Step 4 — Package for Ollama

```python
# src/quantedge_services_handson/week08/ollama_packager.py

def create_ollama_modelfile(
    model_path: str, output_path: str = "Modelfile"
) -> None:
    """
    TODO:
    1. Write an Ollama Modelfile:
       FROM {model_path}
       PARAMETER temperature 0.7
       PARAMETER top_p 0.9
       SYSTEM "You are a senior Java/Spring Boot/Elasticsearch engineer..."
    2. Save to output_path
    3. Print instructions:
       ollama create domain-mistral -f Modelfile
       ollama run domain-mistral
    """
    pass
```

**Checkpoint:** `Modelfile` exists and is valid

#### Step 5 — Test with Enterprise Questions

```python
# src/quantedge_services_handson/week08/domain_tester.py

def test_domain_model(model_name: str = "domain-mistral") -> None:
    """
    TODO:
    1. Define 10 real enterprise questions:
       - "How do I configure connection pooling in Spring Boot with HikariCP?"
       - "What's the best way to handle nested aggregations in Elasticsearch?"
       - ... (8 more)
    2. For each question:
       - Send to Ollama: ollama.chat(model=model_name, messages=[...])
       - Print question and response
       - Rate quality: 1-5 (you do this manually)
    3. Print average quality score
    """
    pass
```

**Checkpoint:** At least 7/10 responses should be domain-relevant and accurate

---

## Phase 5 — Production (Weeks 9-10)

### Week 9 · Export Formats (TorchScript, ONNX, TensorRT)

**Goal:** Export the same model three ways, benchmark each

**Package:** `src/quantedge_services_handson/week09/`

#### Step 1 — Export with TorchScript

```python
# src/quantedge_services_handson/week09/torchscript_export.py

import torch

def export_torchscript(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "models/week09/model_scripted.pt"
) -> None:
    """
    TODO:
    1. Try torch.jit.script(model) first
    2. If that fails (common with dynamic control flow), use torch.jit.trace(model, sample_input)
    3. Save: scripted_model.save(output_path)
    4. Verify: load back and run inference on sample_input
    5. Assert outputs match: torch.allclose(original, loaded, atol=1e-5)
    """
    pass
```

**Checkpoint:** `.pt` file saved; loaded model produces same output

#### Step 2 — Export with ONNX

```python
# src/quantedge_services_handson/week09/onnx_export.py

import torch

def export_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "models/week09/model.onnx"
) -> None:
    """
    TODO:
    1. torch.onnx.export(
           model, sample_input, output_path,
           input_names=['sensor_input'],
           output_names=['rul_prediction'],
           dynamic_axes={'sensor_input': {0: 'batch_size'},
                        'rul_prediction': {0: 'batch_size'}},
           opset_version=17
       )
    2. Validate: import onnx; onnx.checker.check_model(output_path)
    3. Print model graph summary
    """
    pass
```

**Checkpoint:** `.onnx` file saved; `onnx.checker` passes without errors

#### Step 3 — ONNX Runtime Inference

```python
# src/quantedge_services_handson/week09/onnx_inference.py

import onnxruntime as ort
import torch

def run_onnx_inference(
    onnx_path: str, sample_input: torch.Tensor
) -> torch.Tensor:
    """
    TODO:
    1. Create InferenceSession:
       session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    2. Run inference:
       result = session.run(None, {'sensor_input': sample_input.numpy()})
    3. Convert back to tensor
    4. Return result
    """
    pass
```

**Checkpoint:** ONNX Runtime produces same output as PyTorch (within tolerance)

#### Step 4 — Benchmark All Formats

```python
# src/quantedge_services_handson/week09/benchmark.py

import time

def benchmark_inference(
    pytorch_model, torchscript_path: str, onnx_path: str,
    sample_input, num_runs: int = 100
) -> dict:
    """
    TODO:
    1. Warm up each model with 10 runs
    2. Benchmark each for `num_runs` iterations:
       a. PyTorch eager mode
       b. TorchScript
       c. ONNX Runtime (CPU)
       d. ONNX Runtime (CUDA) if available
    3. Record: mean latency, p50, p99
    4. Return dict and print comparison table:
       | Format       | Mean (ms) | P50 (ms) | P99 (ms) |
       |-------------|----------|---------|---------|
       | PyTorch     | ...      | ...     | ...     |
       | TorchScript | ...      | ...     | ...     |
       | ONNX (CPU)  | ...      | ...     | ...     |
       | ONNX (CUDA) | ...      | ...     | ...     |
    """
    pass
```

**Checkpoint:** A latency table with clear winner identified

#### Step 5 — Decision Guide

```python
# src/quantedge_services_handson/week09/decision_guide.py

def write_export_decision_guide(
    benchmark_results: dict,
    save_path: str = "docs/week09_export_guide.md"
) -> None:
    """
    TODO: Write a markdown document covering:
    1. When to use TorchScript: (pros/cons you observed)
    2. When to use ONNX: (pros/cons you observed)
    3. Edge vs Cloud recommendation
    4. Your benchmark table
    5. Dynamic batch axis — why it matters for production
    """
    pass
```

**Checkpoint:** A decision guide backed by YOUR actual benchmark numbers

---

### Week 10 · Quantization, FastAPI Serving & Docker

**Goal:** Build a production inference API with quantized model

**Package:** `src/quantedge_services_handson/week10/`

#### Step 1 — INT8 Quantization

```python
# src/quantedge_services_handson/week10/quantization.py

from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(
    onnx_path: str,
    output_path: str = "models/week10/model_int8.onnx"
) -> dict:
    """
    TODO:
    1. quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    2. Compare file sizes: original vs quantized
    3. Run inference on both; compare outputs
    4. Compute accuracy delta (use test set)
    5. Return: {'original_size_mb': X, 'quantized_size_mb': Y,
                'accuracy_original': A, 'accuracy_quantized': B}
    """
    pass
```

**Checkpoint:** Quantized model is ~2-4x smaller; accuracy drop < 1%

#### Step 2 — FastAPI Endpoint

```python
# src/quantedge_services_handson/week10/serving.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="QuantEdge RUL Prediction Service")

class PredictionRequest(BaseModel):
    """
    TODO: Define input schema:
    - sensor_readings: list of lists (seq_len x 21 features)
    - Validation: seq_len between 10-200, features exactly 21
    - Add Field(..., description=...) for docs
    """
    sensor_readings: list[list[float]] = Field(
        ..., description="Sensor readings: seq_len x 21 features"
    )

class PredictionResponse(BaseModel):
    rul_prediction: float
    confidence: float
    model_version: str
    correlation_id: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    TODO:
    1. Validate input dimensions
    2. Convert to numpy array
    3. Run ONNX inference
    4. Return prediction with metadata
    """
    pass
```

**Checkpoint:** `curl -X POST http://localhost:8000/predict -d '...'` returns a JSON response

#### Step 3 — Middleware (Timeout, Logging, Correlation ID)

```python
# src/quantedge_services_handson/week10/middleware.py

import uuid
import asyncio
import structlog
from starlette.middleware.base import BaseHTTPMiddleware

class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    TODO:
    1. Extract X-Correlation-ID from request headers (or generate UUID)
    2. Add to structlog context
    3. Add to response headers
    """
    pass

class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    TODO:
    1. Wrap request handling in asyncio.wait_for(timeout=30)
    2. On timeout, return 504 Gateway Timeout
    """
    pass

def setup_logging():
    """
    TODO:
    1. Configure structlog with JSON output
    2. Include: timestamp, correlation_id, method, path, status_code, latency_ms
    """
    pass
```

**Checkpoint:** Logs show structured JSON with correlation IDs

#### Step 4 — Dockerfile

Create `src/quantedge_services_handson/week10/Dockerfile`:

```dockerfile
# TODO:
# 1. Base image: python:3.11-slim
# 2. Install dependencies
# 3. Copy model and code
# 4. Expose port 8000
# 5. Health check
# 6. CMD: uvicorn week10.serving:app --host 0.0.0.0 --port 8000
```

**Checkpoint:** `docker build -t quantedge-rul . && docker run -p 8000:8000 quantedge-rul`

#### Step 5 — Load Test & Latency Report

```python
# src/quantedge_services_handson/week10/load_test.py

def load_test(url: str = "http://localhost:8000/predict", num_requests: int = 100):
    """
    TODO:
    1. Send `num_requests` concurrent requests using httpx.AsyncClient
    2. Record latency for each request
    3. Compute and print:
       - P50, P95, P99 latency
       - Throughput (requests/sec)
       - Error rate
    """
    pass
```

**Checkpoint:** P50 < 50ms, P99 < 200ms, 0% error rate

---

## Phase 6 — MLOps (Weeks 11-12)

### Week 11 · Experiment Tracking & Canary Deployment

**Goal:** Track all experiments in MLflow, define canary deployment strategy

**Package:** `src/quantedge_services_handson/week11/`

#### Step 1 — MLflow Setup & Experiment Logging

```python
# src/quantedge_services_handson/week11/mlflow_setup.py

import mlflow

def setup_mlflow(tracking_uri: str = "sqlite:///mlflow.db") -> None:
    """
    TODO:
    1. mlflow.set_tracking_uri(tracking_uri)
    2. Create experiment: "quantedge-pytorch-mastery"
    """
    pass
```

**Checkpoint:** `mlflow ui` at localhost:5000 shows your experiment

#### Step 2 — Log Experiments

```python
# src/quantedge_services_handson/week11/experiment_logger.py

import mlflow

def log_experiment(
    experiment_name: str,
    params: dict,
    metrics: dict,
    model_path: str | None = None,
    tags: dict | None = None
) -> str:
    """
    TODO:
    1. mlflow.set_experiment(experiment_name)
    2. with mlflow.start_run():
       a. mlflow.log_params(params)
       b. mlflow.log_metrics(metrics)
       c. If model_path: mlflow.log_artifact(model_path)
       d. If tags: mlflow.set_tags(tags)
    3. Return run_id
    """
    pass

def log_all_experiments() -> None:
    """
    TODO: Log experiments from all prior weeks with YOUR actual numbers:
    Week 1: params={"task": "autograd"}, metrics={"grad_match": 1.0}
    Week 2: params={"epochs": 10, "lr": 1e-3}, metrics={"best_val_acc": ...}
    Week 3: params={"amp": True}, metrics={"speedup": ..., "peak_mem_mb": ...}
    Week 5: params={"layers": 4, "heads": 8}, metrics={"val_rmse": ...}
    Week 7: params={"rank": 16, "base_model": "Mistral-7B"}, metrics={"train_loss": ...}
    Week 9: params={"format": "onnx"}, metrics={"latency_p50_ms": ..., "latency_p99_ms": ...}
    """
    pass
```

**Checkpoint:** MLflow dashboard shows 6+ experiments with real metrics

#### Step 3 — Model Registry

```python
# src/quantedge_services_handson/week11/model_registry.py

import mlflow

def register_model(run_id: str, model_name: str, stage: str = "Staging") -> None:
    """
    TODO:
    1. Register model from run:
       mlflow.register_model(f"runs:/{run_id}/model", model_name)
    2. Transition to stage:
       client = mlflow.tracking.MlflowClient()
       client.transition_model_version_stage(model_name, version, stage)
    3. Add version description
    """
    pass
```

**Checkpoint:** Model appears in MLflow Models tab with version and stage

#### Step 4 — Model Card

```python
# src/quantedge_services_handson/week11/model_card.py

def write_model_card(save_path: str = "docs/week11_model_card.md") -> None:
    """
    TODO: Write a model card with these sections:
    1. Model Details (name, version, type, framework)
    2. Intended Use (primary use case, out-of-scope uses)
    3. Training Data (source, size, preprocessing)
    4. Evaluation Results (metrics table from MLflow)
    5. Limitations (known failure modes, biases)
    6. Ethical Considerations

    This is YOUR document — write it based on your actual results!
    """
    pass
```

**Checkpoint:** A complete model card you'd be proud to attach to a production model

#### Step 5 — Canary Deployment & Rollback Strategy

```python
# src/quantedge_services_handson/week11/canary_strategy.py

def define_canary_strategy() -> dict:
    """
    TODO: Define and return a canary deployment config:
    {
        "canary_percentage": 10,
        "promotion_criteria": {
            "min_requests": 1000,
            "max_latency_p99_ms": 200,
            "max_error_rate": 0.01,
            "min_accuracy_vs_baseline": 0.98
        },
        "rollback_triggers": {
            "latency_p99_ms_threshold": 500,
            "error_rate_threshold": 0.05,
            "accuracy_drop_threshold": 0.05
        },
        "promotion_steps": [10, 25, 50, 100],
        "observation_period_minutes": 30
    }
    """
    pass

def write_rollback_runbook(save_path: str = "docs/week11_rollback_runbook.md") -> None:
    """
    TODO: Write a runbook covering:
    1. When to rollback (specific thresholds)
    2. How to rollback (exact commands/steps)
    3. Who to notify
    4. Post-mortem template
    """
    pass
```

**Checkpoint:** A canary config and rollback runbook that ops could follow at 3 AM

---

### Week 12 · Drift Detection, Monitoring & ADR

**Goal:** Detect data drift, monitor inference, write Architecture Decision Records

**Package:** `src/quantedge_services_handson/week12/`

#### Step 1 — Data Drift Detection (PSI)

```python
# src/quantedge_services_handson/week12/psi.py

import numpy as np

def compute_psi(
    reference: np.ndarray, current: np.ndarray, num_buckets: int = 10
) -> float:
    """
    TODO: Population Stability Index
    1. Create `num_buckets` equal-frequency bins from reference distribution
    2. Compute percentage of values in each bin for reference and current
    3. PSI = sum((current_pct - ref_pct) * ln(current_pct / ref_pct))
    4. Handle zero bins: add small epsilon (1e-4)
    5. Return PSI value

    Interpretation:
    - PSI < 0.1: No significant drift
    - 0.1 <= PSI < 0.25: Moderate drift
    - PSI >= 0.25: Significant drift
    """
    pass
```

**Checkpoint:** Same dist -> PSI ~0, shifted dist -> PSI > 0.25

#### Step 2 — KS-Test per Feature

```python
# src/quantedge_services_handson/week12/ks_drift.py

from scipy import stats
import numpy as np

def detect_drift_per_feature(
    reference_df, current_df, feature_columns: list[str],
    alpha: float = 0.05
) -> dict:
    """
    TODO:
    1. For each feature column:
       a. Run KS-test: stats.ks_2samp(reference[col], current[col])
       b. Record statistic and p-value
       c. Flag as drifted if p_value < alpha
    2. Return dict:
       {
         "feature_name": {
           "ks_statistic": float,
           "p_value": float,
           "drifted": bool,
           "psi": float
         }
       }
    """
    pass
```

**Checkpoint:** Artificially shift one feature — it should be flagged as drifted

#### Step 3 — Concept Drift Detection

```python
# src/quantedge_services_handson/week12/concept_drift.py

import numpy as np

def detect_concept_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.1
) -> dict:
    """
    TODO:
    1. Compute distribution shift in model predictions:
       - Mean shift: abs(mean(current) - mean(reference)) / mean(reference)
       - Std shift: abs(std(current) - std(reference)) / std(reference)
    2. Run KS-test on prediction distributions
    3. Return: {'mean_shift': float, 'std_shift': float,
                'ks_statistic': float, 'drifted': bool}
    """
    pass
```

**Checkpoint:** Same predictions -> no drift; shifted -> drift detected

#### Step 4 — Prometheus Metrics Exporter

```python
# src/quantedge_services_handson/week12/prometheus_metrics.py

def generate_prometheus_metrics(
    job_stats: dict, inference_stats: dict
) -> str:
    """
    TODO: Generate Prometheus text format metrics:
    1. Inference metrics:
       - quantedge_inference_requests_total{model="rul", status="success"} 1234
       - quantedge_inference_latency_seconds{quantile="0.5"} 0.045
       - quantedge_inference_latency_seconds{quantile="0.99"} 0.189
    2. Model metrics:
       - quantedge_model_predictions_total 5678
       - quantedge_drift_psi{feature="sensor1"} 0.08
    3. System metrics:
       - quantedge_gpu_utilization_percent 67.5
       - quantedge_gpu_memory_used_bytes 4294967296
    4. Return the text format string (Prometheus exposition format)
    """
    pass
```

**Checkpoint:** Output is valid Prometheus text format

#### Step 5 — Audit Logging

```python
# src/quantedge_services_handson/week12/audit_logger.py

import json
from datetime import datetime

class AuditLogger:
    """
    TODO:
    1. __init__: Open a JSONL file for append
    2. log(event_type, details):
       - Write one JSON line: {"timestamp": ISO8601, "event": event_type,
                                "user": "system", "details": {...}}
    3. Support event types:
       - "model_deployed", "model_rolled_back", "drift_detected",
         "experiment_logged", "prediction_served"
    """

    def __init__(self, log_path: str = "logs/audit.jsonl"):
        pass

    def log(self, event_type: str, details: dict) -> None:
        pass

    def query(self, event_type: str | None = None,
              start_time: str | None = None) -> list[dict]:
        """Read back and filter audit entries."""
        pass
```

**Checkpoint:** `audit.jsonl` with valid JSON lines; `query()` returns correct entries

#### Step 6 — Architecture Decision Records (ADRs)

```python
# src/quantedge_services_handson/week12/adr_generator.py

def generate_adr(
    adr_id: str, title: str, context: str,
    decision: str, consequences: str,
    status: str = "Accepted",
    save_dir: str = "docs/adr"
) -> str:
    """
    TODO: Generate a markdown ADR file with this template:
    # ADR-{id}: {title}
    **Status:** {status}
    **Date:** {today}

    ## Context
    {context}

    ## Decision
    {decision}

    ## Consequences
    {consequences}
    """
    pass

def generate_all_adrs() -> None:
    """
    TODO: Write 6 ADRs documenting your key decisions:

    ADR-001: "Use PyTorch over TensorFlow"
    ADR-002: "QLoRA for Fine-Tuning on Consumer Hardware"
    ADR-003: "ONNX as Primary Export Format"
    ADR-004: "Pre-Norm Transformer Architecture"
    ADR-005: "PSI + KS-Test for Drift Detection"
    ADR-006: "MLflow for Experiment Tracking"

    Write these from YOUR experience — what did you learn?
    """
    pass
```

**Checkpoint:** 6 ADR markdown files in `docs/adr/`

#### Final Validation

```bash
python -c "
from quantedge_services_handson.week12.psi import compute_psi
from quantedge_services_handson.week12.concept_drift import detect_concept_drift
import numpy as np

ref = np.random.normal(0, 1, 10000)
same = np.random.normal(0, 1, 10000)
shifted = np.random.normal(2, 1, 10000)

psi_same = compute_psi(ref, same)
psi_shifted = compute_psi(ref, shifted)
print(f'PSI (same dist): {psi_same:.4f} — should be < 0.1')
print(f'PSI (shifted):   {psi_shifted:.4f} — should be > 0.25')

no_drift = detect_concept_drift(ref, same)
has_drift = detect_concept_drift(ref, shifted)
print(f'No drift: {no_drift[\"drifted\"]}')
print(f'Has drift: {has_drift[\"drifted\"]}')

print('Week 12 PASSED')
"
```

---

## Rules for Yourself

1. **Type every line yourself** — no copy-paste from `src/quantedge_services/`
2. **Hit every checkpoint** before moving to the next step
3. **If stuck > 30 min**, read the reference implementation, then close it and write from memory
4. **Commit after each week** — your git history IS your learning journal
5. **Write the docs** (model card, ADRs, runbook) — interviewers ask about these

---

## Completion Checklist

| Week | Package | Key Checkpoint | Done |
|------|---------|---------------|------|
| 1 | `week01/` | Autograd matches manual backward | [ ] |
| 2 | `week02/` | Training resumes from checkpoint | [ ] |
| 3 | `week03/` | AMP 2x faster than baseline | [ ] |
| 4 | `week04/` | Bottleneck identified and documented | [ ] |
| 5 | `week05/` | Transformer builds from raw matmul | [ ] |
| 6 | `week06/` | Attention heatmaps saved | [ ] |
| 7 | `week07/` | QLoRA adapter saved (~25 MB) | [ ] |
| 8 | `week08/` | Domain model answers 7/10 correctly | [ ] |
| 9 | `week09/` | Latency table with 3 formats | [ ] |
| 10 | `week10/` | Dockerized API with P99 < 200ms | [ ] |
| 11 | `week11/` | MLflow dashboard with all experiments | [ ] |
| 12 | `week12/` | 6 ADRs + drift detection working | [ ] |

# Hands-On Log — Week 03: Mixed Precision & OOM Debugging

**Package:** `src/quantedge_services_handson/week03/`
**Goal:** Same model running 2x faster — proven with timing logs

---

## Step 1 — IoT Anomaly MLP
**File:** `week03/mlp_model.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `mlp_model.py`
- [ ] Implement `IoTAnomalyMLP(nn.Module)`: 46 → 256 → 128 → 64 → 35
- [ ] Use BatchNorm + ReLU + Dropout between layers
- [ ] Initialize with kaiming_normal_

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week03.mlp_model import IoTAnomalyMLP
import torch
model = IoTAnomalyMLP(input_dim=46, num_classes=35)
print(f'Output: {model(torch.randn(32, 46)).shape}')
"
```
Expected: `Output: torch.Size([32, 35])`

### Notes


---

## Step 2 — IoT Anomaly LSTM
**File:** `week03/lstm_model.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `lstm_model.py`
- [ ] Implement `IoTAnomalyLSTM(nn.Module)`: bidirectional, 2-layer, hidden=128
- [ ] Use last hidden state → Linear(256, 35)

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week03.lstm_model import IoTAnomalyLSTM
import torch
model = IoTAnomalyLSTM(feature_dim=46, num_classes=35)
print(f'Output: {model(torch.randn(32, 100, 46)).shape}')
"
```
Expected: `Output: torch.Size([32, 35])`

### Notes


---

## Step 3 — Baseline Training (No AMP)
**File:** `week03/baseline_trainer.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `baseline_trainer.py`
- [ ] Implement `train_baseline(model, train_loader, epochs=5) -> dict`
- [ ] Standard float32 training loop
- [ ] Record: wall-clock time/epoch, peak GPU memory, throughput (samples/sec)

### Checkpoint
Record these numbers — you'll compare in Step 6:
- Throughput: _____ samples/sec
- Peak memory: _____ MB
- Time/epoch: _____ sec

### Notes


---

## Step 4 — Automatic Mixed Precision
**File:** `week03/amp_trainer.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `amp_trainer.py`
- [ ] Implement `train_with_amp(model, train_loader, epochs=5) -> dict`
- [ ] Use `torch.autocast(device_type='cuda', dtype=torch.float16)`
- [ ] Use `torch.amp.GradScaler()` for loss scaling
- [ ] Record same metrics as baseline

### Checkpoint
- Throughput: _____ samples/sec (should be ~1.5-2x baseline)
- Peak memory: _____ MB (should be lower)
- Time/epoch: _____ sec

### Notes


---

## Step 5 — OOM Provocation & Gradient Accumulation
**File:** `week03/grad_accumulation.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `grad_accumulation.py`
- [ ] Implement `find_max_batch_size(model, input_shape) -> int`
  - Double batch size until CUDA OOM, return last working size
- [ ] Implement `train_with_gradient_accumulation(model, loader, accumulation_steps=4)`
  - Combine AMP + gradient accumulation
  - Effective batch = actual_batch * accumulation_steps

### Checkpoint
- Max batch size found: _____
- Effective batch with accumulation: _____
- Memory usage stays at single-batch level: Yes/No

### Notes


---

## Step 6 — Compare & Log Results
**File:** `week03/benchmark.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `benchmark.py`
- [ ] Implement `compare_training_modes(model_class, train_loader)`
- [ ] Run all three modes, print comparison table

### Checkpoint
Fill in YOUR numbers:

| Mode      | Time/epoch | Peak Mem (MB) | Throughput (samples/s) |
|-----------|-----------|---------------|----------------------|
| Baseline  |           |               |                      |
| AMP       |           |               |                      |
| AMP+Accum |           |               |                      |

### Notes


---

## Week 03 Summary
**All Steps Passed:** `No`
**AMP Speedup Achieved:** ___x
**Key Learnings:**

**Commit SHA:**

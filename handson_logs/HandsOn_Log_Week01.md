# Hands-On Log — Week 01: Tensors & Autograd

**Package:** `src/quantedge_services_handson/week01/`
**Goal:** Understand `.backward()` in memory on real financial data

---

## Step 1 — Load Forex CSV into Tensors
**File:** `week01/tensors.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `tensors.py`
- [ ] Implement `load_forex_data(csv_path: str) -> torch.Tensor`
- [ ] Read CSV, extract 'close' prices, convert to float32 tensor

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week01.tensors import load_forex_data
import torch
# Use your actual CSV path or create synthetic data for testing
prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
print(f'Shape: {prices.shape}, Dtype: {prices.dtype}')
"
```
Expected: `torch.Size([N]) torch.float32`

### Notes


---

## Step 2 — Compute Price Deltas with Autograd
**File:** `week01/autograd.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `autograd.py`
- [ ] Implement `compute_price_deltas(prices) -> Tensor` using autograd
- [ ] Implement `manual_backward(prices) -> Tensor` using chain rule by hand
- [ ] Compare both — they must match

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week01.autograd import compute_price_deltas, manual_backward
import torch
prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
grad_auto = compute_price_deltas(prices.clone())
grad_manual = manual_backward(prices.clone())
print('Match:', torch.allclose(grad_auto, grad_manual, atol=1e-5))
"
```
Expected: `Match: True`

### Notes


---

## Step 3 — Inject NaN and Fix It
**File:** `week01/nan_handling.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `nan_handling.py`
- [ ] Implement `inject_nan_and_fix(prices) -> Tensor`
- [ ] Inject NaN at every 7th position (simulate weekend gaps)
- [ ] Forward-fill to repair — no NaN must remain

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week01.nan_handling import inject_nan_and_fix
import torch
prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
fixed = inject_nan_and_fix(prices)
print('NaN-free:', not torch.isnan(fixed).any().item())
"
```
Expected: `NaN-free: True`

### Notes


---

## Step 4 — Rolling Volatility & Momentum
**File:** `week01/rolling_ops.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `rolling_ops.py`
- [ ] Implement `rolling_volatility(prices, window=20) -> Tensor`
  - Log returns → unfold into windows → std per window
- [ ] Implement `rolling_momentum(prices, window=10) -> Tensor`
  - `mom[i] = prices[i] - prices[i - window]`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week01.rolling_ops import rolling_volatility, rolling_momentum
import torch
prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
vol = rolling_volatility(prices, window=20)
mom = rolling_momentum(prices, window=10)
print(f'Volatility shape: {vol.shape}')
print(f'Momentum shape: {mom.shape}')
print('No NaN:', not torch.isnan(vol).any().item() and not torch.isnan(mom).any().item())
"
```
Expected: Valid shapes, `No NaN: True`

### Notes


---

## Step 5 — Device Management
**File:** `week01/device_mgmt.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `device_mgmt.py`
- [ ] Implement `demonstrate_device_transfer(prices) -> dict`
- [ ] Check CUDA, move tensor GPU↔CPU, time both
- [ ] Return `{'device_available': bool, 'cpu_time_ms': float, 'gpu_time_ms': float}`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week01.device_mgmt import demonstrate_device_transfer
import torch
prices = torch.randn(100000, dtype=torch.float32)
result = demonstrate_device_transfer(prices)
for k, v in result.items():
    print(f'  {k}: {v}')
"
```
Expected: Timing dict with GPU faster for large tensors

### Notes


---

## Week 01 Summary
**All Steps Passed:** `No`
**Key Learnings:**

**Commit SHA:**

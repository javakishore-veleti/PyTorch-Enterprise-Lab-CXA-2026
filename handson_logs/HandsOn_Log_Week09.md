# Hands-On Log — Week 09: Export Formats (TorchScript, ONNX, TensorRT)

**Package:** `src/quantedge_services_handson/week09/`
**Goal:** Same model exported three ways — latency table with decision guide

---

## Step 1 — Export with TorchScript
**File:** `week09/torchscript_export.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `torchscript_export.py`
- [ ] Implement `export_torchscript(model, sample_input, output_path)`
- [ ] Try `torch.jit.script` first, fallback to `torch.jit.trace`
- [ ] Save, reload, verify output matches

### Checkpoint
- File saved: Yes/No
- Output matches original: Yes/No (atol=1e-5)

### Notes


---

## Step 2 — Export with ONNX
**File:** `week09/onnx_export.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `onnx_export.py`
- [ ] Implement `export_onnx(model, sample_input, output_path)`
- [ ] Use dynamic_axes for batch dimension, opset_version=17
- [ ] Validate with `onnx.checker.check_model`

### Checkpoint
- File saved: Yes/No
- onnx.checker passed: Yes/No

### Notes


---

## Step 3 — ONNX Runtime Inference
**File:** `week09/onnx_inference.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `onnx_inference.py`
- [ ] Implement `run_onnx_inference(onnx_path, sample_input) -> Tensor`
- [ ] Create InferenceSession with CUDAExecutionProvider
- [ ] Verify output matches PyTorch within tolerance

### Checkpoint
- ONNX Runtime output matches PyTorch: Yes/No

### Notes


---

## Step 4 — Benchmark All Formats
**File:** `week09/benchmark.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `benchmark.py`
- [ ] Implement `benchmark_inference(pytorch_model, torchscript_path, onnx_path, sample_input)`
- [ ] Warm up 10 runs, benchmark 100 runs each
- [ ] Record mean, p50, p99 latency

### Checkpoint
Fill in YOUR numbers:

| Format       | Mean (ms) | P50 (ms) | P99 (ms) |
|-------------|----------|---------|---------|
| PyTorch     |          |         |         |
| TorchScript |          |         |         |
| ONNX (CPU)  |          |         |         |
| ONNX (CUDA) |          |         |         |

Winner: _____

### Notes


---

## Step 5 — Decision Guide
**File:** `week09/decision_guide.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `decision_guide.py`
- [ ] Write `docs/week09_export_guide.md` covering:
  - TorchScript pros/cons
  - ONNX pros/cons
  - Edge vs Cloud recommendation
  - Your benchmark table

### Checkpoint
Document written: Yes/No

### Notes


---

## Week 09 Summary
**All Steps Passed:** `No`
**Fastest Format:** _____
**Key Learnings:**

**Commit SHA:**

# Hands-On Log — Week 04: Profiling & Bottleneck Identification

**Package:** `src/quantedge_services_handson/week04/`
**Goal:** Find and fix the actual bottleneck — identified, documented, fixed

---

## Step 1 — PyTorch Profiler Setup
**File:** `week04/profiler.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `profiler.py`
- [ ] Implement `profile_training(model, train_loader, num_steps=5) -> str`
- [ ] Use `torch.profiler.profile` with CPU + CUDA activities
- [ ] Export to TensorBoard: `tensorboard_trace_handler('./logs/profiler')`
- [ ] Return `key_averages().table(sort_by="cuda_time_total", row_limit=10)`

### Checkpoint
Top-3 CUDA operations by time:
1. _____
2. _____
3. _____

### Notes


---

## Step 2 — Memory Summary Analysis
**File:** `week04/memory_analysis.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `memory_analysis.py`
- [ ] Implement `analyze_memory(model, sample_batch) -> str`
- [ ] Reset peak stats → forward+backward → capture `torch.cuda.memory_summary()`

### Checkpoint
- Current allocated: _____ MB
- Peak allocated: _____ MB
- Number of allocations: _____

### Notes


---

## Step 3 — DataLoader Worker Tuning
**File:** `week04/worker_tuning.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `worker_tuning.py`
- [ ] Implement `tune_num_workers(dataset, batch_size=64) -> dict`
- [ ] Test num_workers in [0, 1, 2, 4, 8], time one epoch each

### Checkpoint
| num_workers | Epoch Time (sec) |
|-------------|-----------------|
| 0           |                 |
| 1           |                 |
| 2           |                 |
| 4           |                 |
| 8           |                 |

Optimal: _____

### Notes


---

## Step 4 — Identify Bottleneck & Document
**File:** `week04/diagnosis.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `diagnosis.py`
- [ ] Implement `diagnose_bottleneck(profiler_output, memory_summary, worker_timings) -> str`
- [ ] Analyze: I/O vs GPU compute vs CPU bottleneck
- [ ] Return: `"BOTTLENECK: [IO|GPU|CPU] — [explanation] — FIX: [recommendation]"`

### Checkpoint
Your diagnosis: _____

### Notes


---

## Step 5 — TensorBoard Visualization
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Run `tensorboard --logdir=./logs/profiler --port=6006`
- [ ] Open PyTorch Profiler tab in browser
- [ ] Take screenshot of GPU kernel timeline

### Checkpoint
Screenshot saved: Yes/No

### Notes


---

## Week 04 Summary
**All Steps Passed:** `No`
**Bottleneck Found:** _____ (IO / GPU / CPU)
**Key Learnings:**

**Commit SHA:**

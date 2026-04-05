# Hands-On Log — Week 06: Architecture Decisions & Attention Visualization

**Package:** `src/quantedge_services_handson/week06/`
**Goal:** Trained transformer with visualized attention — every choice explainable

---

## Step 1 — CMAPSS Dataset Loader
**File:** `week06/cmapss_dataset.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `cmapss_dataset.py`
- [ ] Implement `CMAPSSDataset(Dataset)`
- [ ] Load NASA CMAPSS `train_FD001.txt` (unit_id, cycle, 3 op_settings, 21 sensors)
- [ ] Create sequences of length `seq_len=50`
- [ ] RUL = max_cycle - current_cycle, capped at 125
- [ ] Normalize sensor readings (StandardScaler)

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week06.cmapss_dataset import CMAPSSDataset
ds = CMAPSSDataset('path/to/train_FD001.txt', seq_len=50)
x, y = ds[0]
print(f'Sensor seq: {x.shape}, RUL: {y.shape}')
"
```
Expected: `Sensor seq: torch.Size([50, 21]), RUL: torch.Size([1])`

### Notes


---

## Step 2 — Train to Convergence
**File:** `week06/trainer.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `trainer.py`
- [ ] Implement `train_cmapss(model, train_loader, val_loader, epochs=50, lr=1e-4) -> dict`
- [ ] Loss: MSELoss, Optimizer: AdamW, Scheduler: CosineAnnealingLR
- [ ] Early stopping (patience=10), metric: RMSE

### Checkpoint
- Best validation RMSE: _____ (target < 20 for FD001)
- Best epoch: _____

### Notes


---

## Step 3 — Extract Attention Weights
**File:** `week06/attention_extractor.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `attention_extractor.py`
- [ ] Implement `extract_attention_weights(model, sample_input) -> list[Tensor]`
- [ ] Collect weights from each of the 4 layers

### Checkpoint
```bash
# len(weights) == 4, each shape [8, 50, 50]
```
Expected: 4 tensors of shape `[num_heads, seq_len, seq_len]`

### Notes


---

## Step 4 — Attention Heatmap Visualization
**File:** `week06/heatmap_viz.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `heatmap_viz.py`
- [ ] Implement `plot_attention_heatmap(weights, layer, head, save_path)`
- [ ] Implement `plot_sensor_importance(weights, sensor_names, save_path)`
- [ ] Top-5 most-attended sensors identified

### Checkpoint
- Heatmap PNG saved: Yes/No
- Sensor importance PNG saved: Yes/No
- Top-5 sensors: _____, _____, _____, _____, _____

### Notes


---

## Step 5 — Architecture Decision Rationale
**File:** `week06/architecture_rationale.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `architecture_rationale.py`
- [ ] Write to `docs/week06_architecture_rationale.md` answering:
  1. Why encoder-only?
  2. Why pre-norm?
  3. Why 4 layers / 8 heads?
  4. What did attention heatmaps reveal?
  5. What would you change?

### Checkpoint
Document written and you can explain each choice: Yes/No

### Notes


---

## Week 06 Summary
**All Steps Passed:** `No`
**Best RMSE:** _____
**Key Learnings:**

**Commit SHA:**

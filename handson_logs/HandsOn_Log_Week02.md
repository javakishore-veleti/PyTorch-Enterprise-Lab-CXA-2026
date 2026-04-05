# Hands-On Log — Week 02: Training Loop & DataLoader

**Package:** `src/quantedge_services_handson/week02/`
**Goal:** Clean training loop on financial complaints — trains, saves, resumes, fully reproducible

---

## Step 1 — Custom Map-Style Dataset
**File:** `week02/dataset.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `dataset.py`
- [ ] Implement `ComplaintDataset(Dataset)` with `__init__`, `__len__`, `__getitem__`
- [ ] Tokenize texts into fixed-length integer sequences
- [ ] Convert string labels to integer class indices

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week02.dataset import ComplaintDataset
ds = ComplaintDataset(['bad service']*100 + ['good bank']*100, ['complaint']*100 + ['praise']*100, max_len=32)
x, y = ds[0]
print(f'Size: {len(ds)}, x: {x.shape} {x.dtype}, y: {y.shape} {y.dtype}')
"
```
Expected: `Size: 200, x: torch.Size([32]) torch.int64, y: torch.Size([]) torch.int64`

### Notes


---

## Step 2 — Iterable-Style Dataset
**File:** `week02/iterable_dataset.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `iterable_dataset.py`
- [ ] Implement `ComplaintIterableDataset(IterableDataset)` with `__init__`, `__iter__`
- [ ] Read CSV line-by-line — never load full file into memory

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week02.iterable_dataset import ComplaintIterableDataset
# ds = ComplaintIterableDataset('path/to/complaints.csv')
# x, y = next(iter(ds))
# print(f'x: {x.shape}, y: {y.shape}')
print('Provide your CSV path to test')
"
```
Expected: Valid `(input, label)` tuple from iterator

### Notes


---

## Step 3 — DataLoader with pin_memory and num_workers
**File:** `week02/dataloader.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `dataloader.py`
- [ ] Implement `create_dataloaders(train_dataset, val_dataset, batch_size=64)`
- [ ] Train: `shuffle=True, num_workers=4, pin_memory=True, drop_last=True`
- [ ] Val: `shuffle=False, drop_last=False`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week02.dataset import ComplaintDataset
from quantedge_services_handson.week02.dataloader import create_dataloaders
import torch
ds = ComplaintDataset(['text']*400, ['label']*400, max_len=32)
tr, va = torch.utils.data.random_split(ds, [320, 80])
train_loader, val_loader = create_dataloaders(tr, va, batch_size=32)
bx, by = next(iter(train_loader))
print(f'Batch: x={bx.shape}, y={by.shape}')
"
```
Expected: `Batch: x=torch.Size([32, 32]), y=torch.Size([32])`

### Notes


---

## Step 4 — Simple MLP Classifier
**File:** `week02/classifier.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `classifier.py`
- [ ] Implement `ComplaintClassifier(nn.Module)`
- [ ] Embedding → mean pool → Linear → ReLU → Linear → logits

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week02.classifier import ComplaintClassifier
import torch
model = ComplaintClassifier(vocab_size=256, num_classes=2)
out = model(torch.randint(0, 256, (32, 128)))
print(f'Output: {out.shape}')
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')
"
```
Expected: `Output: torch.Size([32, 2])`

### Notes


---

## Step 5 — Training Loop with Eval
**File:** `week02/training_loop.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `training_loop.py`
- [ ] Implement `set_all_seeds(seed=42)` — torch, cuda, numpy, random, cudnn
- [ ] Implement `train(model, train_loader, val_loader, epochs, lr, checkpoint_dir)`
- [ ] Handle class imbalance with weighted CrossEntropyLoss
- [ ] Train loop: forward → loss → backward → step
- [ ] Eval loop: model.eval(), no_grad, accuracy
- [ ] Save best + latest checkpoint each epoch

### Checkpoint
```bash
# Run training twice with same seed — losses must be identical
python -c "
from quantedge_services_handson.week02.training_loop import set_all_seeds, train
from quantedge_services_handson.week02.classifier import ComplaintClassifier
from quantedge_services_handson.week02.dataset import ComplaintDataset
from quantedge_services_handson.week02.dataloader import create_dataloaders
import torch

set_all_seeds(42)
ds = ComplaintDataset(['bad']*300+['good']*300, ['c']*300+['p']*300, max_len=32)
tr, va = torch.utils.data.random_split(ds, [480, 120])
tl, vl = create_dataloaders(tr, va, batch_size=32)
model = ComplaintClassifier(vocab_size=256, num_classes=2)
r = train(model, tl, vl, epochs=3)
print('Losses:', [f'{l:.4f}' for l in r['train_losses']])
print('Accuracies:', [f'{a:.4f}' for a in r['val_accuracies']])
"
```
Expected: Decreasing loss, increasing accuracy

### Notes


---

## Step 6 — Resume from Checkpoint
**File:** `week02/checkpointing.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `checkpointing.py`
- [ ] Implement `resume_from_checkpoint(model, optimizer, checkpoint_path) -> int`
- [ ] Load model state dict, optimizer state dict, return epoch number

### Checkpoint
```bash
# Train 2 epochs, stop, resume, train 2 more — loss should continue smoothly
```
Expected: Loss continues from saved state, no reset

### Notes


---

## Week 02 Summary
**All Steps Passed:** `No`
**Key Learnings:**

**Commit SHA:**

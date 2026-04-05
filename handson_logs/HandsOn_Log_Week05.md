# Hands-On Log — Week 05: Attention from Scratch

**Package:** `src/quantedge_services_handson/week05/`
**Goal:** Transformer encoder built entirely from scratch — every line understood

---

## Step 1 — Scaled Dot-Product Attention
**File:** `week05/attention.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `attention.py`
- [ ] Implement `scaled_dot_product_attention(Q, K, V, mask=None) -> (output, weights)`
- [ ] Use `torch.matmul` only — no `nn.functional`
- [ ] scores = Q @ K^T / sqrt(d_k) → mask → softmax → @ V

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week05.attention import scaled_dot_product_attention
import torch
Q = K = V = torch.randn(2, 8, 50, 16)
out, w = scaled_dot_product_attention(Q, K, V)
print(f'Output: {out.shape}')
print(f'Weights sum to 1: {torch.allclose(w.sum(dim=-1), torch.ones(2, 8, 50), atol=1e-5)}')
"
```
Expected: Shape preserved, weights sum to 1

### Notes


---

## Step 2 — Multi-Head Attention
**File:** `week05/multi_head.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `multi_head.py`
- [ ] Implement `MultiHeadAttention(nn.Module)` — d_model=128, num_heads=8
- [ ] Create W_Q, W_K, W_V, W_O as `nn.Linear`
- [ ] Project → reshape to heads → attention → reshape back → W_O
- [ ] **NO `nn.MultiheadAttention` allowed**

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week05.multi_head import MultiHeadAttention
import torch
mha = MultiHeadAttention(d_model=128, num_heads=8)
x = torch.randn(2, 50, 128)
out, weights = mha(x)
print(f'Output: {out.shape}')  # [2, 50, 128]
print(f'Weights: {weights.shape}')  # [2, 8, 50, 50]
"
```
Expected: Input shape preserved

### Notes


---

## Step 3 — Sinusoidal Positional Encoding
**File:** `week05/positional_encoding.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `positional_encoding.py`
- [ ] Implement `SinusoidalPositionalEncoding(nn.Module)`
- [ ] Precompute PE: `sin` on even dims, `cos` on odd dims
- [ ] Register as buffer (not parameter)
- [ ] forward: return `x + PE[:seq_len]`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week05.positional_encoding import SinusoidalPositionalEncoding
import torch
pe = SinusoidalPositionalEncoding(d_model=128)
x = torch.zeros(1, 10, 128)
out = pe(x)
print(f'PE[0,0,:4] = {out[0, 0, :4].tolist()}')  # should be [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
"
```
Expected: Alternating 0, 1, 0, 1 at position 0

### Notes


---

## Step 4 — Transformer Block (Pre-Norm)
**File:** `week05/transformer_block.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `transformer_block.py`
- [ ] Implement `TransformerBlock(nn.Module)` with Pre-Norm architecture
- [ ] norm1 → MHA → dropout → residual → norm2 → FFN → dropout → residual
- [ ] FFN: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week05.transformer_block import TransformerBlock
import torch
block = TransformerBlock(d_model=128, num_heads=8, d_ff=512)
x = torch.randn(2, 50, 128)
print(f'In: {x.shape} → Out: {block(x).shape}')
"
```
Expected: Shape preserved — `[2, 50, 128]`

### Notes


---

## Step 5 — Full 4-Layer Encoder
**File:** `week05/encoder.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `encoder.py`
- [ ] Implement `TransformerEncoder(nn.Module)`
- [ ] Input projection (21 → d_model) → PE → 4x TransformerBlock → LayerNorm → mean pool → Linear(d_model, 1)
- [ ] Input: `(batch, seq_len, 21)` sensors → Output: `(batch, 1)` RUL

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week05.encoder import TransformerEncoder
import torch
model = TransformerEncoder(input_dim=21, d_model=128, num_heads=8, num_layers=4)
x = torch.randn(32, 50, 21)
out = model(x)
print(f'Output: {out.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```
Expected: `Output: torch.Size([32, 1])`

### Notes


---

## Week 05 Summary
**All Steps Passed:** `No`
**Total Parameters:** _____
**Key Learnings:**

**Commit SHA:**

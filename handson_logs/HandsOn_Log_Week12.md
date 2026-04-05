# Hands-On Log — Week 12: Drift Detection, Monitoring & ADR

**Package:** `src/quantedge_services_handson/week12/`
**Goal:** Full system: train → version → deploy → monitor — documented as 6 ADRs

---

## Step 1 — Data Drift Detection (PSI)
**File:** `week12/psi.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `psi.py`
- [ ] Implement `compute_psi(reference, current, num_buckets=10) -> float`
- [ ] Equal-frequency bins from reference, compute bucket percentages
- [ ] PSI = sum((curr_pct - ref_pct) * ln(curr_pct / ref_pct))
- [ ] Handle zero bins with epsilon=1e-4

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week12.psi import compute_psi
import numpy as np
ref = np.random.normal(0, 1, 10000)
same = np.random.normal(0, 1, 10000)
shifted = np.random.normal(2, 1, 10000)
print(f'Same dist PSI: {compute_psi(ref, same):.4f}')      # < 0.1
print(f'Shifted PSI:   {compute_psi(ref, shifted):.4f}')    # > 0.25
"
```
- Same dist PSI: _____ (expected < 0.1)
- Shifted PSI: _____ (expected > 0.25)

### Notes


---

## Step 2 — KS-Test per Feature
**File:** `week12/ks_drift.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `ks_drift.py`
- [ ] Implement `detect_drift_per_feature(ref_df, curr_df, feature_columns, alpha=0.05) -> dict`
- [ ] KS-test per feature, record statistic, p_value, drifted flag, PSI

### Checkpoint
- Artificially shifted feature flagged as drifted: Yes/No
- Stable features not flagged: Yes/No

### Notes


---

## Step 3 — Concept Drift Detection
**File:** `week12/concept_drift.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `concept_drift.py`
- [ ] Implement `detect_concept_drift(ref_predictions, curr_predictions, threshold=0.1) -> dict`
- [ ] Compute mean_shift, std_shift (relative), KS-test on predictions

### Checkpoint
- Same predictions → drifted=False: Yes/No
- Shifted predictions → drifted=True: Yes/No

### Notes


---

## Step 4 — Prometheus Metrics Exporter
**File:** `week12/prometheus_metrics.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `prometheus_metrics.py`
- [ ] Implement `generate_prometheus_metrics(job_stats, inference_stats) -> str`
- [ ] Output valid Prometheus exposition format text

### Checkpoint
- Valid Prometheus text format: Yes/No
- Includes: requests_total, latency quantiles, drift PSI, GPU metrics: Yes/No

### Notes


---

## Step 5 — Audit Logging
**File:** `week12/audit_logger.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `audit_logger.py`
- [ ] Implement `AuditLogger` class: `__init__`, `log(event_type, details)`, `query(event_type, start_time)`
- [ ] JSONL format: one JSON line per event with timestamp, event, user, details

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week12.audit_logger import AuditLogger
logger = AuditLogger('/tmp/test_audit.jsonl')
logger.log('model_deployed', {'model': 'rul-v1', 'stage': 'production'})
logger.log('drift_detected', {'feature': 'sensor3', 'psi': 0.32})
entries = logger.query(event_type='drift_detected')
print(f'Drift events: {len(entries)}')
"
```
Expected: `Drift events: 1`

### Notes


---

## Step 6 — Architecture Decision Records
**File:** `week12/adr_generator.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `adr_generator.py`
- [ ] Implement `generate_adr(adr_id, title, context, decision, consequences, status, save_dir)`
- [ ] Implement `generate_all_adrs()` — write 6 ADRs from YOUR experience:
  - ADR-001: Use PyTorch over TensorFlow
  - ADR-002: QLoRA for Fine-Tuning on Consumer Hardware
  - ADR-003: ONNX as Primary Export Format
  - ADR-004: Pre-Norm Transformer Architecture
  - ADR-005: PSI + KS-Test for Drift Detection
  - ADR-006: MLflow for Experiment Tracking

### Checkpoint
- 6 ADR files in `docs/adr/`: Yes/No
- Each reflects YOUR actual experience: Yes/No

### Notes


---

## Week 12 Summary
**All Steps Passed:** `No`
**ADRs Written:** _____/6
**Key Learnings:**

**Commit SHA:**

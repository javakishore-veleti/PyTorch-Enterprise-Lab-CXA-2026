# Hands-On Log — Week 11: Experiment Tracking & Canary Deployment

**Package:** `src/quantedge_services_handson/week11/`
**Goal:** MLflow dashboard with all experiments — rollback runbook defensible

---

## Step 1 — MLflow Setup
**File:** `week11/mlflow_setup.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `mlflow_setup.py`
- [ ] Implement `setup_mlflow(tracking_uri="sqlite:///mlflow.db")`
- [ ] Set tracking URI, create experiment "quantedge-pytorch-mastery"

### Checkpoint
```bash
mlflow ui --port 5000
# Open http://localhost:5000 — experiment visible
```
Experiment visible in MLflow UI: Yes/No

### Notes


---

## Step 2 — Log All Experiments
**File:** `week11/experiment_logger.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `experiment_logger.py`
- [ ] Implement `log_experiment(name, params, metrics, model_path, tags) -> run_id`
- [ ] Implement `log_all_experiments()` — log weeks 1-10 with YOUR actual numbers

### Checkpoint
- Experiments logged in MLflow: _____ (target: 6+)
- All have real metrics from your runs: Yes/No

### Notes


---

## Step 3 — Model Registry
**File:** `week11/model_registry.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `model_registry.py`
- [ ] Implement `register_model(run_id, model_name, stage="Staging")`
- [ ] Register from run, transition to stage, add description

### Checkpoint
- Model in MLflow Models tab: Yes/No
- Version and stage shown: Yes/No

### Notes


---

## Step 4 — Model Card
**File:** `week11/model_card.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `model_card.py`
- [ ] Implement `write_model_card(save_path)` → `docs/week11_model_card.md`
- [ ] Sections: Details, Intended Use, Training Data, Results, Limitations, Ethics

### Checkpoint
- Model card written with YOUR results: Yes/No

### Notes


---

## Step 5 — Canary Deployment & Rollback
**File:** `week11/canary_strategy.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `canary_strategy.py`
- [ ] Implement `define_canary_strategy() -> dict`
  - canary_percentage, promotion_criteria, rollback_triggers, promotion_steps
- [ ] Implement `write_rollback_runbook(save_path)` → `docs/week11_rollback_runbook.md`
  - When to rollback, how, who to notify, post-mortem template

### Checkpoint
- Canary config defined: Yes/No
- Rollback runbook an ops engineer could follow at 3 AM: Yes/No

### Notes


---

## Week 11 Summary
**All Steps Passed:** `No`
**Experiments Tracked:** _____
**Key Learnings:**

**Commit SHA:**

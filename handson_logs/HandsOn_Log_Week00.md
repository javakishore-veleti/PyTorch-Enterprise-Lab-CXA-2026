# Hands-On Log — Week 00: Dataset Setup & Validation

**Package:** `src/quantedge_services_handson/week00/`
**Goal:** Every dataset needed for Weeks 1-12 downloaded, validated, and ready — one command runs all

---

## Data Inventory

| Dataset | Source | Size | Used In | Target Path | Currently Present |
|---------|--------|------|---------|-------------|-------------------|
| EUR/USD Forex Tick Data | Kaggle (HistData) | ~8 GB raw, ~17KB x 9 processed | Weeks 1, 3-4 (neural nets) | `data/forex/processed/*.parquet` | YES (9 parquet files) |
| CFPB Consumer Complaints | HuggingFace `cfpb/consumer-finance-complaints` | ~2 GB | Week 2 | `data/cfpb/` | NO (only .gitkeep) |
| CIC IoT 2023 | Kaggle | ~10 GB | Weeks 3, 4 | `data/cic_iot/` | NO |
| NASA CMAPSS (FD001) | Kaggle | ~50 MB | Weeks 5, 6, 9 | `data/cmapss/` | NO |
| OpenAssistant oasst1 | HuggingFace | ~1 GB | Week 7 | `data/oasst1/` | NO |
| StackOverflow Java/ES | Kaggle | ~2 GB filtered | Week 8 | `data/stackoverflow/` | NO |

---

## Package Structure

```
week00/
    __init__.py
    config.py              — data paths, URLs, expected file counts
    forex_data.py          — EUR/USD forex parquet setup
    cfpb_data.py           — CFPB complaints from HuggingFace
    cic_iot_data.py        — CIC IoT 2023 from Kaggle
    cmapss_data.py         — NASA CMAPSS engine degradation
    oasst1_data.py         — OpenAssistant instruction dataset
    stackoverflow_data.py  — StackOverflow Java/ES Q&A
    __main__.py            — orchestrator: run all, skip if present
```

---

## Step 1 — Config Module
**File:** `week00/config.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `config.py`
- [ ] Define a dataclass or dict per dataset:
  - `name` — human label
  - `target_dir` — Path where data lives (e.g. `data/forex/processed`)
  - `check_glob` — glob pattern to verify presence (e.g. `*.parquet`)
  - `min_files` — minimum file count to consider it "present"
  - `source` — where to download from (`kaggle`, `huggingface`, `url`)
  - `source_id` — dataset identifier (e.g. `cfpb/consumer-finance-complaints`)
- [ ] Define `ALL_DATASETS` list containing all 6 configs
- [ ] Implement `is_present(dataset_config) -> bool` — checks if `target_dir` has >= `min_files` matching `check_glob`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.config import ALL_DATASETS, is_present
for ds in ALL_DATASETS:
    print(f'{ds.name:30s} present={is_present(ds)}')
"
```
Expected: Forex shows `True`, rest show `False`

### Notes


---

## Step 2 — Forex Data Setup
**File:** `week00/forex_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `forex_data.py`
- [ ] Implement `setup_forex(config) -> dict`
- [ ] **Skip logic:** if `is_present(config)` → print "Forex: already present, skipping" → return status dict
- [ ] **If missing:** Generate synthetic OHLCV data as fallback (since real HistData is 8GB and needs Kaggle auth):
  - Create 9 parquet files with ~200 rows each
  - Columns: `open, high, low, close, volume, spread` with datetime index
  - Use realistic price ranges (EUR/USD ~1.05-1.15)
  - Save to `data/forex/processed/`
- [ ] Return: `{'name': 'forex', 'status': 'skipped'|'created', 'files': int, 'rows': int}`

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.forex_data import setup_forex
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_forex(ALL_DATASETS[0])
print(result)
"
```
Expected: `{'name': 'forex', 'status': 'skipped', 'files': 9, 'rows': 1800}`

### Notes
Data is already present from the reference implementation. This step should skip.

---

## Step 3 — CFPB Complaints Setup
**File:** `week00/cfpb_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `cfpb_data.py`
- [ ] Implement `setup_cfpb(config) -> dict`
- [ ] **Skip logic:** same pattern — check `is_present`, skip if yes
- [ ] **If missing:** Two options (try in order):
  - **Option A:** `datasets.load_dataset("cfpb/consumer-finance-complaints")` → save as parquet
  - **Option B (fallback):** Generate synthetic complaints data:
    - 1000 rows, columns: `complaint_narrative`, `product`, `company`, `state`, `date_received`
    - Products: "Credit card", "Mortgage", "Student loan", "Checking/savings", "Debt collection"
    - Random short complaint texts
    - Save as `data/cfpb/complaints.parquet`
- [ ] Return status dict

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.cfpb_data import setup_cfpb
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_cfpb(ALL_DATASETS[1])
print(result)
"
```
Expected: Status `created` with row/file counts

### Notes
HuggingFace download may need `huggingface-cli login` first. Synthetic fallback ensures Week 2 is never blocked.

---

## Step 4 — CIC IoT 2023 Setup
**File:** `week00/cic_iot_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `cic_iot_data.py`
- [ ] Implement `setup_cic_iot(config) -> dict`
- [ ] **Skip logic:** same pattern
- [ ] **If missing:** Generate synthetic IoT network traffic data:
  - 5000 rows, 46 numeric feature columns (flow_duration, packet_length_mean, etc.)
  - 1 label column with 35 classes (normal + 34 attack types)
  - Save as `data/cic_iot/iot_traffic.parquet`
- [ ] Real data is 10 GB — synthetic is fine for learning AMP/profiling concepts
- [ ] Return status dict

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.cic_iot_data import setup_cic_iot
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_cic_iot(ALL_DATASETS[2])
print(result)
"
```
Expected: Status `created`, 5000 rows, 47 columns

### Notes
The real Kaggle dataset is huge. Synthetic data keeps Weeks 3-4 unblocked. If you want real data later: `kaggle datasets download -d dhoogla/ciciot2023`.

---

## Step 5 — NASA CMAPSS Setup
**File:** `week00/cmapss_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `cmapss_data.py`
- [ ] Implement `setup_cmapss(config) -> dict`
- [ ] **Skip logic:** same pattern
- [ ] **If missing:** This dataset is small (~50 MB). Two options:
  - **Option A:** Download from Kaggle: `kaggle datasets download -d behrad3d/nasa-cmaps`
  - **Option B (fallback):** Generate synthetic CMAPSS-like data:
    - 100 engine units, each with 50-300 cycles
    - 3 operational settings + 21 sensor columns
    - RUL computed as max_cycle - current_cycle
    - Save as `data/cmapss/train_FD001.txt` (space-separated, no header, matching real format)
- [ ] Return status dict

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.cmapss_data import setup_cmapss
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_cmapss(ALL_DATASETS[3])
print(result)
"
```
Expected: Status `created`, file `train_FD001.txt` present

### Notes
This is the most critical dataset — used in Weeks 5, 6, and 9. The format is: `unit_id cycle op1 op2 op3 sensor1..sensor21` (space-separated, no header).

---

## Step 6 — OpenAssistant oasst1 Setup
**File:** `week00/oasst1_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `oasst1_data.py`
- [ ] Implement `setup_oasst1(config) -> dict`
- [ ] **Skip logic:** same pattern
- [ ] **If missing:**
  - **Option A:** `datasets.load_dataset("OpenAssistant/oasst1")` → filter English → save as parquet
  - **Option B (fallback):** Generate synthetic instruction-response pairs:
    - 500 rows with `instruction` and `response` columns
    - Topics: coding, math, general knowledge
    - Save as `data/oasst1/oasst1_en.parquet`
- [ ] Return status dict

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.oasst1_data import setup_oasst1
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_oasst1(ALL_DATASETS[4])
print(result)
"
```
Expected: Status `created`, instruction-response pairs saved

### Notes
Only needed for Week 7 (LoRA). Synthetic is fine for learning the LoRA pipeline mechanics.

---

## Step 7 — StackOverflow Setup
**File:** `week00/stackoverflow_data.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `stackoverflow_data.py`
- [ ] Implement `setup_stackoverflow(config) -> dict`
- [ ] **Skip logic:** same pattern
- [ ] **If missing:**
  - **Option A:** Download from Kaggle and filter by tags (java, spring-boot, elasticsearch)
  - **Option B (fallback):** Generate synthetic Q&A pairs:
    - 500 rows with `question`, `answer`, `tags`, `score` columns
    - Domain: Java, Spring Boot, Elasticsearch
    - Save as `data/stackoverflow/so_java_es.parquet`
- [ ] Return status dict

### Checkpoint
```bash
python -c "
from quantedge_services_handson.week00.stackoverflow_data import setup_stackoverflow
from quantedge_services_handson.week00.config import ALL_DATASETS
result = setup_stackoverflow(ALL_DATASETS[5])
print(result)
"
```
Expected: Status `created`, Q&A pairs saved

### Notes
Only needed for Week 8. Synthetic gets you through the domain adaptation pipeline.

---

## Step 8 — Orchestrator `__main__.py`
**File:** `week00/__main__.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `__main__.py`
- [ ] Import all 6 setup functions and `ALL_DATASETS` from config
- [ ] Map each config to its setup function
- [ ] Run all 6 in order, collect results
- [ ] Print a final summary table:
  ```
  Dataset                  Status    Files  Rows
  ─────────────────────────────────────────────
  EUR/USD Forex            skipped   9      1800
  CFPB Complaints          created   1      1000
  CIC IoT 2023             created   1      5000
  NASA CMAPSS              created   1      20631
  OpenAssistant oasst1     created   1      500
  StackOverflow Java/ES    created   1      500
  ```
- [ ] Support `--dataset forex` flag to run just one
- [ ] Support `--force` flag to re-generate even if present

### Checkpoint
```bash
# Run all
python -m quantedge_services_handson.week00

# Run just one
python -m quantedge_services_handson.week00 --dataset cmapss

# Force regenerate
python -m quantedge_services_handson.week00 --dataset forex --force
```

### Notes


---

## Step 9 — Update `week00/__init__.py`
**File:** `week00/__init__.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Add module docstring listing all submodules
- [ ] Add convenience import:
  ```python
  from .config import ALL_DATASETS, is_present
  ```

### Checkpoint
```bash
python -c "from quantedge_services_handson.week00 import ALL_DATASETS; print(len(ALL_DATASETS))"
```
Expected: `6`

### Notes


---

## Design Rules

1. **Every `.py` file follows the same pattern:** check if present → skip or create → return status dict
2. **Synthetic fallback always exists** — no week is ever blocked by missing Kaggle auth or slow downloads
3. **All data lands under `data/`** in the project root — consistent with the reference implementation
4. **Parquet is the default format** — except CMAPSS which uses the original space-separated `.txt` to match NASA's format
5. **`__main__.py` is idempotent** — safe to run multiple times, only creates what's missing

---

## Week 00 Summary
**All Steps Passed:** `No`
**Datasets Ready:** ___ /6
**Key Learnings:**

**Commit SHA:**

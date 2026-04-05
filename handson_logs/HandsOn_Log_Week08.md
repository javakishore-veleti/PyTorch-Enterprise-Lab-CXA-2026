# Hands-On Log — Week 08: Domain Adaptation & Ollama Serving

**Package:** `src/quantedge_services_handson/week08/`
**Goal:** Domain-adapted Mistral-7B running locally via Ollama

---

## Step 1 — Filter StackOverflow Data
**File:** `week08/data_filter.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `data_filter.py`
- [ ] Implement `filter_stackoverflow_data(csv_path, tags, min_score) -> DataFrame`
- [ ] Filter by tags (java, spring-boot, elasticsearch), score >= 5
- [ ] Create Q&A pairs, clean HTML

### Checkpoint
- Q&A pairs extracted: _____ (target: 5K-50K)

### Notes


---

## Step 2 — Fine-Tune on Domain Data
**File:** `week08/domain_finetuner.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `domain_finetuner.py`
- [ ] Implement `fine_tune_domain(model, tokenizer, qa_pairs, output_dir)`
- [ ] Format as `[INST] {question} [/INST] {answer}`, tokenize, train with QLoRA

### Checkpoint
- Training completed: Yes/No
- Adapter saved: Yes/No

### Notes


---

## Step 3 — Merge LoRA into Base Model
**File:** `week08/adapter_merger.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `adapter_merger.py`
- [ ] Implement `merge_adapter(base_model_name, adapter_path, output_path)`
- [ ] Load base + adapter → merge_and_unload → save merged + tokenizer

### Checkpoint
- Merged model size: _____ GB (should be 14+)

### Notes


---

## Step 4 — Package for Ollama
**File:** `week08/ollama_packager.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `ollama_packager.py`
- [ ] Implement `create_ollama_modelfile(model_path, output_path)`
- [ ] Write Modelfile with FROM, PARAMETER, SYSTEM prompt
- [ ] Run: `ollama create domain-mistral -f Modelfile`

### Checkpoint
- Modelfile created: Yes/No
- `ollama run domain-mistral` works: Yes/No

### Notes


---

## Step 5 — Test with Enterprise Questions
**File:** `week08/domain_tester.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `domain_tester.py`
- [ ] Implement `test_domain_model(model_name)` with 10 enterprise questions
- [ ] Rate each response 1-5 manually

### Checkpoint
| # | Question (short) | Quality (1-5) |
|---|-----------------|--------------|
| 1 |                 |              |
| 2 |                 |              |
| 3 |                 |              |
| 4 |                 |              |
| 5 |                 |              |
| 6 |                 |              |
| 7 |                 |              |
| 8 |                 |              |
| 9 |                 |              |
| 10|                 |              |

Average score: _____ /5 (target: 7/10 responses accurate)

### Notes


---

## Week 08 Summary
**All Steps Passed:** `No`
**Domain Quality Score:** ___/5
**Key Learnings:**

**Commit SHA:**

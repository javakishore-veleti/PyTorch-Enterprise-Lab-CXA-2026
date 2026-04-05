# Hands-On Log — Week 10: Quantization, FastAPI Serving & Docker

**Package:** `src/quantedge_services_handson/week10/`
**Goal:** Production inference service — Dockerized, P50/P99 latency documented

---

## Step 1 — INT8 Quantization
**File:** `week10/quantization.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `quantization.py`
- [ ] Implement `quantize_model(onnx_path, output_path) -> dict`
- [ ] `quantize_dynamic` with QInt8
- [ ] Compare: file sizes, inference outputs, accuracy delta

### Checkpoint
- Original size: _____ MB
- Quantized size: _____ MB
- Accuracy drop: _____% (should be < 1%)

### Notes


---

## Step 2 — FastAPI Endpoint
**File:** `week10/serving.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `serving.py`
- [ ] Define `PredictionRequest` (Pydantic) — validate seq_len 10-200, features=21
- [ ] Define `PredictionResponse` — rul_prediction, confidence, model_version, correlation_id
- [ ] Implement `POST /predict` — validate → numpy → ONNX inference → response

### Checkpoint
```bash
uvicorn quantedge_services_handson.week10.serving:app --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sensor_readings": [...]}'
```
Returns JSON response: Yes/No

### Notes


---

## Step 3 — Middleware (Timeout, Logging, Correlation ID)
**File:** `week10/middleware.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `middleware.py`
- [ ] Implement `CorrelationMiddleware` — extract/generate X-Correlation-ID
- [ ] Implement `TimeoutMiddleware` — 30s timeout, 504 on breach
- [ ] Implement `setup_logging()` — structlog JSON with timestamp, correlation_id, latency

### Checkpoint
- Correlation ID in response headers: Yes/No
- Structured JSON logs: Yes/No

### Notes


---

## Step 4 — Dockerfile
**File:** `week10/Dockerfile`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `Dockerfile`
- [ ] Base: python:3.11-slim, install deps, copy model+code
- [ ] Expose 8000, health check, CMD uvicorn

### Checkpoint
```bash
docker build -t quantedge-rul -f src/quantedge_services_handson/week10/Dockerfile .
docker run -p 8000:8000 quantedge-rul
```
Container runs and serves requests: Yes/No

### Notes


---

## Step 5 — Load Test & Latency Report
**File:** `week10/load_test.py`
**Status:** `Not Started`
**Started:**
**Completed:**

### What to do
- [ ] Create `load_test.py`
- [ ] Implement `load_test(url, num_requests=100)`
- [ ] Send concurrent requests with httpx.AsyncClient
- [ ] Compute P50, P95, P99, throughput, error rate

### Checkpoint

| Metric          | Value |
|-----------------|-------|
| P50 latency     |    ms |
| P95 latency     |    ms |
| P99 latency     |    ms |
| Throughput      |  req/s|
| Error rate      |     % |

Targets: P50 < 50ms, P99 < 200ms, 0% errors

### Notes


---

## Week 10 Summary
**All Steps Passed:** `No`
**P99 Latency:** _____ ms
**Key Learnings:**

**Commit SHA:**

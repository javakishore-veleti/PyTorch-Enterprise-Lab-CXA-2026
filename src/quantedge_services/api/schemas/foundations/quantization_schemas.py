from pydantic import BaseModel


class QuantizeStaticRequest(BaseModel):
    output_dir: str
    calibration_batches: int = 10
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30


class QuantizeStaticResult(BaseModel):
    output_path: str
    original_size_mb: float
    quantized_size_mb: float
    size_reduction_pct: float
    status: str


class QuantizeDynamicRequest(BaseModel):
    output_dir: str
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30


class QuantizeDynamicResult(BaseModel):
    output_path: str
    original_size_mb: float
    quantized_size_mb: float
    size_reduction_pct: float
    status: str


class QuantizeQATRequest(BaseModel):
    output_dir: str
    train_steps: int = 10
    learning_rate: float = 1e-4
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30
    batch_size: int = 4


class QuantizeQATResult(BaseModel):
    output_path: str
    final_loss: float
    original_size_mb: float
    quantized_size_mb: float
    size_reduction_pct: float
    status: str


class QuantizeCompareRequest(BaseModel):
    output_dir: str
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30
    num_runs: int = 50
    batch_size: int = 4


class QuantizeCompareResult(BaseModel):
    fp32_latency_ms: float
    dynamic_latency_ms: float
    dynamic_size_mb: float
    fp32_size_mb: float
    speedup_ratio: float
    status: str


class ModelInferRequest(BaseModel):
    model_path: str
    model_format: str  # "torchscript", "quantized_dynamic", "eager"
    input_size: int = 14
    seq_len: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4


class ModelInferResult(BaseModel):
    output_value: float
    latency_ms: float
    model_format: str
    status: str


class ServingBenchmarkRequest(BaseModel):
    model_path: str
    model_format: str
    input_size: int = 14
    seq_len: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    num_requests: int = 200
    batch_size: int = 1


class ServingBenchmarkResult(BaseModel):
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    throughput_qps: float
    status: str

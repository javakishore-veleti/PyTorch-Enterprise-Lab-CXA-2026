"""Export Format DTOs — Week 9 (TorchScript, ONNX, TensorRT, Benchmark)."""
from __future__ import annotations
from pydantic import BaseModel


class TorchScriptExportRequest(BaseModel):
    model_checkpoint_path: str
    output_dir: str
    export_mode: str = "trace"
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30


class TorchScriptExportResult(BaseModel):
    output_path: str
    export_mode: str
    validation_passed: bool
    status: str


class ONNXExportRequest(BaseModel):
    model_checkpoint_path: str
    output_dir: str
    input_size: int = 14
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    seq_len: int = 30
    opset_version: int = 17
    dynamic_batch: bool = True


class ONNXExportResult(BaseModel):
    output_path: str
    opset_version: int
    dynamic_batch: bool
    status: str


class ONNXValidateRequest(BaseModel):
    onnx_path: str
    input_size: int = 14
    seq_len: int = 30
    batch_size: int = 4


class ONNXValidateResult(BaseModel):
    max_abs_diff: float
    passed: bool
    status: str


class TensorRTExportRequest(BaseModel):
    torchscript_path: str
    output_dir: str
    precision: str = "fp16"
    input_size: int = 14
    seq_len: int = 30
    batch_size: int = 4


class TensorRTExportResult(BaseModel):
    output_path: str
    precision: str
    status: str


class BenchmarkRequest(BaseModel):
    eager_checkpoint_path: str
    torchscript_path: str
    onnx_path: str
    input_size: int = 14
    seq_len: int = 30
    batch_size: int = 4
    num_runs: int = 100
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4


class BenchmarkResult(BaseModel):
    eager_latency_ms: float
    torchscript_latency_ms: float
    onnxruntime_latency_ms: float
    eager_throughput_qps: float
    torchscript_throughput_qps: float
    onnxruntime_throughput_qps: float
    status: str

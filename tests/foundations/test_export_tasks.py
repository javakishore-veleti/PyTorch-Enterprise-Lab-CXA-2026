"""Week 9 — Export task tests (TorchScript, ONNX, TensorRT, Benchmark, Schemas)."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from quantedge_services.api.schemas.foundations.export_schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    ONNXExportRequest,
    ONNXValidateRequest,
    TensorRTExportRequest,
    TorchScriptExportRequest,
)
from quantedge_services.services.wfs.model_export.tasks.torchscript_export_task import TorchScriptExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_export_task import ONNXExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_validate_task import ONNXValidateTask
from quantedge_services.services.wfs.model_export.tasks.tensorrt_export_task import TensorRTExportTask
from quantedge_services.services.wfs.model_export.tasks.benchmark_task import BenchmarkTask


# ── TorchScript tests ──────────────────────────────────────────────────────────

def test_torchscript_export_task_trace_mode_creates_file(tmp_path):
    task = TorchScriptExportTask()
    req = TorchScriptExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        export_mode="trace",
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
    )
    result = task.execute(req)
    assert os.path.exists(result.output_path)
    assert result.status == "success"


def test_torchscript_export_task_validation_passed(tmp_path):
    task = TorchScriptExportTask()
    req = TorchScriptExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        export_mode="trace",
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
    )
    result = task.execute(req)
    assert result.validation_passed is True


def test_torchscript_export_task_output_shape_correct(tmp_path):
    task = TorchScriptExportTask()
    req = TorchScriptExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        export_mode="trace",
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
    )
    result = task.execute(req)
    scripted = torch.jit.load(result.output_path)
    scripted.eval()
    example = torch.randn(4, 10, 14)
    with torch.no_grad():
        out, _ = scripted(example)
    assert out.shape == (4, 1)


# ── ONNX export tests ──────────────────────────────────────────────────────────

def test_onnx_export_task_creates_onnx_file(tmp_path):
    task = ONNXExportTask()
    req = ONNXExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
        opset_version=17,
        dynamic_batch=True,
    )
    result = task.execute(req)
    assert os.path.exists(result.output_path)
    assert result.output_path.endswith(".onnx")
    assert result.status == "success"


def test_onnx_export_task_dynamic_batch_true(tmp_path):
    task = ONNXExportTask()
    req = ONNXExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
        opset_version=17,
        dynamic_batch=True,
    )
    result = task.execute(req)
    assert result.dynamic_batch is True


def test_onnx_export_task_opset_version_in_result(tmp_path):
    task = ONNXExportTask()
    req = ONNXExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
        opset_version=17,
        dynamic_batch=False,
    )
    result = task.execute(req)
    assert result.opset_version == 17


# ── ONNX validate tests ────────────────────────────────────────────────────────

def test_onnx_validate_task_no_onnxruntime_returns_not_installed(tmp_path):
    task = ONNXValidateTask()
    req = ONNXValidateRequest(onnx_path=str(tmp_path / "model.onnx"))
    with patch.dict(sys.modules, {"onnxruntime": None}):
        result = task.execute(req)
    assert result.status == "onnxruntime_not_installed"
    assert result.passed is False
    assert result.max_abs_diff == -1.0


def test_onnx_validate_task_with_mock_ort(tmp_path):
    # First export an ONNX model
    export_task = ONNXExportTask()
    export_req = ONNXExportRequest(
        model_checkpoint_path="",
        output_dir=str(tmp_path),
        input_size=14,
        d_model=32,
        nhead=2,
        num_layers=2,
        seq_len=10,
        opset_version=17,
        dynamic_batch=True,
    )
    export_result = export_task.execute(export_req)
    assert os.path.exists(export_result.output_path)

    # Mock onnxruntime session
    import numpy as np
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = [np.zeros((4, 1), dtype=np.float32)]
    mock_ort.InferenceSession.return_value = mock_session

    validate_task = ONNXValidateTask()
    req = ONNXValidateRequest(
        onnx_path=export_result.output_path,
        input_size=14,
        seq_len=10,
        batch_size=4,
    )
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        result = validate_task.execute(req)
    assert result.status == "success"
    assert isinstance(result.max_abs_diff, float)


# ── TensorRT tests ─────────────────────────────────────────────────────────────

def test_tensorrt_export_task_no_torch_tensorrt_returns_not_installed(tmp_path):
    task = TensorRTExportTask()
    req = TensorRTExportRequest(
        torchscript_path=str(tmp_path / "model.pt"),
        output_dir=str(tmp_path),
        precision="fp16",
    )
    with patch.dict(sys.modules, {"torch_tensorrt": None}):
        result = task.execute(req)
    assert result.status == "torch_tensorrt_not_installed"
    assert result.output_path == ""
    assert result.precision == "fp16"


# ── Benchmark tests ────────────────────────────────────────────────────────────

def test_benchmark_task_returns_result_dto(tmp_path):
    task = BenchmarkTask()
    req = BenchmarkRequest(
        eager_checkpoint_path="",
        torchscript_path="",
        onnx_path="",
        input_size=14,
        seq_len=10,
        batch_size=2,
        num_runs=3,
        d_model=32,
        nhead=2,
        num_layers=2,
    )
    result = task.execute(req)
    assert isinstance(result, BenchmarkResult)


def test_benchmark_task_eager_latency_positive(tmp_path):
    task = BenchmarkTask()
    req = BenchmarkRequest(
        eager_checkpoint_path="",
        torchscript_path="",
        onnx_path="",
        input_size=14,
        seq_len=10,
        batch_size=2,
        num_runs=3,
        d_model=32,
        nhead=2,
        num_layers=2,
    )
    result = task.execute(req)
    assert result.eager_latency_ms > 0.0


def test_benchmark_task_throughput_positive(tmp_path):
    task = BenchmarkTask()
    req = BenchmarkRequest(
        eager_checkpoint_path="",
        torchscript_path="",
        onnx_path="",
        input_size=14,
        seq_len=10,
        batch_size=2,
        num_runs=3,
        d_model=32,
        nhead=2,
        num_layers=2,
    )
    result = task.execute(req)
    assert result.eager_throughput_qps > 0.0


def test_benchmark_task_no_onnxruntime_still_returns(tmp_path):
    task = BenchmarkTask()
    req = BenchmarkRequest(
        eager_checkpoint_path="",
        torchscript_path="",
        onnx_path="",
        input_size=14,
        seq_len=10,
        batch_size=2,
        num_runs=3,
        d_model=32,
        nhead=2,
        num_layers=2,
    )
    with patch.dict(sys.modules, {"onnxruntime": None}):
        result = task.execute(req)
    assert result.eager_latency_ms > 0.0
    assert result.onnxruntime_latency_ms == 0.0


# ── Schema tests ────────────────────────────────────────────────────────────────

def test_torchscript_export_request_defaults():
    req = TorchScriptExportRequest(model_checkpoint_path="", output_dir="/tmp")
    assert req.export_mode == "trace"
    assert req.input_size == 14
    assert req.d_model == 64
    assert req.seq_len == 30


def test_onnx_export_request_defaults():
    req = ONNXExportRequest(model_checkpoint_path="", output_dir="/tmp")
    assert req.opset_version == 17
    assert req.dynamic_batch is True
    assert req.input_size == 14


def test_benchmark_result_all_fields_present():
    result = BenchmarkResult(
        eager_latency_ms=1.0,
        torchscript_latency_ms=0.8,
        onnxruntime_latency_ms=0.9,
        eager_throughput_qps=4000.0,
        torchscript_throughput_qps=5000.0,
        onnxruntime_throughput_qps=4400.0,
        status="success",
    )
    assert result.eager_latency_ms == 1.0
    assert result.status == "success"
    assert hasattr(result, "torchscript_throughput_qps")
    assert hasattr(result, "onnxruntime_throughput_qps")

"""ModelExportService — async wrappers around export and benchmark tasks."""
from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.export_schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    ONNXExportRequest,
    ONNXExportResult,
    ONNXValidateRequest,
    ONNXValidateResult,
    TensorRTExportRequest,
    TensorRTExportResult,
    TorchScriptExportRequest,
    TorchScriptExportResult,
)
from quantedge_services.services.wfs.model_export.tasks.torchscript_export_task import TorchScriptExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_export_task import ONNXExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_validate_task import ONNXValidateTask
from quantedge_services.services.wfs.model_export.tasks.tensorrt_export_task import TensorRTExportTask
from quantedge_services.services.wfs.model_export.tasks.benchmark_task import BenchmarkTask


class ModelExportService:
    """Async service for model export workflows (TorchScript, ONNX, TensorRT, Benchmark)."""

    def __init__(
        self,
        torchscript_task: TorchScriptExportTask,
        onnx_export_task: ONNXExportTask,
        onnx_validate_task: ONNXValidateTask,
        tensorrt_task: TensorRTExportTask,
        benchmark_task: BenchmarkTask,
    ) -> None:
        self._torchscript = torchscript_task
        self._onnx_export = onnx_export_task
        self._onnx_validate = onnx_validate_task
        self._tensorrt = tensorrt_task
        self._benchmark = benchmark_task

    async def export_torchscript(self, request: TorchScriptExportRequest) -> TorchScriptExportResult:
        return await asyncio.to_thread(self._torchscript.execute, request)

    async def export_onnx(self, request: ONNXExportRequest) -> ONNXExportResult:
        return await asyncio.to_thread(self._onnx_export.execute, request)

    async def validate_onnx(self, request: ONNXValidateRequest) -> ONNXValidateResult:
        return await asyncio.to_thread(self._onnx_validate.execute, request)

    async def export_tensorrt(self, request: TensorRTExportRequest) -> TensorRTExportResult:
        return await asyncio.to_thread(self._tensorrt.execute, request)

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        return await asyncio.to_thread(self._benchmark.execute, request)

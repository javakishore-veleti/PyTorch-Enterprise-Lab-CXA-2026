import pytest
from unittest.mock import MagicMock


class TestDynamicQuantTask:
    def test_dynamic_quant_task_creates_file(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.dynamic_quant_task import DynamicQuantTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeDynamicRequest
        task = DynamicQuantTask()
        req = QuantizeDynamicRequest(output_dir=str(tmp_path))
        result = task.execute(req)
        import os
        assert os.path.exists(result.output_path)

    def test_dynamic_quant_result_is_dto(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.dynamic_quant_task import DynamicQuantTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeDynamicRequest, QuantizeDynamicResult
        task = DynamicQuantTask()
        req = QuantizeDynamicRequest(output_dir=str(tmp_path))
        result = task.execute(req)
        assert isinstance(result, QuantizeDynamicResult)

    def test_dynamic_quant_size_reduction_pct_nonnegative(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.dynamic_quant_task import DynamicQuantTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeDynamicRequest
        task = DynamicQuantTask()
        req = QuantizeDynamicRequest(output_dir=str(tmp_path))
        result = task.execute(req)
        assert isinstance(result.size_reduction_pct, float)


class TestStaticQuantTask:
    def test_static_quant_task_creates_file(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.static_quant_task import StaticQuantTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeStaticRequest
        task = StaticQuantTask()
        req = QuantizeStaticRequest(output_dir=str(tmp_path), calibration_batches=2)
        result = task.execute(req)
        import os
        assert os.path.exists(result.output_path)

    def test_static_quant_result_status_success(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.static_quant_task import StaticQuantTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeStaticRequest
        task = StaticQuantTask()
        req = QuantizeStaticRequest(output_dir=str(tmp_path), calibration_batches=2)
        result = task.execute(req)
        assert result.status == "success"


class TestQATTask:
    def test_qat_task_creates_file(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.qat_task import QATTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeQATRequest
        task = QATTask()
        req = QuantizeQATRequest(output_dir=str(tmp_path), train_steps=2)
        result = task.execute(req)
        import os
        assert os.path.exists(result.output_path)

    def test_qat_task_final_loss_is_float(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.qat_task import QATTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeQATRequest
        task = QATTask()
        req = QuantizeQATRequest(output_dir=str(tmp_path), train_steps=2)
        result = task.execute(req)
        assert isinstance(result.final_loss, float)

    def test_qat_task_result_is_dto(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.qat_task import QATTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeQATRequest, QuantizeQATResult
        task = QATTask()
        req = QuantizeQATRequest(output_dir=str(tmp_path), train_steps=2)
        result = task.execute(req)
        assert isinstance(result, QuantizeQATResult)


class TestQuantCompareTask:
    def test_quant_compare_task_returns_result(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.quant_compare_task import QuantCompareTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeCompareRequest, QuantizeCompareResult
        task = QuantCompareTask()
        req = QuantizeCompareRequest(output_dir=str(tmp_path), num_runs=5, batch_size=2)
        result = task.execute(req)
        assert isinstance(result, QuantizeCompareResult)

    def test_quant_compare_speedup_positive(self, tmp_path):
        from quantedge_services.services.wfs.quantization.tasks.quant_compare_task import QuantCompareTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeCompareRequest
        task = QuantCompareTask()
        req = QuantizeCompareRequest(output_dir=str(tmp_path), num_runs=5, batch_size=2)
        result = task.execute(req)
        assert result.speedup_ratio > 0


class TestSchemas:
    def test_quantize_static_request_defaults(self):
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeStaticRequest
        req = QuantizeStaticRequest(output_dir="/tmp")
        assert req.calibration_batches == 10
        assert req.input_size == 14

    def test_quantize_dynamic_request_defaults(self):
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeDynamicRequest
        req = QuantizeDynamicRequest(output_dir="/tmp")
        assert req.d_model == 64
        assert req.seq_len == 30

    def test_quantize_compare_result_fields(self):
        from quantedge_services.api.schemas.foundations.quantization_schemas import QuantizeCompareResult
        result = QuantizeCompareResult(
            fp32_latency_ms=10.0,
            dynamic_latency_ms=8.0,
            dynamic_size_mb=1.0,
            fp32_size_mb=2.0,
            speedup_ratio=1.25,
            status="success",
        )
        assert result.speedup_ratio == 1.25

    def test_model_infer_request_fields(self):
        from quantedge_services.api.schemas.foundations.quantization_schemas import ModelInferRequest
        req = ModelInferRequest(model_path="/tmp/model.pt", model_format="eager")
        assert req.model_format == "eager"
        assert req.input_size == 14

    def test_serving_benchmark_result_fields(self):
        from quantedge_services.api.schemas.foundations.quantization_schemas import ServingBenchmarkResult
        r = ServingBenchmarkResult(
            p50_latency_ms=1.0,
            p95_latency_ms=2.0,
            p99_latency_ms=3.0,
            mean_latency_ms=1.5,
            throughput_qps=100.0,
            status="success",
        )
        assert r.p99_latency_ms >= r.p50_latency_ms

    def test_quantization_service_has_methods(self):
        from quantedge_services.services.wfs.quantization.quantization_service import QuantizationService
        import inspect
        assert hasattr(QuantizationService, "quantize_static")
        assert hasattr(QuantizationService, "quantize_dynamic")
        assert hasattr(QuantizationService, "quantize_qat")
        assert hasattr(QuantizationService, "quantize_compare")

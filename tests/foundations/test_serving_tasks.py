import pytest


class TestModelInferTask:
    def test_model_infer_task_eager_mode(self, tmp_path):
        from quantedge_services.services.wfs.model_serving.tasks.model_infer_task import ModelInferTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import ModelInferRequest
        task = ModelInferTask()
        req = ModelInferRequest(
            model_path=str(tmp_path / "unused.pt"),
            model_format="eager",
        )
        result = task.execute(req)
        assert result.status == "success"

    def test_model_infer_task_latency_nonnegative(self, tmp_path):
        from quantedge_services.services.wfs.model_serving.tasks.model_infer_task import ModelInferTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import ModelInferRequest
        task = ModelInferTask()
        req = ModelInferRequest(
            model_path=str(tmp_path / "unused.pt"),
            model_format="eager",
        )
        result = task.execute(req)
        assert result.latency_ms >= 0


class TestServingBenchmarkTask:
    def test_serving_benchmark_task_returns_percentiles(self, tmp_path):
        from quantedge_services.services.wfs.model_serving.tasks.serving_benchmark_task import ServingBenchmarkTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import ServingBenchmarkRequest, ServingBenchmarkResult
        task = ServingBenchmarkTask()
        req = ServingBenchmarkRequest(
            model_path=str(tmp_path / "unused.pt"),
            model_format="eager",
            num_requests=10,
        )
        result = task.execute(req)
        assert isinstance(result, ServingBenchmarkResult)

    def test_serving_benchmark_p99_gte_p50(self, tmp_path):
        from quantedge_services.services.wfs.model_serving.tasks.serving_benchmark_task import ServingBenchmarkTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import ServingBenchmarkRequest
        task = ServingBenchmarkTask()
        req = ServingBenchmarkRequest(
            model_path=str(tmp_path / "unused.pt"),
            model_format="eager",
            num_requests=10,
        )
        result = task.execute(req)
        assert result.p99_latency_ms >= result.p50_latency_ms

    def test_serving_benchmark_throughput_positive(self, tmp_path):
        from quantedge_services.services.wfs.model_serving.tasks.serving_benchmark_task import ServingBenchmarkTask
        from quantedge_services.api.schemas.foundations.quantization_schemas import ServingBenchmarkRequest
        task = ServingBenchmarkTask()
        req = ServingBenchmarkRequest(
            model_path=str(tmp_path / "unused.pt"),
            model_format="eager",
            num_requests=10,
        )
        result = task.execute(req)
        assert result.throughput_qps > 0


class TestServicesAndFacade:
    def test_serving_service_has_infer_method(self):
        from quantedge_services.services.wfs.model_serving.serving_service import ModelServingService
        assert hasattr(ModelServingService, "infer")

    def test_serving_service_has_benchmark_method(self):
        from quantedge_services.services.wfs.model_serving.serving_service import ModelServingService
        assert hasattr(ModelServingService, "benchmark")

    def test_facade_has_submit_quantize_static(self):
        from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
        assert hasattr(FoundationsServiceFacade, "submit_quantize_static")

    def test_facade_has_submit_model_infer(self):
        from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
        assert hasattr(FoundationsServiceFacade, "submit_model_infer")

    def test_admin_router_has_quantize_dynamic_endpoint(self):
        from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
        routes = [r.path for r in FoundationsAdminRouter.__dict__.get('router', None).routes] if hasattr(FoundationsAdminRouter, 'router') else []
        # Just check the class has the method
        assert hasattr(FoundationsAdminRouter, "quantize_dynamic")

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    ModelInferRequest,
    ModelInferResult,
    ServingBenchmarkRequest,
    ServingBenchmarkResult,
)
from quantedge_services.services.wfs.model_serving.tasks.model_infer_task import ModelInferTask
from quantedge_services.services.wfs.model_serving.tasks.serving_benchmark_task import ServingBenchmarkTask


class ModelServingService:
    def __init__(
        self,
        infer_task: ModelInferTask,
        benchmark_task: ServingBenchmarkTask,
    ) -> None:
        self._infer_task = infer_task
        self._benchmark_task = benchmark_task

    async def infer(self, request: ModelInferRequest) -> ModelInferResult:
        return self._infer_task.execute(request)

    async def benchmark(self, request: ServingBenchmarkRequest) -> ServingBenchmarkResult:
        return self._benchmark_task.execute(request)

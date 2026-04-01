from quantedge_services.api.schemas.foundations.quantization_schemas import (
    QuantizeStaticRequest,
    QuantizeStaticResult,
    QuantizeDynamicRequest,
    QuantizeDynamicResult,
    QuantizeQATRequest,
    QuantizeQATResult,
    QuantizeCompareRequest,
    QuantizeCompareResult,
)
from quantedge_services.services.wfs.quantization.tasks.static_quant_task import StaticQuantTask
from quantedge_services.services.wfs.quantization.tasks.dynamic_quant_task import DynamicQuantTask
from quantedge_services.services.wfs.quantization.tasks.qat_task import QATTask
from quantedge_services.services.wfs.quantization.tasks.quant_compare_task import QuantCompareTask


class QuantizationService:
    def __init__(
        self,
        static_task: StaticQuantTask,
        dynamic_task: DynamicQuantTask,
        qat_task: QATTask,
        compare_task: QuantCompareTask,
    ) -> None:
        self._static_task = static_task
        self._dynamic_task = dynamic_task
        self._qat_task = qat_task
        self._compare_task = compare_task

    async def quantize_static(self, request: QuantizeStaticRequest) -> QuantizeStaticResult:
        return self._static_task.execute(request)

    async def quantize_dynamic(self, request: QuantizeDynamicRequest) -> QuantizeDynamicResult:
        return self._dynamic_task.execute(request)

    async def quantize_qat(self, request: QuantizeQATRequest) -> QuantizeQATResult:
        return self._qat_task.execute(request)

    async def quantize_compare(self, request: QuantizeCompareRequest) -> QuantizeCompareResult:
        return self._compare_task.execute(request)

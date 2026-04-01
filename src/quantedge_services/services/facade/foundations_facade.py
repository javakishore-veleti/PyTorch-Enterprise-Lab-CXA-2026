"""FoundationsServiceFacade — single entry point for all Foundations API calls."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.export_schemas import (
    TorchScriptExportRequest, TorchScriptExportResult,
    ONNXExportRequest, ONNXExportResult,
    ONNXValidateRequest, ONNXValidateResult,
    TensorRTExportRequest, TensorRTExportResult,
    BenchmarkRequest, BenchmarkResult,
)
from quantedge_services.services.wfs.model_export.export_service import ModelExportService
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    CICIoTDownloadRequest, CICIoTDownloadResponse,
    CICIoTIngestionRequest, CICIoTIngestionResponse,
    DataloaderTuneRequest, DataloaderTuneResponse,
    MemorySummaryRequest, MemorySummaryResponse,
    ProfilerRunRequest, ProfilerRunResponse,
)
from quantedge_services.services.wfs.cic_iot.cic_iot_service import CICIoTService
from quantedge_services.services.wfs.profiling.profiling_service import ProfilingService
from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest, CFPBDatasetResponse,
    CFPBDownloadRequest, CFPBDownloadResponse,
    CFPBIngestionRequest, CFPBIngestionResponse,
    CFPBPreprocessRequest, CFPBPreprocessResponse,
    CFPBTrainRequest, CFPBTrainResponse,
)
from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest, ForexAutogradResponse,
    ForexDownloadRequest, ForexDownloadResponse,
    ForexIngestionRequest, ForexIngestionResponse,
    ForexPreprocessRequest, ForexPreprocessResponse,
    ForexTensorOpsRequest, ForexTensorOpsResponse,
)
from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNEvalRequest, NNEvalResponse,
    NNPredictRequest, NNPredictResponse,
    NNTrainRequest, NNTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.cfpb_complaints.cfpb_service import CFPBComplaintsService
from quantedge_services.services.wfs.forex_eurusd.forex_service import ForexEURUSDService
from quantedge_services.services.wfs.forex_neuralnet.forex_nn_service import ForexNeuralNetService
from quantedge_services.api.schemas.foundations.attention_schemas import (
    CMAPSSDownloadRequest, CMAPSSDownloadResponse,
    CMAPSSIngestionRequest, CMAPSSIngestionResponse,
    AttentionTrainRequest, AttentionTrainResponse,
    AttentionEvalRequest, AttentionEvalResponse,
    AttentionPredictRequest, AttentionPredictResponse,
)
from quantedge_services.services.wfs.cmapss.cmapss_service import CMAPSSService
from quantedge_services.services.wfs.forex_attention.forex_attention_service import ForexAttentionService
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionExtractRequest, AttentionExtractResponse,
    AttentionHeatmapRequest, AttentionHeatmapResponse,
    ArchDecisionRequest, ArchDecisionResponse,
)
from quantedge_services.services.wfs.attention_viz.attention_viz_service import AttentionVizService
from quantedge_services.api.schemas.foundations.lora_schemas import (
    OAsst1DownloadRequest, OAsst1DownloadResponse,
    OAsst1IngestionRequest, OAsst1IngestionResponse,
    LoRATrainRequest, LoRATrainResponse,
    LoRAEvalRequest, LoRAEvalResponse,
    LoRAPredictRequest, LoRAPredictResponse,
    LoRAMergeRequest, LoRAMergeResponse,
)
from quantedge_services.services.wfs.oasst1.oasst1_service import OAsst1Service
from quantedge_services.services.wfs.lora_finetuning.lora_service import LoRAService
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest, StackOverflowDownloadResult,
    StackOverflowIngestionRequest, StackOverflowIngestionResult,
    DomainAdaptTrainRequest, DomainAdaptTrainResult,
    DomainAdaptEvalRequest, DomainAdaptEvalResult,
    OllamaInferRequest, OllamaInferResult,
    OllamaMergeRequest, OllamaMergeResult,
)
from quantedge_services.services.wfs.stackoverflow.stackoverflow_service import StackOverflowService
from quantedge_services.services.wfs.domain_adaptation.domain_adapt_service import DomainAdaptService
from quantedge_services.services.wfs.ollama_serving.ollama_service import OllamaService


class FoundationsServiceFacade:
    """Façade for the Foundations phase (Weeks 1–2).

    All API router classes call this facade exclusively.
    The facade delegates to domain services — the API layer never
    knows about individual task classes.
    """

    def __init__(
        self,
        forex_service: ForexEURUSDService,
        cfpb_service: CFPBComplaintsService,
        nn_service: ForexNeuralNetService,
        cic_iot_service: CICIoTService,
        profiling_service: ProfilingService,
        cmapss_service: CMAPSSService,
        attention_service: ForexAttentionService,
        attention_viz_service: AttentionVizService,
        oasst1_service: OAsst1Service,
        lora_service: LoRAService,
        stackoverflow_service: StackOverflowService,
        domain_adapt_service: DomainAdaptService,
        ollama_service: OllamaService,
        export_service: ModelExportService | None = None,
    ) -> None:
        self._forex = forex_service
        self._cfpb = cfpb_service
        self._nn = nn_service
        self._cic_iot = cic_iot_service
        self._profiling = profiling_service
        self._cmapss = cmapss_service
        self._attention = attention_service
        self._attention_viz = attention_viz_service
        self._oasst1 = oasst1_service
        self._lora = lora_service
        self._stackoverflow = stackoverflow_service
        self._domain_adapt = domain_adapt_service
        self._ollama = ollama_service
        self._export = export_service
        self._logger = StructuredLogger(name=__name__)

    # ── Forex ──────────────────────────────────────────────────────────────

    async def forex_download(self, request: ForexDownloadRequest) -> ForexDownloadResponse:
        return await self._forex.download(request)

    async def forex_ingest(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        self._logger.info("facade_forex_ingest", execution_id=None)
        return await self._forex.ingest(request)

    async def forex_preprocess(self, request: ForexPreprocessRequest) -> ForexPreprocessResponse:
        return await self._forex.preprocess(request)

    async def forex_autograd(self, request: ForexAutogradRequest) -> ForexAutogradResponse:
        return await self._forex.run_autograd(request)

    async def forex_tensor_ops(self, request: ForexTensorOpsRequest) -> ForexTensorOpsResponse:
        return await self._forex.run_tensor_ops(request)

    # ── CFPB ───────────────────────────────────────────────────────────────

    async def cfpb_download(self, request: CFPBDownloadRequest) -> CFPBDownloadResponse:
        return await self._cfpb.download(request)

    async def cfpb_ingest(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        self._logger.info("facade_cfpb_ingest", execution_id=None)
        return await self._cfpb.ingest(request)

    async def cfpb_preprocess(self, request: CFPBPreprocessRequest) -> CFPBPreprocessResponse:
        return await self._cfpb.preprocess(request)

    async def cfpb_build_dataloaders(self, request: CFPBDatasetRequest) -> CFPBDatasetResponse:
        return await self._cfpb.build_dataloaders(request)

    async def cfpb_train(self, request: CFPBTrainRequest) -> CFPBTrainResponse:
        return await self._cfpb.train(request)

    # ── Neural Networks ────────────────────────────────────────────────────

    async def nn_train(self, request: NNTrainRequest) -> NNTrainResponse:
        return await self._nn.train(request)

    async def nn_evaluate(self, request: NNEvalRequest) -> NNEvalResponse:
        return await self._nn.evaluate(request)

    async def nn_predict(self, request: NNPredictRequest) -> NNPredictResponse:
        return await self._nn.predict(request)

    # ── CIC IoT ────────────────────────────────────────────────────────────

    async def cic_iot_download(self, request: CICIoTDownloadRequest) -> CICIoTDownloadResponse:
        return await self._cic_iot.download(request)

    async def cic_iot_ingest(self, request: CICIoTIngestionRequest) -> CICIoTIngestionResponse:
        return await self._cic_iot.ingest(request)

    # ── Profiling ──────────────────────────────────────────────────────────

    async def profiler_run(self, request: ProfilerRunRequest) -> ProfilerRunResponse:
        return await self._profiling.run_profiler(request)

    async def memory_summary(self, request: MemorySummaryRequest) -> MemorySummaryResponse:
        return await self._profiling.memory_summary(request)

    async def tune_dataloader(self, request: DataloaderTuneRequest) -> DataloaderTuneResponse:
        return await self._profiling.tune_dataloader(request)

    # ── CMAPSS ─────────────────────────────────────────────────────────────

    async def cmapss_download(self, request: CMAPSSDownloadRequest) -> CMAPSSDownloadResponse:
        return await self._cmapss.download(request)

    async def cmapss_ingest(self, request: CMAPSSIngestionRequest) -> CMAPSSIngestionResponse:
        return await self._cmapss.ingest(request)

    # ── Attention ──────────────────────────────────────────────────────────

    async def attention_train(self, request: AttentionTrainRequest) -> AttentionTrainResponse:
        return await self._attention.train(request)

    async def attention_evaluate(self, request: AttentionEvalRequest) -> AttentionEvalResponse:
        return await self._attention.evaluate(request)

    async def attention_predict(self, request: AttentionPredictRequest) -> AttentionPredictResponse:
        return await self._attention.predict(request)

    # ── Attention Viz ──────────────────────────────────────────────────────

    async def attention_extract(self, request: AttentionExtractRequest) -> AttentionExtractResponse:
        return await self._attention_viz.extract_weights(request)

    async def attention_heatmap(self, request: AttentionHeatmapRequest) -> AttentionHeatmapResponse:
        return await self._attention_viz.generate_heatmaps(request)

    async def architecture_decision(self, request: ArchDecisionRequest) -> ArchDecisionResponse:
        return await self._attention_viz.architecture_decision(request)

    # ── OAsst1 ─────────────────────────────────────────────────────────────

    async def oasst1_download(self, request: OAsst1DownloadRequest) -> OAsst1DownloadResponse:
        return await self._oasst1.download(request)

    async def oasst1_ingest(self, request: OAsst1IngestionRequest) -> OAsst1IngestionResponse:
        return await self._oasst1.ingest(request)

    # ── LoRA ───────────────────────────────────────────────────────────────

    async def lora_train(self, request: LoRATrainRequest) -> LoRATrainResponse:
        return await self._lora.train(request)

    async def lora_evaluate(self, request: LoRAEvalRequest) -> LoRAEvalResponse:
        return await self._lora.evaluate(request)

    async def lora_predict(self, request: LoRAPredictRequest) -> LoRAPredictResponse:
        return await self._lora.predict(request)

    async def lora_merge(self, request: LoRAMergeRequest) -> LoRAMergeResponse:
        return await self._lora.merge(request)

    # ── StackOverflow ──────────────────────────────────────────────────────────

    async def submit_stackoverflow_download(
        self, request: StackOverflowDownloadRequest
    ) -> StackOverflowDownloadResult:
        return await self._stackoverflow.download(request)

    async def submit_stackoverflow_ingest(
        self, request: StackOverflowIngestionRequest
    ) -> StackOverflowIngestionResult:
        return await self._stackoverflow.ingest(request)

    # ── Domain Adaptation ──────────────────────────────────────────────────────

    async def submit_domain_adapt_train(
        self, request: DomainAdaptTrainRequest
    ) -> DomainAdaptTrainResult:
        return await self._domain_adapt.train(request)

    async def submit_domain_adapt_eval(
        self, request: DomainAdaptEvalRequest
    ) -> DomainAdaptEvalResult:
        return await self._domain_adapt.evaluate(request)

    # ── Ollama ─────────────────────────────────────────────────────────────────

    async def submit_ollama_infer(
        self, request: OllamaInferRequest
    ) -> OllamaInferResult:
        return await self._ollama.infer(request)

    async def submit_ollama_merge(
        self, request: OllamaMergeRequest
    ) -> OllamaMergeResult:
        return await self._ollama.merge(request)

    # ── Model Export ───────────────────────────────────────────────────────────

    async def submit_torchscript_export(
        self, request: TorchScriptExportRequest
    ) -> TorchScriptExportResult:
        return await self._export.export_torchscript(request)

    async def submit_onnx_export(
        self, request: ONNXExportRequest
    ) -> ONNXExportResult:
        return await self._export.export_onnx(request)

    async def submit_onnx_validate(
        self, request: ONNXValidateRequest
    ) -> ONNXValidateResult:
        return await self._export.validate_onnx(request)

    async def submit_tensorrt_export(
        self, request: TensorRTExportRequest
    ) -> TensorRTExportResult:
        return await self._export.export_tensorrt(request)

    async def submit_benchmark(
        self, request: BenchmarkRequest
    ) -> BenchmarkResult:
        return await self._export.benchmark(request)

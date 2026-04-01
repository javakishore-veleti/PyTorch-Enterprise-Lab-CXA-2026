"""FoundationsServiceFacade — single entry point for all Foundations API calls."""
from __future__ import annotations
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
    ) -> None:
        self._forex = forex_service
        self._cfpb = cfpb_service
        self._nn = nn_service
        self._cic_iot = cic_iot_service
        self._profiling = profiling_service
        self._cmapss = cmapss_service
        self._attention = attention_service
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

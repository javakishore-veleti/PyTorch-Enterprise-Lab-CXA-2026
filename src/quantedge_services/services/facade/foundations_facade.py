"""FoundationsServiceFacade — single entry point for all Foundations API calls."""

from __future__ import annotations

import torch.nn as nn

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest,
    CFPBDatasetResponse,
    CFPBIngestionRequest,
    CFPBIngestionResponse,
    CFPBPreprocessRequest,
    CFPBPreprocessResponse,
    CFPBTrainRequest,
    CFPBTrainResponse,
)
from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest,
    ForexAutogradResponse,
    ForexDownloadRequest,
    ForexDownloadResponse,
    ForexIngestionRequest,
    ForexIngestionResponse,
    ForexPreprocessRequest,
    ForexPreprocessResponse,
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.cfpb_complaints.cfpb_service import CFPBComplaintsService
from quantedge_services.services.wfs.forex_eurusd.forex_service import ForexEURUSDService


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
    ) -> None:
        self._forex = forex_service
        self._cfpb = cfpb_service
        self._logger = StructuredLogger(name=__name__)

    # ── Forex ──────────────────────────────────────────────────────────────

    def forex_download(self, request: ForexDownloadRequest) -> ForexDownloadResponse:
        return self._forex.download(request)

    def forex_ingest(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        self._logger.info("facade_forex_ingest", execution_id=None)
        return self._forex.ingest(request)

    def forex_preprocess(self, request: ForexPreprocessRequest) -> ForexPreprocessResponse:
        return self._forex.preprocess(request)

    def forex_autograd(self, request: ForexAutogradRequest) -> ForexAutogradResponse:
        return self._forex.run_autograd(request)

    def forex_tensor_ops(self, request: ForexTensorOpsRequest) -> ForexTensorOpsResponse:
        return self._forex.run_tensor_ops(request)

    # ── CFPB ───────────────────────────────────────────────────────────────

    def cfpb_ingest(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        self._logger.info("facade_cfpb_ingest", execution_id=None)
        return self._cfpb.ingest(request)

    def cfpb_preprocess(self, request: CFPBPreprocessRequest) -> CFPBPreprocessResponse:
        return self._cfpb.preprocess(request)

    def cfpb_build_dataloaders(self, request: CFPBDatasetRequest) -> CFPBDatasetResponse:
        return self._cfpb.build_dataloaders(request)

    def cfpb_train(
        self, request: CFPBTrainRequest, model: nn.Module
    ) -> CFPBTrainResponse:
        return self._cfpb.train(request, model)

"""FoundationsAdminRouter — admin API endpoints for Foundations workflows."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from quantedge_services.api.middleware.nexus.auth import JWTAuthMiddleware
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
    ForexIngestionRequest,
    ForexIngestionResponse,
    ForexPreprocessRequest,
    ForexPreprocessResponse,
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade

_auth = JWTAuthMiddleware()


class FoundationsAdminRouter:
    """Admin-only router for triggering and monitoring Foundations workflows.

    Mount via: app.include_router(FoundationsAdminRouter(facade).router)
    """

    def __init__(self, facade: FoundationsServiceFacade) -> None:
        self._facade = facade
        self.router = APIRouter(
            prefix="/admin/foundations",
            tags=["Admin — Foundations"],
            dependencies=[Depends(_auth)],
        )
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.post("/forex/ingest", response_model=ForexIngestionResponse)(
            self.forex_ingest
        )
        self.router.post("/forex/preprocess", response_model=ForexPreprocessResponse)(
            self.forex_preprocess
        )
        self.router.post("/forex/autograd", response_model=ForexAutogradResponse)(
            self.forex_autograd
        )
        self.router.post("/forex/tensor-ops", response_model=ForexTensorOpsResponse)(
            self.forex_tensor_ops
        )
        self.router.post("/cfpb/ingest", response_model=CFPBIngestionResponse)(
            self.cfpb_ingest
        )
        self.router.post("/cfpb/preprocess", response_model=CFPBPreprocessResponse)(
            self.cfpb_preprocess
        )
        self.router.post("/cfpb/dataloaders", response_model=CFPBDatasetResponse)(
            self.cfpb_build_dataloaders
        )
        self.router.post("/cfpb/train", response_model=CFPBTrainResponse)(
            self.cfpb_train
        )

    async def forex_ingest(self, request: ForexIngestionRequest) -> ForexIngestionResponse:
        return self._facade.forex_ingest(request)

    async def forex_preprocess(self, request: ForexPreprocessRequest) -> ForexPreprocessResponse:
        return self._facade.forex_preprocess(request)

    async def forex_autograd(self, request: ForexAutogradRequest) -> ForexAutogradResponse:
        return self._facade.forex_autograd(request)

    async def forex_tensor_ops(self, request: ForexTensorOpsRequest) -> ForexTensorOpsResponse:
        return self._facade.forex_tensor_ops(request)

    async def cfpb_ingest(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        return self._facade.cfpb_ingest(request)

    async def cfpb_preprocess(self, request: CFPBPreprocessRequest) -> CFPBPreprocessResponse:
        return self._facade.cfpb_preprocess(request)

    async def cfpb_build_dataloaders(self, request: CFPBDatasetRequest) -> CFPBDatasetResponse:
        return self._facade.cfpb_build_dataloaders(request)

    async def cfpb_train(self, request: CFPBTrainRequest) -> CFPBTrainResponse:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            request.model_name, num_labels=20
        )
        return self._facade.cfpb_train(request, model)

"""FoundationsClientRouter — client-facing API for Foundations results."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from quantedge_services.api.middleware.nexus.auth import JWTAuthMiddleware
from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBPredictRequest,
    CFPBPredictResponse,
)
from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade

_auth = JWTAuthMiddleware()


class FoundationsClientRouter:
    """Client-facing router — read-only analytics and inference endpoints.

    Clients see signals and predictions, not raw training controls.
    """

    def __init__(self, facade: FoundationsServiceFacade) -> None:
        self._facade = facade
        self.router = APIRouter(
            prefix="/client/foundations",
            tags=["Client — Foundations"],
            dependencies=[Depends(_auth)],
        )
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.post("/forex/signals", response_model=ForexTensorOpsResponse)(
            self.get_forex_signals
        )
        self.router.post("/cfpb/predict", response_model=CFPBPredictResponse)(
            self.predict_complaint_product
        )

    async def get_forex_signals(
        self, request: ForexTensorOpsRequest
    ) -> ForexTensorOpsResponse:
        return self._facade.forex_tensor_ops(request)

    async def predict_complaint_product(
        self, request: CFPBPredictRequest
    ) -> CFPBPredictResponse:
        # Inference path — to be implemented in Week 2 completion
        return CFPBPredictResponse(
            execution_id=request.execution_id,
            predicted_product="pending",
            confidence=0.0,
            status="failed",
            error="Inference endpoint requires a trained checkpoint (Week 2).",
        )

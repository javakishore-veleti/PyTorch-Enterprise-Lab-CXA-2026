"""QuantEdgeApp — FastAPI application factory."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from quantedge_services.api.dependencies import get_container
from quantedge_services.api.middleware.nexus.correlation import CorrelationIdMiddleware
from quantedge_services.api.middleware.nexus.request_log import RequestLoggingMiddleware
from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
from quantedge_services.api.routers.client.foundations_router import FoundationsClientRouter
from quantedge_services.core.logging import StructuredLogger


class QuantEdgeApp:
    """Builds and owns the FastAPI application instance."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__, json_output=False)
        self._app = self._build()

    def _build(self) -> FastAPI:
        app = FastAPI(
            title="QuantEdge Services API",
            description="Enterprise PyTorch platform for financial AI intelligence",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._register_middleware(app)
        self._register_routers(app)
        return app

    def _register_middleware(self, app: FastAPI) -> None:
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(CorrelationIdMiddleware)

    def _register_routers(self, app: FastAPI) -> None:
        container = get_container()
        facade = container.foundations_facade

        app.include_router(FoundationsAdminRouter(facade).router)
        app.include_router(FoundationsClientRouter(facade).router)

        @app.get("/health", tags=["Health"])
        async def health() -> dict:
            return {"status": "ok", "service": "quantedge-services"}

    @property
    def app(self) -> FastAPI:
        return self._app


# Application instance — imported by uvicorn
app = QuantEdgeApp().app


def main() -> None:
    uvicorn.run(
        "quantedge_services.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

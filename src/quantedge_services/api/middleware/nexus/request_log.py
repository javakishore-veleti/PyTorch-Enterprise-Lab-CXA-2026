"""RequestLoggingMiddleware — structured JSON logging for every request."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from quantedge_services.core.logging import StructuredLogger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._logger = StructuredLogger(name=__name__)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        self._logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )
        return response

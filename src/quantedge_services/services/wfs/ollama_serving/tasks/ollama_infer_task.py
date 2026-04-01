"""OllamaInferTask — send a prompt to a running Ollama server."""
from __future__ import annotations
import time

import requests

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    OllamaInferRequest,
    OllamaInferResult,
)


class OllamaInferTask:
    """POSTs to the Ollama ``/api/generate`` endpoint.

    If Ollama is not running (``ConnectionError`` or timeout) the task returns
    gracefully with ``status="ollama_unavailable"`` rather than raising.
    """

    def execute(self, request: OllamaInferRequest) -> OllamaInferResult:
        url = f"{request.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        start = time.monotonic()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.monotonic() - start) * 1000.0
            return OllamaInferResult(
                response=data.get("response", ""),
                status="success",
                latency_ms=latency_ms,
            )
        except (requests.ConnectionError, requests.Timeout):
            return OllamaInferResult(
                response="",
                status="ollama_unavailable",
                latency_ms=0.0,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            return OllamaInferResult(
                response="",
                status="error",
                latency_ms=latency_ms,
                error=str(exc),
            )

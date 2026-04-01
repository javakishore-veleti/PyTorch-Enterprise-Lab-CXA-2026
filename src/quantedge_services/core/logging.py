"""StructuredLogger — structured JSON logging wrapper around structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


class StructuredLogger:
    """Structured JSON logger. One instance per class, injected via constructor.

    Usage:
        self._logger = StructuredLogger(name=__name__)
        self._logger.info("event_name", key=value)
    """

    _configured: bool = False

    def __init__(self, name: str, json_output: bool = True, level: str = "INFO") -> None:
        self._name = name
        if not StructuredLogger._configured:
            StructuredLogger._configure(json_output=json_output, level=level)
        self._log = structlog.get_logger(name)

    @staticmethod
    def _configure(json_output: bool, level: str) -> None:
        logging.basicConfig(
            stream=sys.stdout,
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(message)s",  # structlog provides all formatting
        )
        processors: list[structlog.types.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True),
        ]
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        StructuredLogger._configured = True

    def info(self, event: str, **kwargs: Any) -> None:
        self._log.info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log.error(event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log.debug(event, **kwargs)

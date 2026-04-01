"""DomainAdaptService — async wrapper around domain adaptation tasks."""
from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    DomainAdaptTrainRequest,
    DomainAdaptTrainResult,
    DomainAdaptEvalRequest,
    DomainAdaptEvalResult,
)
from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_train_task import DomainAdaptTrainTask
from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_eval_task import DomainAdaptEvalTask


class DomainAdaptService:
    def __init__(
        self,
        train_task: DomainAdaptTrainTask,
        eval_task: DomainAdaptEvalTask,
    ) -> None:
        self._train = train_task
        self._eval = eval_task

    async def train(self, request: DomainAdaptTrainRequest) -> DomainAdaptTrainResult:
        return await asyncio.to_thread(self._train.execute, request)

    async def evaluate(self, request: DomainAdaptEvalRequest) -> DomainAdaptEvalResult:
        return await asyncio.to_thread(self._eval.execute, request)

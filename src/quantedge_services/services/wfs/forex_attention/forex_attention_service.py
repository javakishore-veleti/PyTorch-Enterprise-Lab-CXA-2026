from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.attention_schemas import (
    AttentionEvalRequest, AttentionEvalResponse,
    AttentionPredictRequest, AttentionPredictResponse,
    AttentionTrainRequest, AttentionTrainResponse,
)
from quantedge_services.services.wfs.forex_attention.tasks.attention_train_task import AttentionTrainTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_eval_task import AttentionEvalTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_predict_task import AttentionPredictTask


class ForexAttentionService:
    def __init__(
        self,
        train_task: AttentionTrainTask,
        eval_task: AttentionEvalTask,
        predict_task: AttentionPredictTask,
    ) -> None:
        self._train = train_task
        self._eval = eval_task
        self._predict = predict_task

    async def train(self, request: AttentionTrainRequest) -> AttentionTrainResponse:
        return await asyncio.to_thread(self._train.execute, request)

    async def evaluate(self, request: AttentionEvalRequest) -> AttentionEvalResponse:
        return await asyncio.to_thread(self._eval.execute, request)

    async def predict(self, request: AttentionPredictRequest) -> AttentionPredictResponse:
        return await asyncio.to_thread(self._predict.execute, request)

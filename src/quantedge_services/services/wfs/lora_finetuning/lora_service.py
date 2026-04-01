from __future__ import annotations
import asyncio
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRATrainRequest, LoRATrainResponse,
    LoRAEvalRequest, LoRAEvalResponse,
    LoRAPredictRequest, LoRAPredictResponse,
    LoRAMergeRequest, LoRAMergeResponse,
)
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_train_task import LoRATrainTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_eval_task import LoRAEvalTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_predict_task import LoRAPredictTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_merge_task import LoRAMergeTask


class LoRAService:
    def __init__(
        self,
        train_task: LoRATrainTask,
        eval_task: LoRAEvalTask,
        predict_task: LoRAPredictTask,
        merge_task: LoRAMergeTask,
    ) -> None:
        self._train = train_task
        self._eval = eval_task
        self._predict = predict_task
        self._merge = merge_task

    async def train(self, request: LoRATrainRequest) -> LoRATrainResponse:
        return await asyncio.to_thread(self._train.execute, request)

    async def evaluate(self, request: LoRAEvalRequest) -> LoRAEvalResponse:
        return await asyncio.to_thread(self._eval.execute, request)

    async def predict(self, request: LoRAPredictRequest) -> LoRAPredictResponse:
        return await asyncio.to_thread(self._predict.execute, request)

    async def merge(self, request: LoRAMergeRequest) -> LoRAMergeResponse:
        return await asyncio.to_thread(self._merge.execute, request)

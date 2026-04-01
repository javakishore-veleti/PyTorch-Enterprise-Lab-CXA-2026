"""ForexNeuralNetService — async service wrapping NN train/eval/predict tasks."""
from __future__ import annotations

import asyncio

from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNEvalRequest,
    NNEvalResponse,
    NNPredictRequest,
    NNPredictResponse,
    NNTrainRequest,
    NNTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.tasks.eval_task import NNEvalTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.predict_task import NNPredictTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.train_task import NNTrainTask


class ForexNeuralNetService:
    """Orchestrates neural network train/eval/predict workflows for Forex data.

    Called exclusively by FoundationsServiceFacade.
    """

    def __init__(
        self,
        train_task: NNTrainTask,
        eval_task: NNEvalTask,
        predict_task: NNPredictTask,
    ) -> None:
        self._train_task = train_task
        self._eval_task = eval_task
        self._predict_task = predict_task
        self._logger = StructuredLogger(name=__name__)

    async def train(self, request: NNTrainRequest) -> NNTrainResponse:
        return await asyncio.to_thread(self._train_task.execute, request)

    async def evaluate(self, request: NNEvalRequest) -> NNEvalResponse:
        return await asyncio.to_thread(self._eval_task.execute, request)

    async def predict(self, request: NNPredictRequest) -> NNPredictResponse:
        return await asyncio.to_thread(self._predict_task.execute, request)

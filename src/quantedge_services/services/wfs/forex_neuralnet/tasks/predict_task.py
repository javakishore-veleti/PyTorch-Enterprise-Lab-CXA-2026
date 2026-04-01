"""NNPredictTask — runs inference on raw feature input using a saved checkpoint."""
from __future__ import annotations

import numpy as np
import torch

from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNPredictRequest,
    NNPredictResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP


class NNPredictTask:
    """Loads checkpoint, runs forward pass on request.input_features."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: NNPredictRequest) -> NNPredictResponse:
        try:
            ckpt = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
            model = self._restore_model(ckpt)
            model.eval()

            X = np.array(request.input_features, dtype=np.float32)
            if ckpt.get("model_type") == "lstm":
                X = X[:, np.newaxis, :]

            x_tensor = torch.from_numpy(X)
            with torch.no_grad():
                preds = model(x_tensor).squeeze(-1).tolist()

            if not isinstance(preds, list):
                preds = [preds]

            return NNPredictResponse(
                execution_id=request.execution_id,
                predictions=preds,
                status="success",
            )
        except Exception as exc:
            self._logger.error("nn_predict_failed", error=str(exc))
            return NNPredictResponse(
                execution_id=request.execution_id,
                predictions=[],
                status="failed",
                error=str(exc),
            )

    @staticmethod
    def _restore_model(ckpt: dict) -> torch.nn.Module:
        model_type = ckpt.get("model_type", "mlp")
        input_size = ckpt["input_size"]
        dropout_rate = ckpt.get("dropout_rate", 0.3)
        if model_type == "lstm":
            hidden_sizes = ckpt.get("hidden_sizes", [128])
            model = ForexLSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0] if hidden_sizes else 128,
                num_layers=2,
                dropout_rate=dropout_rate,
            )
        else:
            model = ForexMLP(
                input_size=input_size,
                hidden_sizes=ckpt.get("hidden_sizes", [128, 64, 32]),
                dropout_rate=dropout_rate,
            )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

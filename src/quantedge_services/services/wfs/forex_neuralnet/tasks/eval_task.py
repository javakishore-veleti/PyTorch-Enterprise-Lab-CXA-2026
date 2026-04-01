"""NNEvalTask — evaluates a saved NN checkpoint on a validation Parquet file."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNEvalRequest,
    NNEvalResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP


class NNEvalTask:
    """Loads a checkpoint, runs inference on a val Parquet, returns regression metrics."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: NNEvalRequest) -> NNEvalResponse:
        try:
            ckpt = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
            model = self._restore_model(ckpt)
            model.eval()

            df = pd.read_parquet(request.parquet_path)
            df = df.select_dtypes(include=[np.number]).dropna()

            feature_cols = ckpt.get("feature_cols", [c for c in df.columns if c != ckpt.get("target_col", df.columns[-1])])
            target_col = ckpt.get("target_col", df.columns[-1])

            X = df[feature_cols].values.astype(np.float32)
            y_true = df[target_col].values.astype(np.float32)

            if ckpt.get("model_type") == "lstm":
                X = X[:, np.newaxis, :]

            x_tensor = torch.from_numpy(X)
            with torch.no_grad():
                y_pred = model(x_tensor).squeeze(-1).numpy()

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(math.sqrt(mse))

            # Direction accuracy: compare sign of consecutive differences
            if len(y_true) > 1:
                true_dir = np.sign(np.diff(y_true))
                pred_dir = np.sign(np.diff(y_pred))
                direction_accuracy = float(np.mean(true_dir == pred_dir))
            else:
                direction_accuracy = 0.0

            return NNEvalResponse(
                execution_id=request.execution_id,
                mse=round(mse, 6),
                mae=round(mae, 6),
                rmse=round(rmse, 6),
                direction_accuracy=round(direction_accuracy, 4),
                status="success",
            )
        except Exception as exc:
            self._logger.error("nn_eval_failed", error=str(exc))
            return NNEvalResponse(
                execution_id=request.execution_id,
                mse=0.0, mae=0.0, rmse=0.0, direction_accuracy=0.0,
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

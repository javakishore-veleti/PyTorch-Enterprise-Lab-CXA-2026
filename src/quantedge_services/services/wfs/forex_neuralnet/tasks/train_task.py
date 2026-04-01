"""NNTrainTask — trains ForexMLP or ForexLSTM on a Parquet dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNTrainRequest,
    NNTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP

_VAL_SPLIT = 0.2


class NNTrainTask:
    """Loads Parquet, builds model, trains with early stopping, saves checkpoint."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: NNTrainRequest) -> NNTrainResponse:
        try:
            df = pd.read_parquet(request.parquet_path)
            df = df.select_dtypes(include=[np.number]).dropna()

            target_col = "close"
            if target_col not in df.columns:
                target_col = df.columns[-1]

            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols].values.astype(np.float32)
            y = df[target_col].values.astype(np.float32).reshape(-1, 1)

            split = int(len(X) * (1 - _VAL_SPLIT))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            input_size = X_train.shape[1]
            model = self._build_model(request, input_size)

            if request.model_type == "lstm":
                X_train = X_train[:, np.newaxis, :]  # [N, 1, features]
                X_val = X_val[:, np.newaxis, :]

            train_loader = self._make_loader(X_train, y_train, request.batch_size, shuffle=True)
            val_loader = self._make_loader(X_val, y_val, request.batch_size, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)
            criterion = nn.MSELoss()

            best_val_loss = float("inf")
            patience_counter = 0
            best_state: dict = {}
            epochs_trained = 0

            for epoch in range(request.epochs):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()

                val_loss = self._eval_loss(model, val_loader, criterion)
                epochs_trained = epoch + 1

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= request.patience:
                        self._logger.info("early_stopping", epoch=epoch, val_loss=val_loss)
                        break

            checkpoint_dir = Path("data/forex/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{request.execution_id}_{request.model_type}.pt"

            torch.save(
                {
                    "model_type": request.model_type,
                    "input_size": input_size,
                    "hidden_sizes": request.hidden_sizes,
                    "dropout_rate": request.dropout_rate,
                    "model_state_dict": best_state if best_state else model.state_dict(),
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                },
                checkpoint_path,
            )

            return NNTrainResponse(
                execution_id=request.execution_id,
                model_type=request.model_type,
                epochs_trained=epochs_trained,
                best_val_loss=round(best_val_loss, 6),
                checkpoint_path=str(checkpoint_path),
                status="success",
            )
        except Exception as exc:
            self._logger.error("nn_train_failed", error=str(exc))
            return NNTrainResponse(
                execution_id=request.execution_id,
                model_type=request.model_type,
                epochs_trained=0,
                best_val_loss=0.0,
                checkpoint_path="",
                status="failed",
                error=str(exc),
            )

    def _build_model(self, request: NNTrainRequest, input_size: int) -> nn.Module:
        if request.model_type == "lstm":
            return ForexLSTM(
                input_size=input_size,
                hidden_size=request.hidden_sizes[0] if request.hidden_sizes else 128,
                num_layers=2,
                dropout_rate=request.dropout_rate,
            )
        return ForexMLP(
            input_size=input_size,
            hidden_sizes=request.hidden_sizes,
            dropout_rate=request.dropout_rate,
        )

    @staticmethod
    def _make_loader(X: "np.ndarray", y: "np.ndarray", batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    @torch.no_grad()
    def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total, count = 0.0, 0
        for xb, yb in loader:
            pred = model(xb)
            total += criterion(pred, yb).item() * len(xb)
            count += len(xb)
        return total / count if count else float("inf")

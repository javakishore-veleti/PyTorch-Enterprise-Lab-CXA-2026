from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from quantedge_services.api.schemas.foundations.attention_schemas import (
    AttentionTrainRequest, AttentionTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer

_VAL_SPLIT = 0.2


class AttentionTrainTask:
    """Loads parquet, builds ForexTransformer, trains with early stopping, saves checkpoint."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: AttentionTrainRequest) -> AttentionTrainResponse:
        parquet_path = Path(request.parquet_path)
        if not parquet_path.exists():
            self._logger.info("attention_train_skipped", path=str(parquet_path))
            return AttentionTrainResponse(
                execution_id=request.execution_id,
                epochs_trained=0,
                best_val_loss=0.0,
                checkpoint_path="",
                status="skipped",
                error=f"Parquet not found: {parquet_path}",
            )

        try:
            df = pd.read_parquet(parquet_path)
            feature_cols = [c for c in df.columns if c.startswith("f")]
            X_raw = df[feature_cols].values.astype(np.float32)
            y_raw = df["rul"].values.astype(np.float32).reshape(-1, 1)

            X = X_raw.reshape(-1, request.seq_len, request.input_size)

            split = int(len(X) * (1 - _VAL_SPLIT))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y_raw[:split], y_raw[split:]

            model = ForexTransformer(
                input_size=request.input_size,
                d_model=request.d_model,
                nhead=request.nhead,
                num_encoder_layers=request.num_encoder_layers,
                dim_feedforward=request.dim_feedforward,
                dropout=request.dropout,
            )

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
                    pred, _ = model(xb)
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
                        self._logger.info("early_stopping", epoch=epoch)
                        break

            checkpoint_dir = Path(request.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{request.execution_id}_transformer.pt"

            torch.save(
                {
                    "input_size": request.input_size,
                    "d_model": request.d_model,
                    "nhead": request.nhead,
                    "num_encoder_layers": request.num_encoder_layers,
                    "dim_feedforward": request.dim_feedforward,
                    "dropout": request.dropout,
                    "seq_len": request.seq_len,
                    "model_state_dict": best_state if best_state else model.state_dict(),
                },
                checkpoint_path,
            )

            return AttentionTrainResponse(
                execution_id=request.execution_id,
                epochs_trained=epochs_trained,
                best_val_loss=round(best_val_loss, 6),
                checkpoint_path=str(checkpoint_path),
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("attention_train_failed", error=str(exc))
            return AttentionTrainResponse(
                execution_id=request.execution_id,
                epochs_trained=0,
                best_val_loss=0.0,
                checkpoint_path="",
                status="failed",
                error=str(exc),
            )

    @staticmethod
    def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    @torch.no_grad()
    def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total, count = 0.0, 0
        for xb, yb in loader:
            pred, _ = model(xb)
            total += criterion(pred, yb).item() * len(xb)
            count += len(xb)
        return total / count if count else float("inf")

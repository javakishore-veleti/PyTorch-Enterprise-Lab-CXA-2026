from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRATrainRequest, LoRATrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
from quantedge_services.services.wfs.lora_finetuning.models.lora_transformer import LoRATransformer

_VAL_SPLIT = 0.2


class LoRATrainTask:
    """Loads base ForexTransformer, wraps with LoRATransformer, trains only LoRA params."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: LoRATrainRequest) -> LoRATrainResponse:
        base_ckpt_path = Path(request.base_checkpoint_path)
        if not base_ckpt_path.exists():
            self._logger.info("lora_train_skipped_no_base", path=str(base_ckpt_path))
            return LoRATrainResponse(
                execution_id=request.execution_id,
                epochs_trained=0,
                best_val_loss=0.0,
                lora_checkpoint_path="",
                trainable_params=0,
                frozen_params=0,
                trainable_ratio=0.0,
                status="skipped",
                error=f"Base checkpoint not found: {base_ckpt_path}",
            )

        parquet_path = Path(request.parquet_path)
        if not parquet_path.exists():
            self._logger.info("lora_train_skipped_no_parquet", path=str(parquet_path))
            return LoRATrainResponse(
                execution_id=request.execution_id,
                epochs_trained=0,
                best_val_loss=0.0,
                lora_checkpoint_path="",
                trainable_params=0,
                frozen_params=0,
                trainable_ratio=0.0,
                status="skipped",
                error=f"Parquet not found: {parquet_path}",
            )

        try:
            base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            base_model = ForexTransformer(
                input_size=base_ckpt.get("input_size", request.input_size),
                d_model=base_ckpt["d_model"],
                nhead=base_ckpt["nhead"],
                num_encoder_layers=base_ckpt["num_encoder_layers"],
                dim_feedforward=base_ckpt["dim_feedforward"],
                dropout=base_ckpt.get("dropout", 0.1),
            )
            base_model.load_state_dict(base_ckpt["model_state_dict"])

            cfg = request.lora_config
            model = LoRATransformer(
                base_model=base_model,
                rank=cfg.rank,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
                target_modules=cfg.target_modules,
            )

            param_counts = model.count_parameters()
            trainable_params = param_counts["trainable"]
            frozen_params = param_counts["frozen"]
            total_params = param_counts["total"]
            trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

            df = pd.read_parquet(parquet_path)
            feature_cols = [c for c in df.columns if c.startswith("f")]
            X_raw = df[feature_cols].values.astype(np.float32)
            y_raw = df["rul"].values.astype(np.float32).reshape(-1, 1)
            X = X_raw.reshape(-1, request.seq_len, request.input_size)

            split = int(len(X) * (1 - _VAL_SPLIT))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y_raw[:split], y_raw[split:]

            train_loader = self._make_loader(X_train, y_train, request.batch_size, shuffle=True)
            val_loader = self._make_loader(X_val, y_val, request.batch_size, shuffle=False)

            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params_list, lr=request.learning_rate)
            criterion = nn.MSELoss()

            best_val_loss = float("inf")
            patience_counter = 0
            best_lora_state: dict = {}
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
                    best_lora_state = {k: v.clone() for k, v in model.get_lora_state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= request.patience:
                        self._logger.info("lora_early_stopping", epoch=epoch)
                        break

            ckpt_dir = Path(request.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lora_ckpt_path = ckpt_dir / f"{request.execution_id}_lora.pt"

            torch.save(
                {
                    "lora_state_dict": best_lora_state if best_lora_state else model.get_lora_state_dict(),
                    "lora_config": request.lora_config.model_dump(),
                    "base_checkpoint_path": str(base_ckpt_path),
                    "val_loss": best_val_loss,
                    "epoch": epochs_trained,
                },
                lora_ckpt_path,
            )

            return LoRATrainResponse(
                execution_id=request.execution_id,
                epochs_trained=epochs_trained,
                best_val_loss=round(best_val_loss, 6),
                lora_checkpoint_path=str(lora_ckpt_path),
                trainable_params=trainable_params,
                frozen_params=frozen_params,
                trainable_ratio=round(trainable_ratio, 6),
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("lora_train_failed", error=str(exc))
            return LoRATrainResponse(
                execution_id=request.execution_id,
                epochs_trained=0,
                best_val_loss=0.0,
                lora_checkpoint_path="",
                trainable_params=0,
                frozen_params=0,
                trainable_ratio=0.0,
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

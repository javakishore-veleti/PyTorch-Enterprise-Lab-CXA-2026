"""CFPBTrainingTask — training loop, eval loop, checkpoint save/resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBTrainRequest,
    CFPBTrainResponse,
)
from quantedge_services.core.checkpointing import CheckpointManager
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.core.reproducibility import ReproducibilityManager


class CFPBTrainingTask:
    def __init__(
        self,
        device_manager: DeviceManager,
        reproducibility_manager: ReproducibilityManager,
    ) -> None:
        self._device_manager = device_manager
        self._reproducibility_manager = reproducibility_manager
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: CFPBTrainRequest,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> CFPBTrainResponse:
        try:
            self._reproducibility_manager.apply()
            device = self._device_manager.device
            checkpoint_mgr = CheckpointManager(base_dir=request.checkpoint_dir)
            optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
            criterion = nn.CrossEntropyLoss()
            model.to(device)

            start_epoch = 0
            if request.resume_from:
                ckpt = checkpoint_mgr.load(Path(request.resume_from).name, device=device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                self._logger.info("resumed", epoch=start_epoch)

            train_loss = val_loss = val_acc = 0.0
            for epoch in range(start_epoch, request.epochs):
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device, epoch)
                val_metrics = self._eval_epoch(model, val_loader, criterion, device)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

                ckpt_path = checkpoint_mgr.save(
                    state={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    filename=f"epoch_{epoch:03d}.pt",
                    metadata={"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc},
                )

            return CFPBTrainResponse(
                execution_id=request.execution_id,
                epochs_completed=request.epochs - start_epoch,
                final_train_loss=round(train_loss, 6),
                final_val_loss=round(val_loss, 6),
                final_val_accuracy=round(val_acc, 4),
                checkpoint_path=str(ckpt_path),
                status="success",
            )
        except Exception as exc:
            self._logger.error("training_failed", error=str(exc))
            return CFPBTrainResponse(
                execution_id=request.execution_id, epochs_completed=0,
                final_train_loss=0.0, final_val_loss=0.0, final_val_accuracy=0.0,
                checkpoint_path="", status="failed", error=str(exc),
            )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
    ) -> float:
        model.train()
        total = 0.0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                self._logger.info("train_step", epoch=epoch, step=step, loss=round(loss.item(), 6))
        return total / len(loader)

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[Any],
        criterion: nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(input_ids=ids, attention_mask=mask).logits
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return {"loss": round(total_loss / len(loader), 6), "accuracy": round(correct / total, 4)}

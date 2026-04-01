"""ForexAutogradTask — manual backward vs torch.autograd comparison (Week 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest,
    ForexAutogradResponse,
)
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.logging import StructuredLogger


class ForexAutogradTask:
    """Runs the Week 1 autograd comparison exercise on EUR/USD close prices."""

    def __init__(self, device_manager: DeviceManager) -> None:
        self._device_manager = device_manager
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: ForexAutogradRequest,
        tensor: torch.Tensor,
    ) -> ForexAutogradResponse:
        try:
            device = self._device_manager.device
            close = tensor[:, 3].to(device)
            deltas = close[1:] - close[:-1]
            preds = deltas[:-1].clone()
            targets = deltas[1:].clone()

            # Manual MSE backward
            n = preds.shape[0]
            diff = preds - targets
            manual_loss = (diff ** 2).mean()
            manual_grad = (2.0 / n) * diff

            # Autograd MSE backward
            auto_preds = preds.clone().requires_grad_(True)
            auto_loss = F.mse_loss(auto_preds, targets)
            auto_loss.backward()

            grad = auto_preds.grad if auto_preds.grad is not None else torch.zeros_like(manual_grad)
            max_grad_diff = (manual_grad - grad).abs().max().item()
            self._logger.info(
                "autograd_comparison_complete",
                execution_id=request.execution_id,
                manual_loss=manual_loss.item(),
                autograd_loss=auto_loss.item(),
                max_grad_diff=max_grad_diff,
            )
            return ForexAutogradResponse(
                execution_id=request.execution_id,
                manual_loss=round(manual_loss.item(), 8),
                autograd_loss=round(auto_loss.item(), 8),
                max_grad_diff=round(max_grad_diff, 10),
                status="success",
            )
        except Exception as exc:
            self._logger.error("autograd_task_failed", error=str(exc))
            return ForexAutogradResponse(
                execution_id=request.execution_id,
                manual_loss=0.0, autograd_loss=0.0, max_grad_diff=0.0,
                status="failed", error=str(exc),
            )

"""ForexTensorOpsTask — rolling volatility, momentum, NaN injection/repair."""

from __future__ import annotations

import torch

from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexTensorOpsRequest,
    ForexTensorOpsResponse,
)
from quantedge_services.core.logging import StructuredLogger


class ForexTensorOpsTask:
    """Computes rolling financial metrics as pure tensor operations (no loops)."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: ForexTensorOpsRequest,
        tensor: torch.Tensor,
    ) -> ForexTensorOpsResponse:
        try:
            close = tensor[:, 3]
            vol = self._rolling_volatility(close, window=request.volatility_window)
            mom = self._rolling_momentum(close, window=request.momentum_window)
            nan_injected, nan_remaining = self._inject_and_fix(
                tensor, fraction=request.inject_nan_fraction
            )
            self._logger.info(
                "tensor_ops_complete", execution_id=request.execution_id,
                vol_points=vol.shape[0], mom_points=mom.shape[0],
            )
            return ForexTensorOpsResponse(
                execution_id=request.execution_id,
                volatility_points=vol.shape[0],
                momentum_points=mom.shape[0],
                nan_injected=nan_injected,
                nan_remaining=nan_remaining,
                status="success",
            )
        except Exception as exc:
            self._logger.error("tensor_ops_failed", error=str(exc))
            return ForexTensorOpsResponse(
                execution_id=request.execution_id,
                volatility_points=0, momentum_points=0,
                nan_injected=0, nan_remaining=0,
                status="failed", error=str(exc),
            )

    def _rolling_volatility(self, prices: torch.Tensor, window: int) -> torch.Tensor:
        log_returns = torch.log(prices[1:] / prices[:-1])
        windows = log_returns.unfold(0, window, 1)
        return windows.std(dim=1)

    def _rolling_momentum(self, prices: torch.Tensor, window: int) -> torch.Tensor:
        return (prices[window:] - prices[:-window]) / prices[:-window]

    def _inject_and_fix(
        self, tensor: torch.Tensor, fraction: float
    ) -> tuple[int, int]:
        flat = tensor.clone().view(-1)
        n_corrupt = max(1, int(flat.shape[0] * fraction))
        idx = torch.randperm(flat.shape[0])[:n_corrupt]
        flat[idx] = float("nan")
        injected = int(torch.isnan(flat).sum().item())
        # forward fill
        for i in range(1, flat.shape[0]):
            if torch.isnan(flat[i]):
                flat[i] = flat[i - 1]
        # backward fill for leading NaNs (index 0 case)
        for i in range(flat.shape[0] - 2, -1, -1):
            if torch.isnan(flat[i]):
                flat[i] = flat[i + 1]
        remaining = int(torch.isnan(flat).sum().item())
        return injected, remaining

"""Tests for ForexTensorOpsTask."""

from __future__ import annotations

import torch

from quantedge_services.api.schemas.foundations.forex_schemas import ForexTensorOpsRequest
from quantedge_services.services.wfs.forex_eurusd.tasks.tensor_ops_task import ForexTensorOpsTask


class TestForexTensorOpsTask:
    def setup_method(self) -> None:
        self._task = ForexTensorOpsTask()

    def _make_request(self, execution_id: str = "test") -> ForexTensorOpsRequest:
        return ForexTensorOpsRequest(
            execution_id=execution_id,
            volatility_window=10,
            momentum_window=5,
            inject_nan_fraction=0.05,
        )

    def test_success_status(self) -> None:
        tensor = torch.rand(500, 5) + 0.01  # avoid log(0)
        response = self._task.execute(self._make_request(), tensor)
        assert response.status == "success"

    def test_volatility_output_shape(self) -> None:
        n = 500
        tensor = torch.rand(n, 5) + 0.01
        response = self._task.execute(self._make_request(), tensor)
        # rolling vol: n - 1 log_returns, then unfold with window=10 → n-1-10+1 = n-10
        assert response.volatility_points == n - 10

    def test_nan_fully_repaired(self) -> None:
        tensor = torch.rand(300, 5) + 0.01
        response = self._task.execute(self._make_request(), tensor)
        assert response.nan_remaining == 0

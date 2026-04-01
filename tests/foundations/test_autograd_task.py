"""Tests for ForexAutogradTask."""

from __future__ import annotations

import torch

from quantedge_services.api.schemas.foundations.forex_schemas import ForexAutogradRequest
from quantedge_services.core.device import DeviceManager
from quantedge_services.services.wfs.forex_eurusd.tasks.autograd_task import ForexAutogradTask


class TestForexAutogradTask:
    def setup_method(self) -> None:
        self._task = ForexAutogradTask(device_manager=DeviceManager(prefer_gpu=False))

    def test_returns_success_status(self) -> None:
        tensor = torch.rand(100, 5)
        request = ForexAutogradRequest(execution_id="test-001", window_size=20)
        response = self._task.execute(request, tensor)
        assert response.status == "success"

    def test_grad_diff_is_near_zero(self) -> None:
        tensor = torch.rand(200, 5)
        request = ForexAutogradRequest(execution_id="test-002", window_size=20)
        response = self._task.execute(request, tensor)
        assert response.max_grad_diff < 1e-5, (
            f"Manual and autograd gradients diverged: {response.max_grad_diff}"
        )

    def test_losses_are_equal(self) -> None:
        tensor = torch.rand(200, 5)
        request = ForexAutogradRequest(execution_id="test-003", window_size=20)
        response = self._task.execute(request, tensor)
        assert abs(response.manual_loss - response.autograd_loss) < 1e-6

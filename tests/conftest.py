"""Shared test fixtures for QuantEdge test suite."""

from __future__ import annotations

import pytest

from quantedge_services.core.device import DeviceManager
from quantedge_services.core.reproducibility import ReproducibilityManager


@pytest.fixture(scope="session")
def device_manager() -> DeviceManager:
    return DeviceManager(prefer_gpu=False)  # always CPU in tests


@pytest.fixture(scope="session")
def reproducibility_manager() -> ReproducibilityManager:
    mgr = ReproducibilityManager(seed=0)
    mgr.apply()
    return mgr

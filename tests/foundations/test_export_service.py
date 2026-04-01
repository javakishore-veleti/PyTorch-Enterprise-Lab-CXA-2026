"""Week 9 — Export service and facade/router integration tests."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import pytest

from quantedge_services.services.wfs.model_export.export_service import ModelExportService
from quantedge_services.services.wfs.model_export.tasks.torchscript_export_task import TorchScriptExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_export_task import ONNXExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_validate_task import ONNXValidateTask
from quantedge_services.services.wfs.model_export.tasks.tensorrt_export_task import TensorRTExportTask
from quantedge_services.services.wfs.model_export.tasks.benchmark_task import BenchmarkTask
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter


def _make_export_service() -> ModelExportService:
    return ModelExportService(
        torchscript_task=TorchScriptExportTask(),
        onnx_export_task=ONNXExportTask(),
        onnx_validate_task=ONNXValidateTask(),
        tensorrt_task=TensorRTExportTask(),
        benchmark_task=BenchmarkTask(),
    )


def test_export_service_has_torchscript_method():
    svc = _make_export_service()
    assert hasattr(svc, "export_torchscript")
    import asyncio
    assert asyncio.iscoroutinefunction(svc.export_torchscript)


def test_export_service_has_onnx_export_method():
    svc = _make_export_service()
    assert hasattr(svc, "export_onnx")
    import asyncio
    assert asyncio.iscoroutinefunction(svc.export_onnx)


def test_export_service_has_onnx_validate_method():
    svc = _make_export_service()
    assert hasattr(svc, "validate_onnx")
    import asyncio
    assert asyncio.iscoroutinefunction(svc.validate_onnx)


def test_export_service_has_tensorrt_method():
    svc = _make_export_service()
    assert hasattr(svc, "export_tensorrt")
    import asyncio
    assert asyncio.iscoroutinefunction(svc.export_tensorrt)


def test_export_service_has_benchmark_method():
    svc = _make_export_service()
    assert hasattr(svc, "benchmark")
    import asyncio
    assert asyncio.iscoroutinefunction(svc.benchmark)


def _make_facade() -> FoundationsServiceFacade:
    return FoundationsServiceFacade(
        forex_service=MagicMock(),
        cfpb_service=MagicMock(),
        nn_service=MagicMock(),
        cic_iot_service=MagicMock(),
        profiling_service=MagicMock(),
        cmapss_service=MagicMock(),
        attention_service=MagicMock(),
        attention_viz_service=MagicMock(),
        oasst1_service=MagicMock(),
        lora_service=MagicMock(),
        stackoverflow_service=MagicMock(),
        domain_adapt_service=MagicMock(),
        ollama_service=MagicMock(),
        export_service=_make_export_service(),
    )


def test_facade_has_submit_torchscript_export():
    facade = _make_facade()
    assert hasattr(facade, "submit_torchscript_export")
    import asyncio
    assert asyncio.iscoroutinefunction(facade.submit_torchscript_export)


def test_facade_has_submit_onnx_export():
    facade = _make_facade()
    assert hasattr(facade, "submit_onnx_export")
    import asyncio
    assert asyncio.iscoroutinefunction(facade.submit_onnx_export)


def test_facade_has_submit_onnx_validate():
    facade = _make_facade()
    assert hasattr(facade, "submit_onnx_validate")
    import asyncio
    assert asyncio.iscoroutinefunction(facade.submit_onnx_validate)


def test_facade_has_submit_benchmark():
    facade = _make_facade()
    assert hasattr(facade, "submit_benchmark")
    import asyncio
    assert asyncio.iscoroutinefunction(facade.submit_benchmark)


def test_admin_router_has_export_torchscript_endpoint():
    from quantedge_services.core.jobs import JobRegistry
    facade = _make_facade()
    registry = JobRegistry()
    router_obj = FoundationsAdminRouter(facade=facade, registry=registry)
    routes = [r.path for r in router_obj.router.routes]
    assert any("export/torchscript" in p for p in routes)

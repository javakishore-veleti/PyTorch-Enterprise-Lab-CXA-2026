"""Tests for ADR tasks and service — Week 12."""
from __future__ import annotations
import pytest
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    ADRGenerateRequest, ADRGenerateResult,
    ADRListRequest, ADRListResult,
)
from quantedge_services.services.wfs.adr.tasks.adr_generate_task import ADRGeneratorTask
from quantedge_services.services.wfs.adr.tasks.adr_list_task import ADRListTask
from quantedge_services.services.wfs.adr.adr_service import ADRService


# ── ADRGeneratorTask ───────────────────────────────────────────────────────


def test_adr_generate_task_creates_six_files(tmp_path):
    task = ADRGeneratorTask()
    req = ADRGenerateRequest(output_dir=str(tmp_path / "adr"))
    result = task.execute(req)
    assert len(result.generated_files) == 6


def test_adr_generate_task_result_count(tmp_path):
    task = ADRGeneratorTask()
    req = ADRGenerateRequest(output_dir=str(tmp_path / "adr"))
    result = task.execute(req)
    assert isinstance(result, ADRGenerateResult)
    assert result.adr_count == 6
    assert result.status == "completed"


def test_adr_files_contain_status_accepted(tmp_path):
    task = ADRGeneratorTask()
    req = ADRGenerateRequest(output_dir=str(tmp_path / "adr"))
    result = task.execute(req)
    for path in result.generated_files:
        content = open(path).read()
        assert "Accepted" in content


# ── ADRListTask ────────────────────────────────────────────────────────────


def test_adr_list_task_scans_directory(tmp_path):
    gen_task = ADRGeneratorTask()
    gen_task.execute(ADRGenerateRequest(output_dir=str(tmp_path / "adr")))

    list_task = ADRListTask()
    req = ADRListRequest(adr_dir=str(tmp_path / "adr"))
    result = list_task.execute(req)
    assert isinstance(result, ADRListResult)
    assert result.total_count == 6


def test_adr_list_task_parses_id_and_title(tmp_path):
    gen_task = ADRGeneratorTask()
    gen_task.execute(ADRGenerateRequest(output_dir=str(tmp_path / "adr")))

    list_task = ADRListTask()
    req = ADRListRequest(adr_dir=str(tmp_path / "adr"))
    result = list_task.execute(req)
    for adr in result.adrs:
        assert adr["id"].startswith("ADR-")
        assert len(adr["title"]) > 0
        assert "path" in adr


def test_adr_list_empty_dir_returns_empty(tmp_path):
    task = ADRListTask()
    req = ADRListRequest(adr_dir=str(tmp_path / "empty_adr"))
    result = task.execute(req)
    assert result.total_count == 0
    assert result.adrs == []


# ── ADRService ─────────────────────────────────────────────────────────────


def test_adr_service_has_generate_method():
    service = ADRService(ADRGeneratorTask(), ADRListTask())
    assert hasattr(service, "generate_adrs")


def test_adr_service_has_list_method():
    service = ADRService(ADRGeneratorTask(), ADRListTask())
    assert hasattr(service, "list_adrs")


# ── Facade ─────────────────────────────────────────────────────────────────


def test_facade_has_submit_adr_generate():
    from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
    assert hasattr(FoundationsServiceFacade, "submit_adr_generate")


# ── Admin Router ───────────────────────────────────────────────────────────


def test_admin_router_has_adr_generate_endpoint():
    from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
    assert hasattr(FoundationsAdminRouter, "adr_generate")

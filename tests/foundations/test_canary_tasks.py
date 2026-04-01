"""Tests for Week 11 — Canary Deployment tasks, service, and facade."""
from __future__ import annotations
import pytest
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    CanaryDeployRequest, CanaryDeployResult,
    CanaryEvalRequest, CanaryEvalResult,
)
from quantedge_services.services.wfs.canary_deployment.tasks.canary_deploy_task import CanaryDeployTask
from quantedge_services.services.wfs.canary_deployment.tasks.canary_eval_task import CanaryEvalTask
from quantedge_services.services.wfs.canary_deployment.canary_service import CanaryService


def test_canary_deploy_task_creates_routing_file(tmp_path):
    task = CanaryDeployTask()
    req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=20.0,
        deployment_id="test-deploy-1",
        state_dir=str(tmp_path),
    )
    task.execute(req)
    routing_file = tmp_path / "test-deploy-1_routing.json"
    assert routing_file.exists()


def test_canary_deploy_result_is_dto(tmp_path):
    task = CanaryDeployTask()
    req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=10.0,
        deployment_id="test-deploy-2",
        state_dir=str(tmp_path),
    )
    result = task.execute(req)
    assert isinstance(result, CanaryDeployResult)
    assert result.status == "deployed"


def test_canary_deploy_traffic_split_approximate(tmp_path):
    task = CanaryDeployTask()
    pct = 30.0
    req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=pct,
        deployment_id="test-deploy-3",
        state_dir=str(tmp_path),
    )
    result = task.execute(req)
    total = result.baseline_routed + result.candidate_routed
    assert total == 1000
    # Allow 15% tolerance on traffic split
    actual_pct = result.candidate_routed / total * 100
    assert abs(actual_pct - pct) < 15.0


def test_canary_eval_task_returns_decision(tmp_path):
    deploy_task = CanaryDeployTask()
    deploy_req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=20.0,
        deployment_id="eval-test-1",
        state_dir=str(tmp_path),
    )
    deploy_task.execute(deploy_req)

    eval_task = CanaryEvalTask()
    eval_req = CanaryEvalRequest(
        deployment_id="eval-test-1",
        num_eval_requests=10,
        state_dir=str(tmp_path),
    )
    result = eval_task.execute(eval_req)
    assert isinstance(result, CanaryEvalResult)
    assert result.promotion_decision in ("promote", "rollback", "inconclusive")


def test_canary_eval_promotion_decision_valid_values(tmp_path):
    deploy_task = CanaryDeployTask()
    deploy_req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=10.0,
        deployment_id="eval-test-2",
        state_dir=str(tmp_path),
    )
    deploy_task.execute(deploy_req)

    eval_task = CanaryEvalTask()
    eval_req = CanaryEvalRequest(
        deployment_id="eval-test-2",
        num_eval_requests=5,
        state_dir=str(tmp_path),
    )
    result = eval_task.execute(eval_req)
    assert result.promotion_decision in ("promote", "rollback", "inconclusive")


def test_canary_eval_mean_loss_nonnegative(tmp_path):
    deploy_task = CanaryDeployTask()
    deploy_req = CanaryDeployRequest(
        baseline_model_path="/models/base.pt",
        candidate_model_path="/models/candidate.pt",
        canary_traffic_pct=10.0,
        deployment_id="eval-test-3",
        state_dir=str(tmp_path),
    )
    deploy_task.execute(deploy_req)

    eval_task = CanaryEvalTask()
    eval_req = CanaryEvalRequest(
        deployment_id="eval-test-3",
        num_eval_requests=5,
        state_dir=str(tmp_path),
    )
    result = eval_task.execute(eval_req)
    assert result.baseline_mean_loss >= 0.0
    assert result.candidate_mean_loss >= 0.0


def test_canary_service_has_deploy_method():
    svc = CanaryService(deploy_task=CanaryDeployTask(), eval_task=CanaryEvalTask())
    assert hasattr(svc, "deploy")


def test_canary_service_has_eval_method():
    svc = CanaryService(deploy_task=CanaryDeployTask(), eval_task=CanaryEvalTask())
    assert hasattr(svc, "evaluate")


def test_facade_has_submit_canary_deploy():
    from quantedge_services.api.dependencies import get_container
    container = get_container()
    assert hasattr(container.foundations_facade, "submit_canary_deploy")


def test_admin_router_has_canary_deploy_endpoint():
    from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter
    from quantedge_services.api.dependencies import get_container
    container = get_container()
    router_obj = FoundationsAdminRouter(
        facade=container.foundations_facade,
        registry=container.job_registry,
    )
    routes = [r.path for r in router_obj.router.routes]
    assert any("canary/deploy" in r for r in routes)

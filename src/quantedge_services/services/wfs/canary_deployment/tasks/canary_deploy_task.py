"""CanaryDeployTask — simulates probabilistic traffic split between two models."""
from __future__ import annotations
import json
import os
import random
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    CanaryDeployRequest, CanaryDeployResult,
)


class CanaryDeployTask:
    """Simulates routing 1000 requests at canary_traffic_pct% to candidate."""

    _N_REQUESTS = 1000

    def execute(self, request: CanaryDeployRequest) -> CanaryDeployResult:
        baseline_routed = 0
        candidate_routed = 0
        for _ in range(self._N_REQUESTS):
            if random.random() < request.canary_traffic_pct / 100.0:
                candidate_routed += 1
            else:
                baseline_routed += 1

        state_dir = request.state_dir
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, f"{request.deployment_id}_routing.json")
        state = {
            "baseline_routed": baseline_routed,
            "candidate_routed": candidate_routed,
            "canary_traffic_pct": request.canary_traffic_pct,
            "baseline_model_path": request.baseline_model_path,
            "candidate_model_path": request.candidate_model_path,
        }
        with open(state_path, "w") as fh:
            json.dump(state, fh)

        return CanaryDeployResult(
            deployment_id=request.deployment_id,
            canary_traffic_pct=request.canary_traffic_pct,
            baseline_routed=baseline_routed,
            candidate_routed=candidate_routed,
            status="deployed",
        )

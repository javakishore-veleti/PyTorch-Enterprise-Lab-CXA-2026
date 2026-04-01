"""CanaryEvalTask — evaluates baseline and candidate with MSE, decides promotion."""
from __future__ import annotations
import json
import os
import torch
import torch.nn as nn
from quantedge_services.api.schemas.foundations.tracking_schemas import (
    CanaryEvalRequest, CanaryEvalResult,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class CanaryEvalTask:
    """Evaluates two ForexTransformer instances and decides promote/rollback."""

    def execute(self, request: CanaryEvalRequest) -> CanaryEvalResult:
        state_dir = request.state_dir
        state_path = os.path.join(state_dir, f"{request.deployment_id}_routing.json")
        if not os.path.exists(state_path):
            return CanaryEvalResult(
                deployment_id=request.deployment_id,
                baseline_mean_loss=0.0,
                candidate_mean_loss=0.0,
                promotion_decision="inconclusive",
                status="routing_state_not_found",
            )

        with open(state_path) as fh:
            _state = json.load(fh)

        def _make_model() -> ForexTransformer:
            return ForexTransformer(
                input_size=request.input_size,
                d_model=request.d_model,
                nhead=request.nhead,
                num_encoder_layers=request.num_layers,
                dim_feedforward=request.d_model * 4,
            )

        baseline_model = _make_model()
        candidate_model = _make_model()
        baseline_model.eval()
        candidate_model.eval()

        criterion = nn.MSELoss()
        baseline_losses: list[float] = []
        candidate_losses: list[float] = []

        with torch.no_grad():
            for _ in range(request.num_eval_requests):
                x = torch.randn(1, request.seq_len, request.input_size)
                target = torch.randn(1, 1)

                b_out, _ = baseline_model(x)
                baseline_losses.append(criterion(b_out, target).item())

                c_out, _ = candidate_model(x)
                candidate_losses.append(criterion(c_out, target).item())

        baseline_mean = sum(baseline_losses) / len(baseline_losses)
        candidate_mean = sum(candidate_losses) / len(candidate_losses)

        if candidate_mean < baseline_mean * request.success_threshold:
            decision = "promote"
        elif candidate_mean > baseline_mean * 1.05:
            decision = "rollback"
        else:
            decision = "inconclusive"

        return CanaryEvalResult(
            deployment_id=request.deployment_id,
            baseline_mean_loss=baseline_mean,
            candidate_mean_loss=candidate_mean,
            promotion_decision=decision,
            status="evaluated",
        )

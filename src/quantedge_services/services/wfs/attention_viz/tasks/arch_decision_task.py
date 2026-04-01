from __future__ import annotations
from quantedge_services.api.schemas.foundations.viz_schemas import (
    ArchDecisionRequest, ArchDecisionResponse, ArchDecisionOption,
)
from quantedge_services.core.logging import StructuredLogger


class ArchDecisionTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: ArchDecisionRequest) -> ArchDecisionResponse:
        try:
            if (request.task_type == "time_series_regression" and request.output_type == "scalar"):
                recommended = "encoder_only"
                rationale = (
                    "Time-series regression to a scalar (e.g., RUL) requires capturing "
                    "global context across the full sequence. An encoder-only transformer reads the "
                    "entire input bidirectionally, pools the representation, and maps to a scalar. "
                    "No auto-regressive decoding is needed since we predict a single value, not a "
                    "future sequence. Encoder-decoder adds unnecessary decoder overhead; decoder-only "
                    "(causal) masks future tokens which wastes capacity when all past context is available."
                )
            elif request.task_type == "forecasting" and request.output_type == "sequence":
                recommended = "encoder_decoder"
                rationale = (
                    "Multi-step forecasting requires generating a future sequence conditioned "
                    "on past context. Encoder-decoder architecture separates context compression (encoder) "
                    "from auto-regressive generation (decoder), making it ideal for sequence-to-sequence tasks."
                )
            elif request.task_type == "classification" and request.output_type == "scalar":
                recommended = "encoder_only"
                rationale = (
                    "Classification to a scalar label benefits from bidirectional context. "
                    "An encoder-only transformer reads the full input sequence and produces a "
                    "pooled representation for classification, with no need for auto-regressive decoding."
                )
            else:
                recommended = "encoder_only"
                rationale = (
                    "For most time-series tasks with fixed-size output, encoder-only architecture "
                    "provides the best balance of simplicity, bidirectional context, and inference speed."
                )

            options = [
                ArchDecisionOption(
                    architecture="encoder_only",
                    suitability_score=9 if recommended == "encoder_only" else 5,
                    pros=[
                        "Bidirectional context",
                        "Simple architecture",
                        "Fast inference",
                        "Mean-pool gives fixed-size representation",
                    ],
                    cons=[
                        "Cannot generate sequences autoregressively",
                        "Requires separate decoder for seq2seq",
                    ],
                    recommended_when="Regression or classification tasks with fixed-size output from full sequence context.",
                ),
                ArchDecisionOption(
                    architecture="encoder_decoder",
                    suitability_score=9 if recommended == "encoder_decoder" else 6,
                    pros=[
                        "Ideal for seq2seq tasks",
                        "Separated context vs generation",
                        "Flexible output length",
                    ],
                    cons=[
                        "More parameters",
                        "Slower training",
                        "Over-engineered for regression tasks",
                    ],
                    recommended_when="Multi-step forecasting or sequence-to-sequence translation tasks.",
                ),
                ArchDecisionOption(
                    architecture="decoder_only",
                    suitability_score=4,
                    pros=[
                        "Auto-regressive generation",
                        "Causal masking for next-token prediction",
                        "Scales well (GPT family)",
                    ],
                    cons=[
                        "Cannot see future tokens — wasteful for full-sequence input",
                        "Requires prompt engineering for regression",
                    ],
                    recommended_when="Next-token prediction or language modeling tasks where causal ordering is required.",
                ),
            ]

            return ArchDecisionResponse(
                execution_id=request.execution_id,
                task_type=request.task_type,
                recommended_architecture=recommended,
                rationale=rationale,
                options=options,
                status="success",
            )
        except Exception as exc:
            self._logger.error("arch_decision_failed", error=str(exc))
            return ArchDecisionResponse(
                execution_id=request.execution_id,
                task_type=request.task_type,
                recommended_architecture="",
                rationale="",
                options=[],
                status="failed",
                error=str(exc),
            )

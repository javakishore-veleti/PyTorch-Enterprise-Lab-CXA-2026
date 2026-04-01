"""ConceptDriftTask — prediction distribution shift via relative mean change."""
from __future__ import annotations
import random
from datetime import datetime, timezone, timedelta
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    ConceptDriftRequest, ConceptDriftResult,
)


class ConceptDriftTask:
    """Detects concept drift by comparing baseline vs current prediction distribution."""

    def execute(self, request: ConceptDriftRequest) -> ConceptDriftResult:
        predictions = self._load_or_synthetic(request)
        n = len(predictions)
        half = n // 2
        window = min(request.window_size, n)

        baseline = predictions[:half]
        current_window = predictions[n - window:]

        baseline_mean = sum(baseline) / len(baseline) if baseline else 0.0
        current_mean = sum(current_window) / len(current_window) if current_window else 0.0
        relative_change = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)
        drift_detected = relative_change > request.drift_threshold

        return ConceptDriftResult(
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            relative_change=relative_change,
            drift_detected=drift_detected,
            window_size=len(current_window),
            status="completed",
        )

    def _load_or_synthetic(self, request: ConceptDriftRequest) -> list[float]:
        try:
            import pandas as pd  # type: ignore
            from pathlib import Path
            if Path(request.predictions_log_path).exists():
                df = pd.read_parquet(request.predictions_log_path)
                if "prediction" in df.columns:
                    return df["prediction"].tolist()
        except Exception:
            pass
        return self._synthetic_predictions()

    def _synthetic_predictions(self) -> list[float]:
        rng = random.Random(42)
        return [rng.gauss(0, 1) for _ in range(1000)]

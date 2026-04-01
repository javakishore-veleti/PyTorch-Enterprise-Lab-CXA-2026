"""Integration tests for ForexPreprocessTask using synthetic OHLCV DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from quantedge_services.api.schemas.foundations.forex_schemas import ForexPreprocessRequest
from quantedge_services.core.device import DeviceManager
from quantedge_services.services.wfs.forex_eurusd.tasks.preprocess_task import ForexPreprocessTask


def _make_ohlcv(rows: int = 200) -> pd.DataFrame:
    """Return a synthetic OHLCV+spread DataFrame with DatetimeIndex."""
    import numpy as np
    rng = pd.date_range("2023-01-03 09:00", periods=rows, freq="1min")
    np.random.seed(42)
    close = 1.08 + np.cumsum(np.random.randn(rows) * 0.0002)
    spread = abs(np.random.randn(rows)) * 0.0001 + 0.0001
    df = pd.DataFrame(
        {
            "open":   close - abs(np.random.randn(rows)) * 0.0001,
            "high":   close + abs(np.random.randn(rows)) * 0.0002,
            "low":    close - abs(np.random.randn(rows)) * 0.0002,
            "close":  close,
            "volume": abs(np.random.randn(rows)) * 50 + 10,
            "spread": spread,
        },
        index=rng,
    )
    return df


class TestForexPreprocessTask:
    @pytest.fixture()
    def task(self) -> ForexPreprocessTask:
        return ForexPreprocessTask(device_manager=DeviceManager(prefer_gpu=False))

    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return _make_ohlcv(200)

    # ── Status ──────────────────────────────────────────────────────────────

    @pytest.mark.integration
    def test_returns_success_status(self, task: ForexPreprocessTask, df: pd.DataFrame, tmp_path: object) -> None:
        request = ForexPreprocessRequest(execution_id="pp-001")
        resp, tensor = task.execute(request, df)
        assert resp.status == "success", resp.error

    # ── Tensor shape ────────────────────────────────────────────────────────

    @pytest.mark.integration
    def test_tensor_has_correct_row_count(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(execution_id="pp-002")
        resp, tensor = task.execute(request, df)
        assert tensor.shape[0] == resp.total_bars

    @pytest.mark.integration
    def test_tensor_has_feature_columns(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(execution_id="pp-003")
        resp, tensor = task.execute(request, df)
        assert tensor.shape[1] == len(resp.features)

    # ── Train / val / test split ────────────────────────────────────────────

    @pytest.mark.integration
    def test_split_rows_sum_to_total(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(
            execution_id="pp-004", train_ratio=0.70, val_ratio=0.15
        )
        resp, _ = task.execute(request, df)
        assert resp.train_bars + resp.val_bars + resp.test_bars == resp.total_bars

    @pytest.mark.integration
    def test_train_ratio_is_approximately_correct(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(
            execution_id="pp-005", train_ratio=0.70, val_ratio=0.15
        )
        resp, _ = task.execute(request, df)
        actual_ratio = resp.train_bars / resp.total_bars
        assert abs(actual_ratio - 0.70) < 0.02, f"Train ratio off: {actual_ratio:.3f}"

    # ── Normalisation ───────────────────────────────────────────────────────

    @pytest.mark.integration
    def test_minmax_values_in_unit_range(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(
            execution_id="pp-006", normalize=True, scaler_type="minmax"
        )
        _, tensor = task.execute(request, df)
        assert tensor.min().item() >= -1e-6
        assert tensor.max().item() <= 1.0 + 1e-6

    @pytest.mark.integration
    def test_zscore_mean_near_zero(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(
            execution_id="pp-007", normalize=True, scaler_type="zscore"
        )
        _, tensor = task.execute(request, df)
        col_means = tensor.mean(dim=0)
        assert col_means.abs().max().item() < 0.5, "Z-score mean should be near 0"

    # ── Features list ───────────────────────────────────────────────────────

    @pytest.mark.integration
    def test_features_list_contains_ohlcv(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        request = ForexPreprocessRequest(execution_id="pp-008")
        resp, _ = task.execute(request, df)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in resp.features

    # ── NaN handling ────────────────────────────────────────────────────────

    @pytest.mark.integration
    def test_nan_filled_count_is_non_negative(self, task: ForexPreprocessTask, df: pd.DataFrame) -> None:
        import numpy as np
        df_with_nans = df.copy()
        df_with_nans.iloc[10:15, 0] = float("nan")  # inject NaNs in 'open'
        request = ForexPreprocessRequest(execution_id="pp-009", fill_gaps=True)
        resp, _ = task.execute(request, df_with_nans)
        assert resp.nan_filled >= 5

"""Integration tests for ForexIngestionTask using synthetic HistData tick data."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from quantedge_services.api.schemas.foundations.forex_schemas import ForexIngestionRequest
from quantedge_services.services.wfs.forex_eurusd.tasks.ingest_task import ForexIngestionTask


class TestForexIngestionTask:
    """Unit / integration tests for ForexIngestionTask.

    These tests create tiny synthetic CSVs (no network access) to exercise the
    real parsing and resampling logic.  They are marked ``integration`` so the
    default ``pytest`` run (which excludes that marker) keeps CI fast; run them
    explicitly with ``pytest -m integration``.
    """

    @pytest.fixture()
    def task(self) -> ForexIngestionTask:
        return ForexIngestionTask()

    @pytest.fixture()
    def tick_csv(self, tmp_path: Path) -> Path:
        """Write a minimal 3-column HistData tick CSV (YYYYMMDD HHmmss,bid,ask)."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        csv_file = raw_dir / "EURUSD_2023.csv"
        # 60 ticks spread over 1 minute (2 bars at 1-min resampling)
        rows = []
        for sec in range(60):
            dt = f"20230103 09{sec // 60:02d}{sec % 60:02d}00" if sec < 60 else f"20230103 090100"
            bid = 1.0800 + sec * 0.0001
            ask = bid + 0.0002
            rows.append(f"20230103 090{sec:04d},{bid:.5f},{ask:.5f}")
        csv_file.write_text("\n".join(rows))
        return raw_dir

    @pytest.fixture()
    def ohlcv_csv(self, tmp_path: Path) -> Path:
        """Write a minimal 6-column pre-built OHLCV CSV."""
        raw_dir = tmp_path / "raw_ohlcv"
        raw_dir.mkdir()
        csv_file = raw_dir / "EURUSD_OHLCV_2023.csv"
        rows = [
            "20230103 090000,1.0800,1.0850,1.0790,1.0830,120",
            "20230103 090100,1.0830,1.0870,1.0820,1.0860,98",
            "20230103 090200,1.0860,1.0900,1.0855,1.0880,75",
        ]
        csv_file.write_text("\n".join(rows))
        return raw_dir

    # ── Tick-format tests ───────────────────────────────────────────────────

    @pytest.mark.integration
    def test_tick_format_returns_success(self, task: ForexIngestionTask, tick_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(tick_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.status == "success", resp.error

    @pytest.mark.integration
    def test_tick_format_produces_parquet(self, task: ForexIngestionTask, tick_csv: Path, tmp_path: Path) -> None:
        parquet_dir = tmp_path / "parquet"
        request = ForexIngestionRequest(
            data_dir=str(tick_csv),
            parquet_dir=str(parquet_dir),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert Path(resp.parquet_path).exists(), "Parquet file was not created"

    @pytest.mark.integration
    def test_tick_format_parquet_has_ohlcv_columns(self, task: ForexIngestionTask, tick_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(tick_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        df = pd.read_parquet(resp.parquet_path)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Column '{col}' missing from parquet"

    @pytest.mark.integration
    def test_tick_format_bar_count(self, task: ForexIngestionTask, tick_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(tick_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.bars_resampled >= 1

    @pytest.mark.integration
    def test_tick_ticks_loaded_matches_rows(self, task: ForexIngestionTask, tick_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(tick_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.ticks_loaded == 60

    # ── OHLCV-format tests ──────────────────────────────────────────────────

    @pytest.mark.integration
    def test_ohlcv_format_returns_success(self, task: ForexIngestionTask, ohlcv_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(ohlcv_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.status == "success", resp.error

    @pytest.mark.integration
    def test_ohlcv_format_bar_count(self, task: ForexIngestionTask, ohlcv_csv: Path, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(ohlcv_csv),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.bars_resampled == 3

    # ── Error-handling tests ────────────────────────────────────────────────

    def test_missing_data_dir_returns_failed_status(self, task: ForexIngestionTask, tmp_path: Path) -> None:
        request = ForexIngestionRequest(
            data_dir=str(tmp_path / "nonexistent"),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.status == "failed"
        assert resp.error is not None

    def test_empty_data_dir_returns_failed_status(self, task: ForexIngestionTask, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        request = ForexIngestionRequest(
            data_dir=str(empty_dir),
            parquet_dir=str(tmp_path / "parquet"),
            resample_freq="1min",
        )
        resp = task.execute(request)
        assert resp.status == "failed"

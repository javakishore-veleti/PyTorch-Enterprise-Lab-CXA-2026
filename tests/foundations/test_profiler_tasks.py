from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_parquet(tmp_path: Path, n: int = 500, n_features: int = 5) -> Path:
    df = pd.DataFrame(np.random.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    df["target"] = np.random.randn(n)
    p = tmp_path / "train.parquet"
    df.to_parquet(p)
    return p


class TestProfilerRunTask:
    def test_profiler_creates_trace_dir(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.profiler_run_task import ProfilerRunTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import ProfilerRunRequest
        parquet_path = _make_parquet(tmp_path)
        trace_dir = tmp_path / "traces"
        task = ProfilerRunTask()
        req = ProfilerRunRequest(
            execution_id="test-001",
            parquet_path=str(parquet_path),
            model_type="mlp",
            wait_steps=0,
            warmup_steps=0,
            active_steps=2,
            trace_dir=str(trace_dir),
        )
        resp = task.execute(req)
        assert trace_dir.exists()

    def test_profiler_returns_step_count(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.profiler_run_task import ProfilerRunTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import ProfilerRunRequest
        parquet_path = _make_parquet(tmp_path)
        trace_dir = tmp_path / "traces2"
        task = ProfilerRunTask()
        req = ProfilerRunRequest(
            execution_id="test-002",
            parquet_path=str(parquet_path),
            model_type="mlp",
            wait_steps=0,
            warmup_steps=0,
            active_steps=3,
            trace_dir=str(trace_dir),
        )
        resp = task.execute(req)
        assert resp.steps_profiled == 3

    def test_profiler_avg_step_ms_positive(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.profiler_run_task import ProfilerRunTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import ProfilerRunRequest
        parquet_path = _make_parquet(tmp_path)
        trace_dir = tmp_path / "traces3"
        task = ProfilerRunTask()
        req = ProfilerRunRequest(
            execution_id="test-003",
            parquet_path=str(parquet_path),
            model_type="mlp",
            wait_steps=0,
            warmup_steps=0,
            active_steps=2,
            trace_dir=str(trace_dir),
        )
        resp = task.execute(req)
        assert resp.status == "success"
        assert resp.avg_step_ms > 0


class TestMemorySummaryTask:
    def test_memory_summary_returns_success(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.memory_summary_task import MemorySummaryTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import MemorySummaryRequest
        parquet_path = _make_parquet(tmp_path)
        task = MemorySummaryTask()
        req = MemorySummaryRequest(
            execution_id="mem-001",
            parquet_path=str(parquet_path),
            model_type="mlp",
            batch_size=64,
        )
        resp = task.execute(req)
        assert resp.status == "success"

    def test_memory_summary_device_is_string(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.memory_summary_task import MemorySummaryTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import MemorySummaryRequest
        parquet_path = _make_parquet(tmp_path)
        task = MemorySummaryTask()
        req = MemorySummaryRequest(
            execution_id="mem-002",
            parquet_path=str(parquet_path),
            model_type="mlp",
            batch_size=64,
        )
        resp = task.execute(req)
        assert resp.device in ("cpu", "cuda")


class TestDataloaderTuneTask:
    def test_tune_returns_best_workers(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.dataloader_tune_task import DataloaderTuneTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import DataloaderTuneRequest
        parquet_path = _make_parquet(tmp_path, n=200)
        task = DataloaderTuneTask()
        req = DataloaderTuneRequest(
            execution_id="dl-001",
            parquet_path=str(parquet_path),
            batch_size=32,
            num_workers_sweep=[0, 1],
            pin_memory=False,
        )
        resp = task.execute(req)
        assert resp.best_num_workers in (0, 1)

    def test_tune_speedup_is_positive(self, tmp_path):
        from quantedge_services.services.wfs.profiling.tasks.dataloader_tune_task import DataloaderTuneTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import DataloaderTuneRequest
        parquet_path = _make_parquet(tmp_path, n=200)
        task = DataloaderTuneTask()
        req = DataloaderTuneRequest(
            execution_id="dl-002",
            parquet_path=str(parquet_path),
            batch_size=32,
            num_workers_sweep=[0, 1],
            pin_memory=False,
        )
        resp = task.execute(req)
        assert resp.speedup_vs_single > 0


class TestCICIoTIngestionTask:
    def test_ingest_skips_gracefully_when_no_csvs(self, tmp_path):
        from quantedge_services.services.wfs.cic_iot.tasks.ingest_task import CICIoTIngestionTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import CICIoTIngestionRequest
        empty_dir = tmp_path / "raw_empty"
        empty_dir.mkdir()
        task = CICIoTIngestionTask()
        req = CICIoTIngestionRequest(
            execution_id="cic-ingest-001",
            raw_dir=str(empty_dir),
            parquet_dir=str(tmp_path / "parquet"),
        )
        resp = task.execute(req)
        assert resp.status == "skipped"
        assert resp.rows_loaded == 0

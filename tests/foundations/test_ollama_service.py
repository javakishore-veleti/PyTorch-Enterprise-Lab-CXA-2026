"""Week 8 — test_ollama_service.py (10 tests)."""
from __future__ import annotations
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest,
    StackOverflowIngestionRequest,
    DomainAdaptTrainRequest,
    DomainAdaptEvalRequest,
    OllamaInferRequest,
    OllamaMergeRequest,
    StackOverflowDownloadResult,
    StackOverflowIngestionResult,
    DomainAdaptTrainResult,
    DomainAdaptEvalResult,
    OllamaInferResult,
    OllamaMergeResult,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_text_parquet(tmp_path: Path, n: int = 8) -> Path:
    rows = [{"text": f"Enterprise text row {i}."} for i in range(n)]
    p = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


# ── Service async tests ────────────────────────────────────────────────────────

class TestStackOverflowServiceAsync:
    def test_stackoverflow_service_download_async(self, tmp_path):
        from quantedge_services.services.wfs.stackoverflow.stackoverflow_service import StackOverflowService

        expected = StackOverflowDownloadResult(output_path="/tmp/x.parquet", record_count=20, status="success")
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = StackOverflowService(download_task=mock_task, ingest_task=MagicMock())

        req = StackOverflowDownloadRequest(dataset_source="kaggle", output_dir=str(tmp_path))
        result = asyncio.get_event_loop().run_until_complete(svc.download(req))
        assert result.status == "success"
        assert result.record_count == 20

    def test_stackoverflow_service_ingest_async(self, tmp_path):
        from quantedge_services.services.wfs.stackoverflow.stackoverflow_service import StackOverflowService

        expected = StackOverflowIngestionResult(output_path="/tmp/filtered.parquet", filtered_count=7, status="success")
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = StackOverflowService(download_task=MagicMock(), ingest_task=mock_task)

        req = StackOverflowIngestionRequest(input_path="/tmp/x.parquet", output_dir=str(tmp_path), tags_filter=["java"])
        result = asyncio.get_event_loop().run_until_complete(svc.ingest(req))
        assert result.filtered_count == 7


class TestDomainAdaptServiceAsync:
    def test_domain_adapt_service_train_async(self, tmp_path):
        from quantedge_services.services.wfs.domain_adaptation.domain_adapt_service import DomainAdaptService

        expected = DomainAdaptTrainResult(
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            train_loss=1.5,
            steps=10,
            status="success",
        )
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = DomainAdaptService(train_task=mock_task, eval_task=MagicMock())

        req = DomainAdaptTrainRequest(
            data_path=str(tmp_path / "data.parquet"),
            model_name="gpt2",
            output_dir=str(tmp_path),
            lora_rank=4,
            lora_alpha=8,
            max_steps=10,
            batch_size=2,
        )
        result = asyncio.get_event_loop().run_until_complete(svc.train(req))
        assert result.train_loss == 1.5
        assert result.steps == 10

    def test_domain_adapt_service_eval_async(self, tmp_path):
        from quantedge_services.services.wfs.domain_adaptation.domain_adapt_service import DomainAdaptService

        expected = DomainAdaptEvalResult(eval_loss=1.2, perplexity=3.32, status="success")
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = DomainAdaptService(train_task=MagicMock(), eval_task=mock_task)

        req = DomainAdaptEvalRequest(
            data_path=str(tmp_path / "data.parquet"),
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            model_name="gpt2",
            lora_rank=4,
            lora_alpha=8,
        )
        result = asyncio.get_event_loop().run_until_complete(svc.evaluate(req))
        assert result.perplexity == pytest.approx(3.32)


class TestOllamaServiceAsync:
    def test_ollama_service_infer_async(self):
        from quantedge_services.services.wfs.ollama_serving.ollama_service import OllamaService

        expected = OllamaInferResult(response="A good answer.", status="success", latency_ms=42.0)
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = OllamaService(infer_task=mock_task, merge_task=MagicMock())

        req = OllamaInferRequest(model_name="llama3", prompt="Hello", ollama_base_url="http://localhost:11434")
        result = asyncio.get_event_loop().run_until_complete(svc.infer(req))
        assert result.status == "success"
        assert result.latency_ms == 42.0

    def test_ollama_service_merge_async(self, tmp_path):
        from quantedge_services.services.wfs.ollama_serving.ollama_service import OllamaService

        expected = OllamaMergeResult(
            merged_model_path=str(tmp_path / "merged"),
            modelfile_path=str(tmp_path / "modelfile.txt"),
            status="success",
        )
        mock_task = MagicMock()
        mock_task.execute.return_value = expected
        svc = OllamaService(infer_task=MagicMock(), merge_task=mock_task)

        req = OllamaMergeRequest(
            adapter_checkpoint_path=str(tmp_path / "ckpt.pt"),
            base_model_name="gpt2",
            output_dir=str(tmp_path / "merged"),
            ollama_model_name="qe-model",
        )
        result = asyncio.get_event_loop().run_until_complete(svc.merge(req))
        assert result.status == "success"


# ── Facade attribute checks ────────────────────────────────────────────────────

class TestFacadeMethods:
    def _make_facade(self):
        from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade

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
            quantization_service=MagicMock(),
            serving_service=MagicMock(),
        )

    def test_facade_has_stackoverflow_download_method(self):
        facade = self._make_facade()
        assert callable(getattr(facade, "submit_stackoverflow_download", None))

    def test_facade_has_domain_adapt_train_method(self):
        facade = self._make_facade()
        assert callable(getattr(facade, "submit_domain_adapt_train", None))

    def test_facade_has_ollama_infer_method(self):
        facade = self._make_facade()
        assert callable(getattr(facade, "submit_ollama_infer", None))


# ── Router endpoint check ──────────────────────────────────────────────────────

class TestAdminRouterEndpoints:
    def test_admin_router_has_ollama_infer_endpoint(self):
        from quantedge_services.api.routers.admin.foundations_router import FoundationsAdminRouter

        router_obj = FoundationsAdminRouter(facade=MagicMock(), registry=MagicMock())
        routes = [r.path for r in router_obj.router.routes]
        assert any("ollama/infer" in p for p in routes)

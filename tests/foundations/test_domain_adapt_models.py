"""Week 8 — test_domain_adapt_models.py (15 tests)."""
from __future__ import annotations
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest,
    StackOverflowIngestionRequest,
    DomainAdaptTrainRequest,
    DomainAdaptEvalRequest,
    OllamaInferRequest,
    OllamaMergeRequest,
)
from quantedge_services.services.wfs.stackoverflow.tasks.download_task import StackOverflowDownloadTask
from quantedge_services.services.wfs.stackoverflow.tasks.ingest_task import StackOverflowIngestionTask


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_synthetic_parquet(tmp_path: Path) -> Path:
    """Write a small synthetic StackOverflow parquet for use in tests."""
    rows = []
    tags_pool = ["java", "python", "javascript"]
    for i in range(20):
        tag = tags_pool[i % len(tags_pool)]
        rows.append(
            {
                "id": i + 1,
                "post_type": "question",
                "tags": f"<{tag}>",
                "title": f"How to do {tag} thing {i}?",
                "body": f"<p>Details about {tag}.</p>",
                "score": i,
                "answer_count": i % 3,
                "text": f"How to do {tag} thing {i}? Details about {tag}.",
            }
        )
    p = tmp_path / "stackoverflow_raw.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _make_text_parquet(tmp_path: Path, n: int = 10) -> Path:
    """Write a parquet with a ``text`` column for domain-adapt tasks."""
    rows = [{"text": f"Sample enterprise text sample number {i}."} for i in range(n)]
    p = tmp_path / "text_data.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


# ── PEFTModelWrapper ───────────────────────────────────────────────────────────

class TestPEFTModelWrapperWithoutPeft:
    def test_peft_model_wrapper_init_without_peft_raises(self):
        """If peft is absent the wrapper must raise RuntimeError."""
        import sys
        peft_backup = sys.modules.pop("peft", None)
        transformers_backup = sys.modules.pop("transformers", None)
        try:
            # Force the module to be re-imported without peft
            import importlib
            import quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper as m
            # Patch the _PEFT_AVAILABLE flag inside the already-loaded module
            original = m._PEFT_AVAILABLE
            m._PEFT_AVAILABLE = False
            try:
                with pytest.raises(RuntimeError, match="peft not installed"):
                    m.PEFTModelWrapper(
                        base_model_name="gpt2",
                        lora_rank=4,
                        lora_alpha=8,
                        target_modules=["c_attn"],
                    )
            finally:
                m._PEFT_AVAILABLE = original
        finally:
            if peft_backup is not None:
                sys.modules["peft"] = peft_backup
            if transformers_backup is not None:
                sys.modules["transformers"] = transformers_backup


class TestPEFTModelWrapperMocked:
    def _build_wrapper(self):
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            MagicMock(numel=lambda: 512, requires_grad=True),
            MagicMock(numel=lambda: 256, requires_grad=False),
        ]
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = mock_model.parameters()
        mock_peft_model.merge_and_unload.return_value = MagicMock()

        with (
            patch(
                "quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper.AutoModelForCausalLM"
            ) as mock_auto,
            patch(
                "quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper.get_peft_model",
                return_value=mock_peft_model,
            ),
            patch(
                "quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper.LoraConfig"
            ),
            patch(
                "quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper._PEFT_AVAILABLE",
                True,
            ),
        ):
            import quantedge_services.services.wfs.domain_adaptation.models.peft_model_wrapper as m
            m._PEFT_AVAILABLE = True
            wrapper = m.PEFTModelWrapper(
                base_model_name="gpt2",
                lora_rank=8,
                lora_alpha=16,
                target_modules=["c_attn"],
            )
        return wrapper

    def test_peft_model_wrapper_trainable_params(self):
        """get_trainable_param_count returns a non-negative integer."""
        wrapper = self._build_wrapper()
        count = wrapper.get_trainable_param_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_lora_config_target_modules(self):
        """Wrapper stores the target_modules it receives."""
        wrapper = self._build_wrapper()
        assert wrapper._target_modules == ["c_attn"]


# ── DomainAdaptTrainTask ───────────────────────────────────────────────────────

class TestDomainAdaptTrainTask:
    @staticmethod
    def _make_model_mock(loss_val: float = 1.5):
        import torch

        mock_output = MagicMock()
        mock_output.loss = torch.tensor(loss_val, requires_grad=True)

        mock_peft_model = MagicMock()
        mock_peft_model.train.return_value = None
        mock_peft_model.state_dict.return_value = {"lora_A_weight": torch.zeros(4)}
        mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(4))]
        mock_peft_model.return_value = mock_output  # controls mock_peft_model(...) return
        return mock_peft_model

    @staticmethod
    def _make_tokenizer_mock():
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        return mock_tokenizer

    def _run_train(self, tmp_path: Path, loss_val: float = 1.5, max_steps: int = 2) -> "DomainAdaptTrainResult":
        from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_train_task import DomainAdaptTrainTask

        parquet = _make_text_parquet(tmp_path)
        task = DomainAdaptTrainTask()
        req = DomainAdaptTrainRequest(
            data_path=str(parquet),
            model_name="gpt2",
            output_dir=str(tmp_path / "ckpt"),
            lora_rank=4,
            lora_alpha=8,
            max_steps=max_steps,
            batch_size=2,
        )

        mock_peft_model = self._make_model_mock(loss_val)
        mock_tokenizer = self._make_tokenizer_mock()

        with (
            patch(
                "quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_train_task.AutoTokenizer"
            ) as mock_tok_cls,
            patch(
                "quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_train_task.PEFTModelWrapper"
            ) as mock_wrapper_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            instance = MagicMock()
            instance.model = mock_peft_model
            mock_wrapper_cls.return_value = instance
            return task.execute(req)

    def test_domain_adapt_train_task_returns_result_dto(self, tmp_path):
        result = self._run_train(tmp_path)
        assert result.status == "success"
        assert isinstance(result.train_loss, float)
        assert result.steps > 0

    def test_domain_adapt_train_task_checkpoint_saved(self, tmp_path):
        result = self._run_train(tmp_path, max_steps=1)
        assert result.status == "success"
        assert Path(result.checkpoint_path).exists()

    def test_domain_adapt_train_loss_is_float(self, tmp_path):
        result = self._run_train(tmp_path, loss_val=2.3, max_steps=1)
        assert isinstance(result.train_loss, float)


# ── DomainAdaptEvalTask ────────────────────────────────────────────────────────

class TestDomainAdaptEvalTask:
    def _run_eval(self, tmp_path: Path):
        import torch
        from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_eval_task import DomainAdaptEvalTask

        parquet = _make_text_parquet(tmp_path)
        ckpt_path = tmp_path / "adapter_checkpoint.pt"
        torch.save({"adapter_state_dict": {}, "config": {"model_name": "gpt2", "lora_rank": 4, "lora_alpha": 8}}, ckpt_path)

        task = DomainAdaptEvalTask()
        req = DomainAdaptEvalRequest(
            data_path=str(parquet),
            checkpoint_path=str(ckpt_path),
            model_name="gpt2",
            lora_rank=4,
            lora_alpha=8,
        )

        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.8)

        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = None
        mock_peft_model.state_dict.return_value = {}
        mock_peft_model.load_state_dict = MagicMock()
        mock_peft_model.return_value = mock_output  # controls mock_peft_model(...) return

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}

        with (
            patch(
                "quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_eval_task.AutoTokenizer"
            ) as mock_tok_cls,
            patch(
                "quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_eval_task.PEFTModelWrapper"
            ) as mock_wrapper_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            instance = MagicMock()
            instance.model = mock_peft_model
            mock_wrapper_cls.return_value = instance
            return task.execute(req)

    def test_domain_adapt_eval_task_returns_result_dto(self, tmp_path):
        result = self._run_eval(tmp_path)
        assert result.status == "success"
        assert isinstance(result.eval_loss, float)
        assert isinstance(result.perplexity, float)

    def test_domain_adapt_eval_perplexity_positive(self, tmp_path):
        result = self._run_eval(tmp_path)
        assert result.perplexity > 0.0


# ── OllamaMergeTask ────────────────────────────────────────────────────────────

class TestOllamaMergeTask:
    def _make_checkpoint(self, tmp_path: Path) -> Path:
        import torch
        ckpt = tmp_path / "adapter.pt"
        torch.save(
            {
                "adapter_state_dict": {},
                "config": {"model_name": "gpt2", "lora_rank": 4, "lora_alpha": 8},
            },
            ckpt,
        )
        return ckpt

    def test_ollama_merge_task_creates_modelfile(self, tmp_path):
        from quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task import OllamaMergeTask

        ckpt = self._make_checkpoint(tmp_path)
        task = OllamaMergeTask()
        req = OllamaMergeRequest(
            adapter_checkpoint_path=str(ckpt),
            base_model_name="gpt2",
            output_dir=str(tmp_path / "merged"),
            ollama_model_name="test-model",
        )

        mock_merged = MagicMock()
        mock_merged.state_dict.return_value = {}

        with patch(
            "quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task.PEFTModelWrapper"
        ) as mock_wrapper_cls:
            instance = MagicMock()
            instance.model.state_dict.return_value = {}
            instance.model.load_state_dict = MagicMock()
            instance.merge_and_unload.return_value = mock_merged
            mock_wrapper_cls.return_value = instance
            result = task.execute(req)

        assert Path(result.modelfile_path).exists()
        content = Path(result.modelfile_path).read_text()
        assert "FROM" in content

    def test_ollama_merge_task_merged_path_set(self, tmp_path):
        from quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task import OllamaMergeTask

        ckpt = self._make_checkpoint(tmp_path)
        task = OllamaMergeTask()
        req = OllamaMergeRequest(
            adapter_checkpoint_path=str(ckpt),
            base_model_name="gpt2",
            output_dir=str(tmp_path / "merged2"),
            ollama_model_name="test-model-2",
        )

        mock_merged = MagicMock()
        mock_merged.state_dict.return_value = {}

        with patch(
            "quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task.PEFTModelWrapper"
        ) as mock_wrapper_cls:
            instance = MagicMock()
            instance.model.state_dict.return_value = {}
            instance.model.load_state_dict = MagicMock()
            instance.merge_and_unload.return_value = mock_merged
            mock_wrapper_cls.return_value = instance
            result = task.execute(req)

        assert result.merged_model_path != ""
        assert result.status == "success"


# ── OllamaInferTask ────────────────────────────────────────────────────────────

class TestOllamaInferTask:
    def test_ollama_infer_task_success(self):
        from quantedge_services.services.wfs.ollama_serving.tasks.ollama_infer_task import OllamaInferTask

        task = OllamaInferTask()
        req = OllamaInferRequest(
            model_name="llama3",
            prompt="What is Spring Boot?",
            max_tokens=64,
            temperature=0.7,
            ollama_base_url="http://localhost:11434",
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Spring Boot is a framework."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = task.execute(req)

        assert result.status == "success"
        assert result.response == "Spring Boot is a framework."

    def test_ollama_infer_task_connection_error_returns_unavailable(self):
        from quantedge_services.services.wfs.ollama_serving.tasks.ollama_infer_task import OllamaInferTask
        import requests as req_mod

        task = OllamaInferTask()
        req = OllamaInferRequest(
            model_name="llama3",
            prompt="Test prompt",
            ollama_base_url="http://localhost:11434",
        )

        with patch("requests.post", side_effect=req_mod.ConnectionError("refused")):
            result = task.execute(req)

        assert result.status == "ollama_unavailable"
        assert result.response == ""

    def test_ollama_infer_latency_ms_nonnegative(self):
        from quantedge_services.services.wfs.ollama_serving.tasks.ollama_infer_task import OllamaInferTask

        task = OllamaInferTask()
        req = OllamaInferRequest(
            model_name="llama3",
            prompt="Latency test",
            ollama_base_url="http://localhost:11434",
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = task.execute(req)

        assert result.latency_ms >= 0.0


# ── StackOverflow tasks ────────────────────────────────────────────────────────

class TestStackOverflowDownloadTask:
    def test_stackoverflow_download_task_creates_parquet(self, tmp_path):
        task = StackOverflowDownloadTask()
        req = StackOverflowDownloadRequest(
            dataset_source="kaggle",
            output_dir=str(tmp_path / "so"),
        )
        result = task.execute(req)
        assert result.status == "success"
        assert Path(result.output_path).exists()
        df = pd.read_parquet(result.output_path)
        assert len(df) == result.record_count
        assert "tags" in df.columns


class TestStackOverflowIngestionTask:
    def test_stackoverflow_ingest_task_filters_by_tags(self, tmp_path):
        raw = _make_synthetic_parquet(tmp_path)
        task = StackOverflowIngestionTask()
        req = StackOverflowIngestionRequest(
            input_path=str(raw),
            output_dir=str(tmp_path / "filtered"),
            tags_filter=["java"],
        )
        result = task.execute(req)
        assert result.status == "success"
        assert result.filtered_count > 0
        df_filtered = pd.read_parquet(result.output_path)
        assert all("java" in str(t).lower() for t in df_filtered["tags"])

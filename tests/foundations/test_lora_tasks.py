from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest
from quantedge_services.api.schemas.foundations.lora_schemas import (
    LoRAConfig, LoRATrainRequest, LoRAEvalRequest, LoRAPredictRequest,
    OAsst1IngestionRequest,
)
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_train_task import LoRATrainTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_eval_task import LoRAEvalTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_predict_task import LoRAPredictTask
from quantedge_services.services.wfs.oasst1.tasks.ingest_task import OAsst1IngestionTask
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


def _make_base_checkpoint(tmp_path, seq_len=5, input_size=4, d_model=16, nhead=2, layers=2):
    model = ForexTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=layers,
        dim_feedforward=32,
        dropout=0.0,
    )
    p = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": layers,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "seq_len": seq_len,
        },
        p,
    )
    return p, model


def _make_parquet(tmp_path, n=200, seq_len=5, input_size=4):
    feature_cols = [f"f{i}" for i in range(seq_len * input_size)]
    data = {c: np.random.randn(n).astype(np.float32) for c in feature_cols}
    data["rul"] = np.random.randint(0, 100, n).astype(float)
    data["engine_id"] = np.ones(n, dtype=int)
    df = pd.DataFrame(data)
    p = tmp_path / "train.parquet"
    df.to_parquet(p)
    return p


_SEQ_LEN = 5
_INPUT_SIZE = 4
_LORA_CFG = LoRAConfig(rank=4, alpha=8.0, dropout=0.0, target_modules=["w_q", "w_k", "w_v", "w_o"])


class TestLoRATrainTask:
    def test_train_creates_lora_checkpoint(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        task = LoRATrainTask()
        req = LoRATrainRequest(
            execution_id="test-train-01",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=2,
            batch_size=32,
            patience=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        resp = task.execute(req)
        assert resp.status == "success", resp.error
        assert Path(resp.lora_checkpoint_path).exists()

    def test_train_trainable_params_less_than_total(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        task = LoRATrainTask()
        req = LoRATrainRequest(
            execution_id="test-train-02",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            batch_size=32,
            patience=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        resp = task.execute(req)
        assert resp.status == "success", resp.error
        assert resp.trainable_params < resp.trainable_params + resp.frozen_params
        assert resp.trainable_params > 0
        assert resp.frozen_params > 0

    def test_train_skips_gracefully_on_missing_base_checkpoint(self, tmp_path):
        task = LoRATrainTask()
        req = LoRATrainRequest(
            execution_id="test-train-03",
            base_checkpoint_path=str(tmp_path / "nonexistent.pt"),
            parquet_path=str(tmp_path / "train.parquet"),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        resp = task.execute(req)
        assert resp.status == "skipped"

    def test_train_skips_gracefully_on_missing_parquet(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        task = LoRATrainTask()
        req = LoRATrainRequest(
            execution_id="test-train-04",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(tmp_path / "nonexistent.parquet"),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        resp = task.execute(req)
        assert resp.status == "skipped"

    def test_trainable_ratio_between_zero_and_one(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        task = LoRATrainTask()
        req = LoRATrainRequest(
            execution_id="test-train-05",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        resp = task.execute(req)
        assert resp.status == "success", resp.error
        assert 0.0 < resp.trainable_ratio < 1.0


class TestLoRAEvalTask:
    def test_eval_skips_on_missing_lora_checkpoint(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        task = LoRAEvalTask()
        req = LoRAEvalRequest(
            execution_id="test-eval-01",
            base_checkpoint_path=str(base_ckpt),
            lora_checkpoint_path=str(tmp_path / "nonexistent_lora.pt"),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
        )
        resp = task.execute(req)
        assert resp.status == "skipped"

    def test_eval_skips_on_missing_base_checkpoint(self, tmp_path):
        task = LoRAEvalTask()
        req = LoRAEvalRequest(
            execution_id="test-eval-02",
            base_checkpoint_path=str(tmp_path / "nonexistent_base.pt"),
            lora_checkpoint_path=str(tmp_path / "nonexistent_lora.pt"),
            parquet_path=str(tmp_path / "train.parquet"),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
        )
        resp = task.execute(req)
        assert resp.status == "skipped"

    def test_eval_runs_with_valid_checkpoints(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        # First train to get a LoRA checkpoint
        train_task = LoRATrainTask()
        train_req = LoRATrainRequest(
            execution_id="test-eval-pre",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        train_resp = train_task.execute(train_req)
        assert train_resp.status == "success"

        eval_task = LoRAEvalTask()
        eval_req = LoRAEvalRequest(
            execution_id="test-eval-03",
            base_checkpoint_path=str(base_ckpt),
            lora_checkpoint_path=train_resp.lora_checkpoint_path,
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
        )
        resp = eval_task.execute(eval_req)
        assert resp.status == "success", resp.error
        assert resp.rmse >= 0.0
        assert resp.trainable_params > 0


class TestLoRAPredictTask:
    def test_predict_skips_on_missing_checkpoint(self, tmp_path):
        task = LoRAPredictTask()
        req = LoRAPredictRequest(
            execution_id="test-pred-01",
            base_checkpoint_path=str(tmp_path / "nonexistent_base.pt"),
            lora_checkpoint_path=str(tmp_path / "nonexistent_lora.pt"),
            sequences=[[[0.0] * _INPUT_SIZE] * _SEQ_LEN],
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
        )
        resp = task.execute(req)
        assert resp.status == "skipped"

    def test_predict_output_length(self, tmp_path):
        base_ckpt, _ = _make_base_checkpoint(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        parquet = _make_parquet(tmp_path, seq_len=_SEQ_LEN, input_size=_INPUT_SIZE)
        # Train first
        train_task = LoRATrainTask()
        train_req = LoRATrainRequest(
            execution_id="test-pred-pre",
            base_checkpoint_path=str(base_ckpt),
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
            epochs=1,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        train_resp = train_task.execute(train_req)
        assert train_resp.status == "success"

        predict_task = LoRAPredictTask()
        sequences = [[[float(i)] * _INPUT_SIZE for _ in range(_SEQ_LEN)] for i in range(3)]
        req = LoRAPredictRequest(
            execution_id="test-pred-02",
            base_checkpoint_path=str(base_ckpt),
            lora_checkpoint_path=train_resp.lora_checkpoint_path,
            sequences=sequences,
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            lora_config=_LORA_CFG,
        )
        resp = predict_task.execute(req)
        assert resp.status == "success", resp.error
        assert len(resp.rul_predictions) == 3


class TestOAsst1IngestionTask:
    def test_ingest_skips_gracefully_when_no_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        task = OAsst1IngestionTask()
        req = OAsst1IngestionRequest(
            execution_id="test-ingest-01",
            raw_dir=str(raw_dir),
            parquet_dir=str(tmp_path / "parquet"),
        )
        resp = task.execute(req)
        assert resp.status == "skipped"
        assert resp.rows_ingested == 0

    def test_ingest_filters_by_lang_and_min_len(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {
                "message_id": ["a", "b", "c", "d"],
                "role": ["assistant", "prompter", "assistant", "prompter"],
                "text": ["hello world this is a long text", "hi", "another long assistant message", "short"],
                "lang": ["en", "en", "de", "en"],
            }
        )
        df.to_parquet(raw_dir / "data.parquet")

        task = OAsst1IngestionTask()
        req = OAsst1IngestionRequest(
            execution_id="test-ingest-02",
            raw_dir=str(raw_dir),
            parquet_dir=str(tmp_path / "parquet"),
            lang="en",
            min_text_len=10,
        )
        resp = task.execute(req)
        assert resp.status == "success", resp.error
        # Only English rows with text len >= 10: row 0 (en, assistant, len=31) and row 3 (en, prompter, len=5 < 10 excluded)
        # row 1 (en, prompter, len=2 < 10 excluded)
        # So only row 0 passes
        assert resp.rows_ingested == 1
        assert resp.assistant_turns == 1
        assert resp.human_turns == 0

    def test_ingest_max_rows(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {
                "message_id": [str(i) for i in range(50)],
                "role": ["assistant"] * 50,
                "text": ["this is a long message " * 2] * 50,
                "lang": ["en"] * 50,
            }
        )
        df.to_parquet(raw_dir / "data.parquet")

        task = OAsst1IngestionTask()
        req = OAsst1IngestionRequest(
            execution_id="test-ingest-03",
            raw_dir=str(raw_dir),
            parquet_dir=str(tmp_path / "parquet"),
            lang="en",
            min_text_len=5,
            max_rows=10,
        )
        resp = task.execute(req)
        assert resp.status == "success", resp.error
        assert resp.rows_ingested == 10

"""Tests for attention tasks and CMAPSS ingestion task."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from quantedge_services.api.schemas.foundations.attention_schemas import (
    AttentionTrainRequest,
    AttentionEvalRequest,
    AttentionPredictRequest,
    CMAPSSIngestionRequest,
)
from quantedge_services.services.wfs.forex_attention.tasks.attention_train_task import AttentionTrainTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_eval_task import AttentionEvalTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_predict_task import AttentionPredictTask
from quantedge_services.services.wfs.cmapss.tasks.ingest_task import CMAPSSIngestionTask
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


_SEQ_LEN = 5
_INPUT_SIZE = 4
_N_SEQS = 50


def _make_synthetic_parquet(tmp_path: Path) -> Path:
    """Create a synthetic parquet with flattened sequences."""
    feature_cols = [f"f{i}" for i in range(_SEQ_LEN * _INPUT_SIZE)]
    rng = np.random.default_rng(42)
    data = rng.random((_N_SEQS, _SEQ_LEN * _INPUT_SIZE)).astype(np.float32)
    rul = rng.integers(0, 100, size=_N_SEQS).astype(np.float32)
    engine_ids = np.ones(_N_SEQS, dtype=np.int32)
    df = pd.DataFrame(data, columns=feature_cols)
    df["rul"] = rul
    df["engine_id"] = engine_ids
    p = tmp_path / "test_seq.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_checkpoint(tmp_path: Path) -> Path:
    """Train a tiny model and save a checkpoint."""
    model = ForexTransformer(
        input_size=_INPUT_SIZE,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    ckpt_path = tmp_path / "test_transformer.pt"
    torch.save(
        {
            "input_size": _INPUT_SIZE,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "seq_len": _SEQ_LEN,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


class TestAttentionTrainTask:
    def test_train_creates_checkpoint(self, tmp_path: Path):
        parquet = _make_synthetic_parquet(tmp_path)
        request = AttentionTrainRequest(
            execution_id="test-train",
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            epochs=2,
            learning_rate=0.001,
            batch_size=16,
            patience=10,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        resp = AttentionTrainTask().execute(request)
        assert resp.status == "success"
        assert Path(resp.checkpoint_path).exists()

    def test_train_returns_best_val_loss(self, tmp_path: Path):
        parquet = _make_synthetic_parquet(tmp_path)
        request = AttentionTrainRequest(
            execution_id="test-loss",
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            epochs=2,
            learning_rate=0.001,
            batch_size=16,
            patience=10,
            checkpoint_dir=str(tmp_path / "ckpts2"),
        )
        resp = AttentionTrainTask().execute(request)
        assert resp.best_val_loss > 0

    def test_train_skips_when_no_parquet(self, tmp_path: Path):
        request = AttentionTrainRequest(
            execution_id="test-skip",
            parquet_path=str(tmp_path / "nonexistent.parquet"),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            checkpoint_dir=str(tmp_path),
        )
        resp = AttentionTrainTask().execute(request)
        assert resp.status == "skipped"


class TestAttentionEvalTask:
    def test_eval_skips_gracefully_when_no_parquet(self, tmp_path: Path):
        ckpt = _make_checkpoint(tmp_path)
        request = AttentionEvalRequest(
            execution_id="test-eval-skip",
            checkpoint_path=str(ckpt),
            parquet_path=str(tmp_path / "nonexistent.parquet"),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
        )
        resp = AttentionEvalTask().execute(request)
        assert resp.status == "skipped"

    def test_eval_returns_metrics(self, tmp_path: Path):
        parquet = _make_synthetic_parquet(tmp_path)
        # First train to get a checkpoint
        train_req = AttentionTrainRequest(
            execution_id="eval-train",
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            epochs=1,
            learning_rate=0.001,
            batch_size=16,
            patience=10,
            checkpoint_dir=str(tmp_path / "eval_ckpts"),
        )
        train_resp = AttentionTrainTask().execute(train_req)
        assert train_resp.status == "success"

        eval_req = AttentionEvalRequest(
            execution_id="test-eval",
            checkpoint_path=train_resp.checkpoint_path,
            parquet_path=str(parquet),
            seq_len=_SEQ_LEN,
            input_size=_INPUT_SIZE,
        )
        resp = AttentionEvalTask().execute(eval_req)
        assert resp.status == "success"
        assert resp.rmse >= 0


class TestAttentionPredictTask:
    def test_predict_output_length(self, tmp_path: Path):
        ckpt = _make_checkpoint(tmp_path)
        sequences = [
            [[float(j) for j in range(_INPUT_SIZE)] for _ in range(_SEQ_LEN)]
            for _ in range(5)
        ]
        request = AttentionPredictRequest(
            execution_id="test-predict",
            checkpoint_path=str(ckpt),
            sequences=sequences,
        )
        resp = AttentionPredictTask().execute(request)
        assert resp.status == "success"
        assert len(resp.rul_predictions) == 5

    def test_predict_returns_finite_values(self, tmp_path: Path):
        ckpt = _make_checkpoint(tmp_path)
        sequences = [
            [[float(j) for j in range(_INPUT_SIZE)] for _ in range(_SEQ_LEN)]
            for _ in range(3)
        ]
        request = AttentionPredictRequest(
            execution_id="test-finite",
            checkpoint_path=str(ckpt),
            sequences=sequences,
        )
        resp = AttentionPredictTask().execute(request)
        assert all(np.isfinite(v) for v in resp.rul_predictions)


class TestCMAPSSIngestionTask:
    def test_ingest_skips_gracefully_when_no_file(self, tmp_path: Path):
        request = CMAPSSIngestionRequest(
            execution_id="test-ingest-skip",
            raw_dir=str(tmp_path / "empty_raw"),
            parquet_dir=str(tmp_path / "parquet"),
            subset="FD001",
            seq_len=10,
        )
        resp = CMAPSSIngestionTask().execute(request)
        assert resp.status == "skipped"
        assert resp.sequences_created == 0

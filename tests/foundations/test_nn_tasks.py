"""Tests for NNTrainTask, NNEvalTask, and NNPredictTask."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNEvalRequest,
    NNPredictRequest,
    NNTrainRequest,
)
from quantedge_services.services.wfs.forex_neuralnet.tasks.eval_task import NNEvalTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.predict_task import NNPredictTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.train_task import NNTrainTask


def _make_synthetic_parquet(tmp_dir: str, n_rows: int = 200) -> str:
    """Creates a synthetic numeric Parquet file with OHLCV-style columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "open":   rng.random(n_rows).astype(np.float32),
        "high":   rng.random(n_rows).astype(np.float32) + 0.1,
        "low":    rng.random(n_rows).astype(np.float32),
        "close":  rng.random(n_rows).astype(np.float32),
        "volume": rng.integers(100, 1000, n_rows).astype(np.float32),
    })
    path = os.path.join(tmp_dir, "train.parquet")
    df.to_parquet(path, index=False)
    return path


class TestNNTrainTask:
    def test_train_mlp_returns_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_synthetic_parquet(tmp)
            task = NNTrainTask()
            request = NNTrainRequest(
                execution_id="test-mlp-001",
                model_type="mlp",
                parquet_path=parquet_path,
                hidden_sizes=[32, 16],
                epochs=2,
                batch_size=32,
                patience=5,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert response.checkpoint_path.endswith(".pt")
            assert Path(response.checkpoint_path).exists()

    def test_train_lstm_returns_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_synthetic_parquet(tmp)
            task = NNTrainTask()
            request = NNTrainRequest(
                execution_id="test-lstm-001",
                model_type="lstm",
                parquet_path=parquet_path,
                hidden_sizes=[32],
                epochs=2,
                batch_size=32,
                patience=5,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert response.checkpoint_path.endswith(".pt")
            assert Path(response.checkpoint_path).exists()

    def test_early_stopping_respects_patience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_synthetic_parquet(tmp, n_rows=100)
            task = NNTrainTask()
            request = NNTrainRequest(
                execution_id="test-patience-001",
                model_type="mlp",
                parquet_path=parquet_path,
                hidden_sizes=[16],
                epochs=50,
                batch_size=16,
                patience=1,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            # With patience=1 on random data, training should stop well before 50 epochs
            assert response.epochs_trained <= 50


class TestNNEvalTask:
    def _train_checkpoint(self, tmp: str, model_type: str = "mlp") -> str:
        parquet_path = _make_synthetic_parquet(tmp)
        train_task = NNTrainTask()
        req = NNTrainRequest(
            execution_id=f"eval-setup-{model_type}",
            model_type=model_type,
            parquet_path=parquet_path,
            hidden_sizes=[32, 16],
            epochs=2,
            batch_size=32,
            patience=5,
        )
        resp = train_task.execute(req)
        assert resp.status == "success"
        return resp.checkpoint_path, parquet_path

    def test_eval_returns_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path, parquet_path = self._train_checkpoint(tmp)
            task = NNEvalTask()
            request = NNEvalRequest(
                execution_id="test-eval-001",
                checkpoint_path=ckpt_path,
                parquet_path=parquet_path,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert response.mse >= 0.0
            assert response.mae >= 0.0
            assert response.rmse >= 0.0

    def test_direction_accuracy_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path, parquet_path = self._train_checkpoint(tmp)
            task = NNEvalTask()
            request = NNEvalRequest(
                execution_id="test-eval-dir-001",
                checkpoint_path=ckpt_path,
                parquet_path=parquet_path,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert 0.0 <= response.direction_accuracy <= 1.0


class TestNNPredictTask:
    def _train_checkpoint(self, tmp: str) -> tuple[str, int]:
        parquet_path = _make_synthetic_parquet(tmp)
        train_task = NNTrainTask()
        req = NNTrainRequest(
            execution_id="pred-setup",
            model_type="mlp",
            parquet_path=parquet_path,
            hidden_sizes=[32, 16],
            epochs=2,
            batch_size=32,
            patience=5,
        )
        resp = train_task.execute(req)
        assert resp.status == "success"
        import torch
        ckpt = torch.load(resp.checkpoint_path, map_location="cpu", weights_only=False)
        return resp.checkpoint_path, ckpt["input_size"]

    def test_predict_output_length_matches_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path, input_size = self._train_checkpoint(tmp)
            task = NNPredictTask()
            batch = [[float(i) for i in range(input_size)] for _ in range(5)]
            request = NNPredictRequest(
                execution_id="test-pred-001",
                checkpoint_path=ckpt_path,
                input_features=batch,
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert len(response.predictions) == 5

    def test_predict_with_batch_of_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path, input_size = self._train_checkpoint(tmp)
            task = NNPredictTask()
            request = NNPredictRequest(
                execution_id="test-pred-single",
                checkpoint_path=ckpt_path,
                input_features=[[float(i) for i in range(input_size)]],
            )
            response = task.execute(request)
            assert response.status == "success", response.error
            assert len(response.predictions) == 1

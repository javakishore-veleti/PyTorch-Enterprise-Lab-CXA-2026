import pytest
import torch
import numpy as np
from pathlib import Path
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionExtractRequest, AttentionHeatmapRequest, ArchDecisionRequest,
)
from quantedge_services.services.wfs.attention_viz.tasks.attention_extract_task import AttentionExtractTask
from quantedge_services.services.wfs.attention_viz.tasks.attention_heatmap_task import AttentionHeatmapTask
from quantedge_services.services.wfs.attention_viz.tasks.arch_decision_task import ArchDecisionTask


def _save_mini_checkpoint(tmp_path: Path, seq_len=5, input_size=4, d_model=16, nhead=2, layers=2) -> Path:
    from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer
    model = ForexTransformer(input_size=input_size, d_model=d_model, nhead=nhead,
                              num_encoder_layers=layers, dim_feedforward=32, dropout=0.0)
    ckpt_path = tmp_path / "mini_transformer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size, "d_model": d_model, "nhead": nhead,
        "num_encoder_layers": layers, "dim_feedforward": 32, "dropout": 0.0,
        "val_loss": 0.5, "epoch": 1,
    }, ckpt_path)
    return ckpt_path


def _make_sequences(batch=2, seq_len=5, input_size=4) -> list:
    return np.random.randn(batch, seq_len, input_size).tolist()


class TestAttentionExtractTask:
    def test_extract_returns_correct_num_layers(self, tmp_path):
        ckpt = _save_mini_checkpoint(tmp_path, seq_len=5, input_size=4, d_model=16, nhead=2, layers=2)
        seqs = _make_sequences(batch=2, seq_len=5, input_size=4)
        req = AttentionExtractRequest(
            execution_id="test-extract", checkpoint_path=str(ckpt),
            sequences=seqs, seq_len=5, input_size=4, d_model=16, nhead=2,
            num_encoder_layers=2, dim_feedforward=32, dropout=0.0,
        )
        resp = AttentionExtractTask().execute(req)
        assert resp.status == "success"
        assert resp.num_layers == 2
        assert len(resp.layers) == 2

    def test_extract_weights_shape(self, tmp_path):
        ckpt = _save_mini_checkpoint(tmp_path, seq_len=5, input_size=4, d_model=16, nhead=2, layers=2)
        seqs = _make_sequences(batch=1, seq_len=5, input_size=4)
        req = AttentionExtractRequest(
            execution_id="test-shape", checkpoint_path=str(ckpt),
            sequences=seqs, seq_len=5, input_size=4, d_model=16, nhead=2,
            num_encoder_layers=2, dim_feedforward=32, dropout=0.0,
        )
        resp = AttentionExtractTask().execute(req)
        layer0 = resp.layers[0].weights
        assert len(layer0) == 1
        assert len(layer0[0]) == 2
        assert len(layer0[0][0]) == 5
        assert len(layer0[0][0][0]) == 5

    def test_extract_skips_gracefully_on_missing_checkpoint(self, tmp_path):
        req = AttentionExtractRequest(
            execution_id="test-skip", checkpoint_path=str(tmp_path / "nonexistent.pt"),
            sequences=_make_sequences(), seq_len=5, input_size=4, d_model=16, nhead=2,
            num_encoder_layers=2, dim_feedforward=32, dropout=0.0,
        )
        resp = AttentionExtractTask().execute(req)
        assert resp.status == "skipped"


class TestAttentionHeatmapTask:
    def test_heatmap_creates_png_files(self, tmp_path):
        ckpt = _save_mini_checkpoint(tmp_path, seq_len=5, input_size=4, d_model=16, nhead=2, layers=2)
        seqs = _make_sequences(batch=1, seq_len=5, input_size=4)
        req = AttentionHeatmapRequest(
            execution_id="test-heatmap", checkpoint_path=str(ckpt),
            sequences=seqs, seq_len=5, input_size=4, d_model=16, nhead=2,
            num_encoder_layers=2, dim_feedforward=32,
            output_dir=str(tmp_path / "heatmaps"),
        )
        resp = AttentionHeatmapTask().execute(req)
        assert resp.status == "success"
        assert len(resp.files_created) == 4
        for f in resp.files_created:
            assert Path(f).exists()

    def test_heatmap_skips_on_missing_checkpoint(self, tmp_path):
        req = AttentionHeatmapRequest(
            execution_id="test-skip", checkpoint_path=str(tmp_path / "nope.pt"),
            sequences=_make_sequences(batch=1, seq_len=5, input_size=4),
            seq_len=5, input_size=4, d_model=16, nhead=2, num_encoder_layers=2, dim_feedforward=32,
            output_dir=str(tmp_path / "heatmaps"),
        )
        resp = AttentionHeatmapTask().execute(req)
        assert resp.status == "skipped"


class TestArchDecisionTask:
    def test_regression_recommends_encoder_only(self):
        req = ArchDecisionRequest(execution_id="test-arch",
                                   task_type="time_series_regression", output_type="scalar")
        resp = ArchDecisionTask().execute(req)
        assert resp.status == "success"
        assert resp.recommended_architecture == "encoder_only"

    def test_forecasting_recommends_encoder_decoder(self):
        req = ArchDecisionRequest(execution_id="test-forecast",
                                   task_type="forecasting", output_type="sequence")
        resp = ArchDecisionTask().execute(req)
        assert resp.recommended_architecture == "encoder_decoder"

    def test_response_has_three_options(self):
        req = ArchDecisionRequest(execution_id="test-options",
                                   task_type="time_series_regression", output_type="scalar")
        resp = ArchDecisionTask().execute(req)
        assert len(resp.options) == 3

    def test_rationale_is_non_empty(self):
        req = ArchDecisionRequest(execution_id="test-rationale",
                                   task_type="time_series_regression", output_type="scalar")
        resp = ArchDecisionTask().execute(req)
        assert len(resp.rationale) > 50

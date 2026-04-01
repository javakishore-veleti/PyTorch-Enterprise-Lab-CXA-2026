"""Tests for attention model components and ForexTransformer."""
from __future__ import annotations
import pytest
import torch
from quantedge_services.services.wfs.forex_attention.models.attention_components import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
)
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


class TestScaledDotProductAttention:
    def test_output_shape(self):
        attn = ScaledDotProductAttention(dropout=0.0)
        attn.eval()
        q = torch.randn(2, 4, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        out, weights = attn(q, k, v)
        assert out.shape == (2, 4, 10, 16)
        assert weights.shape == (2, 4, 10, 10)

    def test_attention_weights_sum_to_one(self):
        attn = ScaledDotProductAttention(dropout=0.0)
        attn.eval()
        q = torch.randn(2, 4, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _, weights = attn(q, k, v)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=64, nhead=4, dropout=0.0)
        mha.eval()
        x = torch.randn(4, 20, 64)
        out, weights = mha(x, x, x)
        assert out.shape == (4, 20, 64)

    def test_returns_attention_weights(self):
        mha = MultiHeadAttention(d_model=64, nhead=4, dropout=0.0)
        mha.eval()
        x = torch.randn(4, 20, 64)
        _, weights = mha(x, x, x)
        assert weights.shape == (4, 4, 20, 20)


class TestPositionalEncoding:
    def test_output_shape_preserved(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        pe.eval()
        x = torch.randn(2, 30, 64)
        out = pe(x)
        assert out.shape == (2, 30, 64)

    def test_encoding_is_deterministic(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        pe.eval()
        x = torch.randn(2, 30, 64)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)


class TestForexTransformer:
    def _make_model(self, num_encoder_layers: int = 4) -> ForexTransformer:
        return ForexTransformer(
            input_size=14,
            d_model=64,
            nhead=4,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=256,
            dropout=0.0,
        )

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(8, 30, 14)
        out, _ = model(x)
        assert out.shape == (8, 1)

    def test_returns_attention_weights_per_layer(self):
        model = self._make_model(num_encoder_layers=4)
        model.eval()
        x = torch.randn(8, 30, 14)
        _, attn_list = model(x)
        assert len(attn_list) == 4

    def test_no_nan_in_output(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(8, 30, 14)
        out, _ = model(x)
        assert not out.isnan().any()

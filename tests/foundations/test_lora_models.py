from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import pytest
from quantedge_services.services.wfs.lora_finetuning.models.lora_layer import LoRALinear
from quantedge_services.services.wfs.lora_finetuning.models.lora_transformer import LoRATransformer
from quantedge_services.services.wfs.forex_attention.models.forex_transformer import ForexTransformer


def _make_base_model(input_size=4, d_model=16, nhead=2, layers=2):
    return ForexTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=layers,
        dim_feedforward=32,
        dropout=0.0,
    )


def _make_base_checkpoint(tmp_path, input_size=4, d_model=16, nhead=2, layers=2):
    model = _make_base_model(input_size=input_size, d_model=d_model, nhead=nhead, layers=layers)
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
        },
        p,
    )
    return p, model


class TestLoRALinear:
    def test_output_shape_matches_base(self):
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        x = torch.randn(3, 8)
        out = lora(x)
        assert out.shape == (3, 16)

    def test_output_shape_with_bias(self):
        base = nn.Linear(8, 16, bias=True)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        x = torch.randn(3, 8)
        out = lora(x)
        assert out.shape == (3, 16)

    def test_base_weight_is_frozen(self):
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert lora.base_weight.requires_grad is False

    def test_lora_matrices_are_trainable(self):
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True

    def test_lora_B_initialized_to_zero(self):
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert lora.lora_B.data.abs().sum().item() == 0.0

    def test_merge_weights_produces_linear(self):
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        merged = lora.merge_weights()
        assert isinstance(merged, nn.Linear)
        assert merged.in_features == 8
        assert merged.out_features == 16

    def test_merge_weights_with_bias(self):
        base = nn.Linear(8, 16, bias=True)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        merged = lora.merge_weights()
        assert isinstance(merged, nn.Linear)
        assert merged.bias is not None

    def test_initial_output_matches_base(self):
        """At init, lora_B=0, so LoRA output should equal base output."""
        base = nn.Linear(8, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        x = torch.randn(3, 8)
        with torch.no_grad():
            base_out = base(x)
            lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-5)


class TestLoRATransformer:
    def test_only_lora_params_are_trainable(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        params = lora_model.count_parameters()
        assert params["trainable"] < params["total"]
        assert params["trainable"] > 0

    def test_trainable_ratio_is_small(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        params = lora_model.count_parameters()
        ratio = params["trainable"] / params["total"]
        assert ratio < 0.5

    def test_forward_output_shape(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path, input_size=4, d_model=16, nhead=2)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        x = torch.randn(3, 5, 4)
        out, _ = lora_model(x)
        assert out.shape == (3, 1)

    def test_lora_state_dict_contains_only_lora_keys(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        lsd = lora_model.get_lora_state_dict()
        assert len(lsd) > 0
        assert all("lora_A" in k or "lora_B" in k for k in lsd.keys())

    def test_count_parameters_total_consistent(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        params = lora_model.count_parameters()
        assert params["total"] == params["trainable"] + params["frozen"]

    def test_merge_and_export_returns_forex_transformer(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        merged = lora_model.merge_and_export()
        assert isinstance(merged, ForexTransformer)

    def test_load_lora_state_dict_roundtrip(self, tmp_path):
        _, base = _make_base_checkpoint(tmp_path)
        lora_model = LoRATransformer(base, rank=4, alpha=8.0)
        lsd = lora_model.get_lora_state_dict()
        # Modify lora_A to all-ones to verify loading
        lsd_modified = {k: torch.ones_like(v) for k, v in lsd.items()}
        lora_model.load_lora_state_dict(lsd_modified)
        for k, v in lora_model.get_lora_state_dict().items():
            if "lora_A" in k:
                assert torch.all(v == 1.0)

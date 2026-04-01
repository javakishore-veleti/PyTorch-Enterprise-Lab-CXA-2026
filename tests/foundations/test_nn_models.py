"""Tests for ForexMLP and ForexLSTM model architectures."""
from __future__ import annotations

import torch

from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP


class TestForexMLP:
    def test_forward_output_shape(self) -> None:
        model = ForexMLP(input_size=10, hidden_sizes=[128, 64, 32])
        x = torch.randn(32, 10)
        out = model(x)
        assert out.shape == (32, 1)

    def test_forward_with_batch_norm(self) -> None:
        model = ForexMLP(input_size=10, hidden_sizes=[64, 32])
        x = torch.randn(16, 10)
        out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_different_hidden_configs(self) -> None:
        for hidden_sizes in ([64], [128, 64], [256, 128, 64]):
            model = ForexMLP(input_size=10, hidden_sizes=hidden_sizes)
            x = torch.randn(8, 10)
            out = model(x)
            assert out.shape == (8, 1), f"Failed for hidden_sizes={hidden_sizes}"


class TestForexLSTM:
    def test_forward_output_shape(self) -> None:
        model = ForexLSTM(input_size=10, hidden_size=64, num_layers=2)
        x = torch.randn(32, 20, 10)
        out = model(x)
        assert out.shape == (32, 1)

    def test_stacked_layers(self) -> None:
        model = ForexLSTM(input_size=10, hidden_size=32, num_layers=3)
        x = torch.randn(4, 5, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_gradient_flow(self) -> None:
        model = ForexLSTM(input_size=10, hidden_size=32, num_layers=2)
        x = torch.randn(8, 5, 10)
        target = torch.randn(8, 1)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"None gradient for param: {name}"

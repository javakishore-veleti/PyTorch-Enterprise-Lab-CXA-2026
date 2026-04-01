import os
import torch
import torch.nn as nn
import torch.optim as optim

from quantedge_services.api.schemas.foundations.quantization_schemas import (
    QuantizeQATRequest,
    QuantizeQATResult,
)


class _QuantizableMLP(nn.Module):
    def __init__(self, input_size: int, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(input_size * seq_len, d_model)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(d_model, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)


class QATTask:
    def execute(self, request: QuantizeQATRequest) -> QuantizeQATResult:
        os.makedirs(request.output_dir, exist_ok=True)
        torch.backends.quantized.engine = "qnnpack"

        model = _QuantizableMLP(request.input_size, request.seq_len, request.d_model)

        original_path = os.path.join(request.output_dir, "fp32_qat_model.pt")
        torch.save(model.state_dict(), original_path)
        original_size_mb = os.path.getsize(original_path) / (1024 * 1024)

        model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        torch.quantization.prepare_qat(model, inplace=True)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=request.learning_rate)
        loss_fn = nn.MSELoss()
        final_loss = 0.0

        for _ in range(request.train_steps):
            x = torch.randn(request.batch_size, request.input_size * request.seq_len)
            target = torch.randn(request.batch_size, 1)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        model.eval()
        torch.quantization.convert(model, inplace=True)

        output_path = os.path.join(request.output_dir, "quantized_qat.pt")
        torch.save(model, output_path)
        quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        size_reduction_pct = (
            (original_size_mb - quantized_size_mb) / original_size_mb * 100
            if original_size_mb > 0
            else 0.0
        )

        return QuantizeQATResult(
            output_path=output_path,
            final_loss=final_loss,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            size_reduction_pct=size_reduction_pct,
            status="success",
        )

from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    ProfilerRunRequest, ProfilerRunResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.forex_neuralnet.models.forex_mlp import ForexMLP
from quantedge_services.services.wfs.forex_neuralnet.models.forex_lstm import ForexLSTM


class ProfilerRunTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: ProfilerRunRequest) -> ProfilerRunResponse:
        try:
            trace_dir = Path(request.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            df = pd.read_parquet(request.parquet_path)
            feature_cols = [c for c in df.columns if c != "target"]
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            n_features = X.shape[1]

            # Build model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if request.model_type == "lstm":
                model = ForexLSTM(input_size=n_features)
            else:
                model = ForexMLP(input_size=n_features, hidden_sizes=[128, 64])
            model = model.to(device)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Determine batch size (use min of 256 or dataset size)
            batch_size = min(256, X.shape[0])
            seq_len = 10

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            total_steps = request.wait_steps + request.warmup_steps + request.active_steps

            step_times: list[float] = []

            prof_schedule = schedule(
                wait=request.wait_steps,
                warmup=request.warmup_steps,
                active=request.active_steps,
            )

            with profile(
                activities=activities,
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=True,
                with_stack=request.with_stack,
            ) as prof:
                for step_idx in range(total_steps):
                    t0 = time.perf_counter()
                    # Build batch
                    idx = torch.randint(0, max(X.shape[0] - batch_size, 1), (1,)).item()
                    batch_x = X[idx: idx + batch_size].to(device)

                    if request.model_type == "lstm":
                        # Reshape to [n_seqs, seq_len, n_features]
                        n_seqs = batch_x.shape[0] // seq_len
                        if n_seqs == 0:
                            n_seqs = 1
                            if batch_x.shape[0] >= seq_len:
                                batch_x = batch_x[:seq_len]
                            else:
                                repeat_n = seq_len // batch_x.shape[0] + 1
                                batch_x = batch_x.repeat(repeat_n, 1)[:seq_len]
                        batch_x = batch_x[: n_seqs * seq_len].view(n_seqs, seq_len, n_features)

                    optimizer.zero_grad()
                    out = model(batch_x)
                    target = torch.zeros(out.shape, device=device)
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    t1 = time.perf_counter()
                    step_times.append((t1 - t0) * 1000)  # ms
                    prof.step()

            avg_step_ms = sum(step_times[-request.active_steps:]) / max(request.active_steps, 1)
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

            return ProfilerRunResponse(
                execution_id=request.execution_id,
                trace_path=str(trace_dir),
                steps_profiled=request.active_steps,
                avg_step_ms=avg_step_ms,
                peak_memory_mb=peak_memory_mb,
                status="success",
            )
        except Exception as exc:
            self._logger.error("profiler_run_failed", execution_id=request.execution_id, error=str(exc))
            return ProfilerRunResponse(
                execution_id=request.execution_id,
                trace_path="",
                steps_profiled=0,
                avg_step_ms=0.0,
                peak_memory_mb=0.0,
                status="failed",
                error=str(exc),
            )

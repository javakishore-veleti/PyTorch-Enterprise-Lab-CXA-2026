"""Week 03 — Mixed Precision & OOM Debugging.

Modules:
    mlp_model          — IoT anomaly detection MLP
    lstm_model         — IoT anomaly detection LSTM
    baseline_trainer   — Standard float32 training
    amp_trainer        — Automatic Mixed Precision training
    grad_accumulation  — Gradient accumulation + OOM probing
    benchmark          — Compare all training modes
"""

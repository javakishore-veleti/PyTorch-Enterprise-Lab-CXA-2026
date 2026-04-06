from quantedge_services_handson.week01.tensors import load_forex_data
from quantedge_services_handson.week01.autograd import compute_price_deltas, compute_price_deltas_manual

import torch


def step01_load_tensors() -> dict:
    success = True
    reason_codes = []
    try:
        tensors = load_forex_data("data/forex/processed")

        if tensors.ndim != 1:
            success = False
            reason_codes.append("Tensor should be 1-dimensional.")
        if tensors.dtype != torch.float32:
            success = False
            reason_codes.append("Tensor should be of type float32.")
        if len(tensors) == 0:
            success = False
            reason_codes.append("Tensor should not be empty.")
    except Exception as e:
        success = False
        reason_codes.append(str(e))

    return {"step": "01", "status": str(success), "reason_codes": reason_codes}


def step_02_autograd() -> dict:
    success = True
    reason_codes = []

    try:
        prices = load_forex_data("data/forex/processed")
        grad_auto = compute_price_deltas(prices)
        grad_manual = compute_price_deltas_manual(prices)

        if grad_auto.shape != grad_manual.shape:
            success = False
            reason_codes.append(f"Shape mismatch: auto={grad_auto.shape}, manual={grad_manual.shape}")
        if not torch.allclose(grad_auto, grad_manual, atol=1e-5):
            success = False
            max_diff = (grad_auto - grad_manual).abs().max().item()
            reason_codes.append(f"Gradients do not match (max diff: {max_diff:.2e})")

    except Exception as e:
        success = False
        reason_codes.append(str(e))

    return {"step": "02", "status": str(success), "reason_codes": reason_codes}


def run(**kwargs) -> dict:
    print("  WEEK 01 — Tensors & Autograd")
    results = [step01_load_tensors(), step_02_autograd()]

    for r in results:
        print(f"  Step {r['step']}: status={r['status']}, reasons={r['reason_codes']}")
    return {"week": "01", "results": results}

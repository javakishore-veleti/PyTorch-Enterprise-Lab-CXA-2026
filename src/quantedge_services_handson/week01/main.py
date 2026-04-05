"""Week 01 — Tensors & Autograd runner."""

import torch

DATA_PATH = "data/forex/processed"


def step1_load_tensors() -> dict:
    print("\n  --- Step 1: Load Forex CSV into Tensors ---")
    try:
        from quantedge_services_handson.week01.tensors import load_forex_data

        t = load_forex_data(DATA_PATH)
        print(f"  Shape: {t.shape}, Dtype: {t.dtype}")
        print(f"  Min: {t.min():.4f}, Max: {t.max():.4f}")
        ok = t.ndim == 1 and t.dtype == torch.float32 and len(t) > 0
        print(f"  {'PASSED' if ok else 'FAILED'}")
        return {"step": 1, "status": "passed" if ok else "failed"}
    except ImportError:
        print("  SKIPPED — create week01/tensors.py")
        return {"step": 1, "status": "skipped"}
    except Exception as e:
        print(f"  FAILED — {e}")
        return {"step": 1, "status": "failed", "error": str(e)}


def step2_autograd() -> dict:
    print("\n  --- Step 2: Autograd vs Manual Backward ---")
    try:
        from quantedge_services_handson.week01.autograd import (
            compute_price_deltas,
            manual_backward,
        )
        from quantedge_services_handson.week01.tensors import load_forex_data

        prices = load_forex_data(DATA_PATH)
        grad_auto = compute_price_deltas(prices.clone())
        grad_manual = manual_backward(prices.clone())
        match = torch.allclose(grad_auto, grad_manual, atol=1e-5)
        print(f"  Autograd shape: {grad_auto.shape}")
        print(f"  Manual shape:   {grad_manual.shape}")
        print(f"  Match: {match}")
        print(f"  {'PASSED' if match else 'FAILED — gradients do not match'}")
        return {"step": 2, "status": "passed" if match else "failed"}
    except ImportError as e:
        missing = "tensors.py" if "tensors" in str(e) else "autograd.py"
        print(f"  SKIPPED — create week01/{missing}")
        return {"step": 2, "status": "skipped"}
    except Exception as e:
        print(f"  FAILED — {e}")
        return {"step": 2, "status": "failed", "error": str(e)}


def step3_nan_handling() -> dict:
    print("\n  --- Step 3: NaN Injection & Fix ---")
    try:
        from quantedge_services_handson.week01.nan_handling import inject_nan_and_fix
        from quantedge_services_handson.week01.tensors import load_forex_data

        prices = load_forex_data(DATA_PATH)
        nan_count_before = torch.isnan(prices).sum().item()
        dirty = prices.clone()
        dirty[::7] = float("nan")
        nan_count_injected = torch.isnan(dirty).sum().item()
        fixed = inject_nan_and_fix(prices.clone())
        nan_count_after = torch.isnan(fixed).sum().item()
        print(f"  NaN before inject: {nan_count_before}")
        print(f"  NaN after inject:  {nan_count_injected}")
        print(f"  NaN after fix:     {nan_count_after}")
        ok = nan_count_after == 0
        print(f"  {'PASSED' if ok else 'FAILED — NaN still present'}")
        return {"step": 3, "status": "passed" if ok else "failed"}
    except ImportError as e:
        missing = "tensors.py" if "tensors" in str(e) else "nan_handling.py"
        print(f"  SKIPPED — create week01/{missing}")
        return {"step": 3, "status": "skipped"}
    except Exception as e:
        print(f"  FAILED — {e}")
        return {"step": 3, "status": "failed", "error": str(e)}


def step4_rolling_ops() -> dict:
    print("\n  --- Step 4: Rolling Volatility & Momentum ---")
    try:
        from quantedge_services_handson.week01.rolling_ops import (
            rolling_momentum,
            rolling_volatility,
        )
        from quantedge_services_handson.week01.tensors import load_forex_data

        prices = load_forex_data(DATA_PATH)
        vol = rolling_volatility(prices, window=20)
        mom = rolling_momentum(prices, window=10)
        vol_ok = vol.ndim == 1 and len(vol) > 0 and not torch.isnan(vol).any()
        mom_ok = mom.ndim == 1 and len(mom) > 0 and not torch.isnan(mom).any()
        print(f"  Volatility shape: {vol.shape}, NaN-free: {not torch.isnan(vol).any()}")
        print(f"  Momentum shape:   {mom.shape}, NaN-free: {not torch.isnan(mom).any()}")
        ok = vol_ok and mom_ok
        print(f"  {'PASSED' if ok else 'FAILED'}")
        return {"step": 4, "status": "passed" if ok else "failed"}
    except ImportError as e:
        missing = "tensors.py" if "tensors" in str(e) else "rolling_ops.py"
        print(f"  SKIPPED — create week01/{missing}")
        return {"step": 4, "status": "skipped"}
    except Exception as e:
        print(f"  FAILED — {e}")
        return {"step": 4, "status": "failed", "error": str(e)}


def step5_device_mgmt() -> dict:
    print("\n  --- Step 5: Device Management ---")
    try:
        from quantedge_services_handson.week01.device_mgmt import demonstrate_device_transfer
        from quantedge_services_handson.week01.tensors import load_forex_data

        prices = load_forex_data(DATA_PATH)
        result = demonstrate_device_transfer(prices)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("  PASSED")
        return {"step": 5, "status": "passed"}
    except ImportError as e:
        missing = "tensors.py" if "tensors" in str(e) else "device_mgmt.py"
        print(f"  SKIPPED — create week01/{missing}")
        return {"step": 5, "status": "skipped"}
    except Exception as e:
        print(f"  FAILED — {e}")
        return {"step": 5, "status": "failed", "error": str(e)}


STEPS = [step1_load_tensors, step2_autograd, step3_nan_handling,
         step4_rolling_ops, step5_device_mgmt]


def run(**kwargs) -> dict:
    print("  WEEK 01 — Tensors & Autograd")
    results = []
    for step_fn in STEPS:
        results.append(step_fn())

    print("\n  " + "-" * 40)
    passed = sum(1 for r in results if r["status"] == "passed")
    total = len(results)
    print(f"  Result: {passed}/{total} steps passed")
    return {"week": "01", "results": results}

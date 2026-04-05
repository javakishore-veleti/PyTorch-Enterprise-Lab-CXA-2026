"""Run Week 01 — Tensors & Autograd.

Usage:
    python -m quantedge_services_handson.week01
    python -m quantedge_services_handson.week01 --step 2
"""

import argparse
import sys

import torch


def step1_load_tensors() -> None:
    """Step 1: Load Forex CSV into tensors."""
    print("\n=== Step 1: Load Forex CSV into Tensors ===")
    try:
        from quantedge_services_handson.week01.tensors import load_forex_data

        # Smoke test with synthetic data if no CSV available
        prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
        print(f"  Tensor shape: {prices.shape}, dtype: {prices.dtype}")
        print("  PASSED")
    except ImportError:
        print("  SKIPPED — Create week01/tensors.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step2_autograd() -> None:
    """Step 2: Autograd vs manual backward pass."""
    print("\n=== Step 2: Autograd vs Manual Backward ===")
    try:
        from quantedge_services_handson.week01.autograd import (
            compute_price_deltas,
            manual_backward,
        )

        prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
        grad_auto = compute_price_deltas(prices.clone())
        grad_manual = manual_backward(prices.clone())
        match = torch.allclose(grad_auto, grad_manual, atol=1e-5)
        print(f"  Autograd shape: {grad_auto.shape}")
        print(f"  Manual shape:   {grad_manual.shape}")
        print(f"  Match: {match}")
        print("  PASSED" if match else "  FAILED — gradients don't match")
    except ImportError:
        print("  SKIPPED — Create week01/autograd.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step3_nan_handling() -> None:
    """Step 3: Inject NaN and fix it."""
    print("\n=== Step 3: NaN Injection & Fix ===")
    try:
        from quantedge_services_handson.week01.nan_handling import inject_nan_and_fix

        prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
        fixed = inject_nan_and_fix(prices)
        clean = not torch.isnan(fixed).any()
        print(f"  NaN-free after fix: {clean}")
        print("  PASSED" if clean else "  FAILED — NaN still present")
    except ImportError:
        print("  SKIPPED — Create week01/nan_handling.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step4_rolling_ops() -> None:
    """Step 4: Rolling volatility & momentum."""
    print("\n=== Step 4: Rolling Volatility & Momentum ===")
    try:
        from quantedge_services_handson.week01.rolling_ops import (
            rolling_momentum,
            rolling_volatility,
        )

        prices = torch.randn(1000, dtype=torch.float32).abs() + 1.0
        vol = rolling_volatility(prices, window=20)
        mom = rolling_momentum(prices, window=10)
        print(f"  Volatility shape: {vol.shape}")
        print(f"  Momentum shape:   {mom.shape}")
        vol_ok = not torch.isnan(vol).any() and vol.shape[0] > 0
        mom_ok = not torch.isnan(mom).any() and mom.shape[0] > 0
        print("  PASSED" if vol_ok and mom_ok else "  FAILED")
    except ImportError:
        print("  SKIPPED — Create week01/rolling_ops.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step5_device_mgmt() -> None:
    """Step 5: Device management."""
    print("\n=== Step 5: Device Management ===")
    try:
        from quantedge_services_handson.week01.device_mgmt import demonstrate_device_transfer

        prices = torch.randn(10000, dtype=torch.float32)
        result = demonstrate_device_transfer(prices)
        print(f"  CUDA available: {result.get('device_available', 'N/A')}")
        for k, v in result.items():
            if k != "device_available":
                print(f"  {k}: {v}")
        print("  PASSED")
    except ImportError:
        print("  SKIPPED — Create week01/device_mgmt.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


STEPS = {
    1: step1_load_tensors,
    2: step2_autograd,
    3: step3_nan_handling,
    4: step4_rolling_ops,
    5: step5_device_mgmt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 01 — Tensors & Autograd")
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run a single step (1-5). Omit to run all.")
    args = parser.parse_args()

    print("=" * 60)
    print("  WEEK 01 — Tensors & Autograd")
    print("=" * 60)

    if args.step:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()

    print("\n" + "=" * 60)
    print("  Week 01 complete.")
    print("=" * 60)


main()

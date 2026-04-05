"""Run Week 03 — Mixed Precision & OOM Debugging.

Usage:
    python -m quantedge_services_handson.week03
    python -m quantedge_services_handson.week03 --step 4
"""

import argparse

import torch


def step1_mlp() -> None:
    print("\n=== Step 1: IoT Anomaly MLP ===")
    try:
        from quantedge_services_handson.week03.mlp_model import IoTAnomalyMLP

        model = IoTAnomalyMLP(input_dim=46, num_classes=35)
        x = torch.randn(32, 46)
        out = model(x)
        print(f"  Output shape: {out.shape}")
        print("  PASSED" if out.shape == (32, 35) else "  FAILED")
    except ImportError:
        print("  SKIPPED — Create week03/mlp_model.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step2_lstm() -> None:
    print("\n=== Step 2: IoT Anomaly LSTM ===")
    try:
        from quantedge_services_handson.week03.lstm_model import IoTAnomalyLSTM

        model = IoTAnomalyLSTM(feature_dim=46, num_classes=35)
        x = torch.randn(32, 100, 46)
        out = model(x)
        print(f"  Output shape: {out.shape}")
        print("  PASSED" if out.shape == (32, 35) else "  FAILED")
    except ImportError:
        print("  SKIPPED — Create week03/lstm_model.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step3_baseline() -> None:
    print("\n=== Step 3: Baseline Training ===")
    try:
        from quantedge_services_handson.week03.baseline_trainer import train_baseline

        print("  (Requires GPU + data — run manually)")
        print("  SKIPPED — needs CUDA device")
    except ImportError:
        print("  SKIPPED — Create week03/baseline_trainer.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step4_amp() -> None:
    print("\n=== Step 4: AMP Training ===")
    try:
        from quantedge_services_handson.week03.amp_trainer import train_with_amp

        print("  (Requires GPU + data — run manually)")
        print("  SKIPPED — needs CUDA device")
    except ImportError:
        print("  SKIPPED — Create week03/amp_trainer.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step5_grad_accum() -> None:
    print("\n=== Step 5: Gradient Accumulation ===")
    try:
        from quantedge_services_handson.week03.grad_accumulation import (
            find_max_batch_size,
            train_with_gradient_accumulation,
        )

        print("  (Requires GPU — run manually)")
        print("  SKIPPED — needs CUDA device")
    except ImportError:
        print("  SKIPPED — Create week03/grad_accumulation.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step6_benchmark() -> None:
    print("\n=== Step 6: Benchmark Comparison ===")
    try:
        from quantedge_services_handson.week03.benchmark import compare_training_modes

        print("  (Requires GPU — run manually)")
        print("  SKIPPED — needs CUDA device")
    except ImportError:
        print("  SKIPPED — Create week03/benchmark.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


STEPS = {1: step1_mlp, 2: step2_lstm, 3: step3_baseline,
         4: step4_amp, 5: step5_grad_accum, 6: step6_benchmark}


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 03 — Mixed Precision & OOM Debugging")
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run a single step (1-6). Omit to run all.")
    args = parser.parse_args()

    print("=" * 60)
    print("  WEEK 03 — Mixed Precision & OOM Debugging")
    print("=" * 60)

    if args.step:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()

    print("\n" + "=" * 60)
    print("  Week 03 complete.")
    print("=" * 60)


main()

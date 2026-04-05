"""Run Week 04 — Profiling & Bottleneck Identification.

Usage:
    python -m quantedge_services_handson.week04
    python -m quantedge_services_handson.week04 --step 2
"""

import argparse


def step1_profiler() -> None:
    print("\n=== Step 1: PyTorch Profiler ===")
    try:
        from quantedge_services_handson.week04.profiler import profile_training

        print("  (Requires GPU + model — run manually)")
        print("  SKIPPED — needs CUDA device and trained model")
    except ImportError:
        print("  SKIPPED — Create week04/profiler.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step2_memory() -> None:
    print("\n=== Step 2: Memory Analysis ===")
    try:
        from quantedge_services_handson.week04.memory_analysis import analyze_memory

        print("  (Requires GPU — run manually)")
        print("  SKIPPED — needs CUDA device")
    except ImportError:
        print("  SKIPPED — Create week04/memory_analysis.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step3_workers() -> None:
    print("\n=== Step 3: Worker Tuning ===")
    try:
        from quantedge_services_handson.week04.worker_tuning import tune_num_workers

        print("  (Requires dataset — run manually)")
        print("  SKIPPED — provide a dataset")
    except ImportError:
        print("  SKIPPED — Create week04/worker_tuning.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step4_diagnosis() -> None:
    print("\n=== Step 4: Bottleneck Diagnosis ===")
    try:
        from quantedge_services_handson.week04.diagnosis import diagnose_bottleneck

        print("  (Requires profiler outputs — run steps 1-3 first)")
        print("  SKIPPED — complete prior steps")
    except ImportError:
        print("  SKIPPED — Create week04/diagnosis.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


STEPS = {1: step1_profiler, 2: step2_memory, 3: step3_workers, 4: step4_diagnosis}


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 04 — Profiling & Bottleneck ID")
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run a single step (1-4). Omit to run all.")
    args = parser.parse_args()

    print("=" * 60)
    print("  WEEK 04 — Profiling & Bottleneck Identification")
    print("=" * 60)

    if args.step:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()

    print("\n" + "=" * 60)
    print("  Week 04 complete.")
    print("=" * 60)


main()

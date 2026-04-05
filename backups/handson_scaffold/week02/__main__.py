"""Run Week 02 — Training Loop & DataLoader.

Usage:
    python -m quantedge_services_handson.week02
    python -m quantedge_services_handson.week02 --step 3
"""

import argparse

import torch


def step1_dataset() -> None:
    print("\n=== Step 1: Custom Map-Style Dataset ===")
    try:
        from quantedge_services_handson.week02.dataset import ComplaintDataset

        texts = ["bad service"] * 100 + ["good bank"] * 100
        labels = ["complaint"] * 100 + ["praise"] * 100
        ds = ComplaintDataset(texts, labels, max_len=32)
        x, y = ds[0]
        print(f"  Dataset size: {len(ds)}")
        print(f"  Sample shapes: x={x.shape}, y={y.shape}")
        print("  PASSED")
    except ImportError:
        print("  SKIPPED — Create week02/dataset.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step2_iterable_dataset() -> None:
    print("\n=== Step 2: Iterable-Style Dataset ===")
    try:
        from quantedge_services_handson.week02.iterable_dataset import ComplaintIterableDataset

        print("  (Requires a CSV file — test with your data)")
        print("  SKIPPED — provide csv_path to test")
    except ImportError:
        print("  SKIPPED — Create week02/iterable_dataset.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step3_dataloader() -> None:
    print("\n=== Step 3: DataLoader with pin_memory ===")
    try:
        from quantedge_services_handson.week02.dataloader import create_dataloaders
        from quantedge_services_handson.week02.dataset import ComplaintDataset

        texts = ["bad service"] * 200 + ["good bank"] * 200
        labels = ["complaint"] * 200 + ["praise"] * 200
        ds = ComplaintDataset(texts, labels, max_len=32)
        train_ds, val_ds = torch.utils.data.random_split(ds, [320, 80])
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
        batch_x, batch_y = next(iter(train_loader))
        print(f"  Batch shape: x={batch_x.shape}, y={batch_y.shape}")
        print("  PASSED")
    except ImportError:
        print("  SKIPPED — Create week02/dataloader.py (and dataset.py) first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step4_classifier() -> None:
    print("\n=== Step 4: MLP Classifier ===")
    try:
        from quantedge_services_handson.week02.classifier import ComplaintClassifier

        model = ComplaintClassifier(vocab_size=256, num_classes=2)
        x = torch.randint(0, 256, (32, 128))
        out = model(x)
        print(f"  Output shape: {out.shape}")
        ok = out.shape == (32, 2)
        print("  PASSED" if ok else f"  FAILED — expected (32, 2), got {out.shape}")
    except ImportError:
        print("  SKIPPED — Create week02/classifier.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step5_training() -> None:
    print("\n=== Step 5: Training Loop ===")
    try:
        from quantedge_services_handson.week02.classifier import ComplaintClassifier
        from quantedge_services_handson.week02.dataloader import create_dataloaders
        from quantedge_services_handson.week02.dataset import ComplaintDataset
        from quantedge_services_handson.week02.training_loop import set_all_seeds, train

        set_all_seeds(42)
        texts = ["bad service awful"] * 300 + ["good bank great"] * 300
        labels = ["complaint"] * 300 + ["praise"] * 300
        ds = ComplaintDataset(texts, labels, max_len=32)
        train_ds, val_ds = torch.utils.data.random_split(ds, [480, 120])
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
        model = ComplaintClassifier(vocab_size=256, num_classes=2)
        result = train(model, train_loader, val_loader, epochs=3)
        print(f"  Train losses: {[f'{l:.4f}' for l in result['train_losses']]}")
        print(f"  Val accuracies: {[f'{a:.4f}' for a in result['val_accuracies']]}")
        print("  PASSED")
    except ImportError:
        print("  SKIPPED — Complete steps 1-4 first")
    except Exception as e:
        print(f"  FAILED — {e}")


def step6_checkpointing() -> None:
    print("\n=== Step 6: Checkpoint Resume ===")
    try:
        from quantedge_services_handson.week02.checkpointing import resume_from_checkpoint

        print("  (Requires a saved checkpoint — run step 5 first)")
        print("  SKIPPED — train first, then test resume")
    except ImportError:
        print("  SKIPPED — Create week02/checkpointing.py first")
    except Exception as e:
        print(f"  FAILED — {e}")


STEPS = {
    1: step1_dataset,
    2: step2_iterable_dataset,
    3: step3_dataloader,
    4: step4_classifier,
    5: step5_training,
    6: step6_checkpointing,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 02 — Training Loop & DataLoader")
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run a single step (1-6). Omit to run all.")
    args = parser.parse_args()

    print("=" * 60)
    print("  WEEK 02 — Training Loop & DataLoader")
    print("=" * 60)

    if args.step:
        STEPS[args.step]()
    else:
        for step_fn in STEPS.values():
            step_fn()

    print("\n" + "=" * 60)
    print("  Week 02 complete.")
    print("=" * 60)


main()

"""Week 00 — Dataset Setup & Validation."""

from pathlib import Path

DATA_ROOT = Path("data")

DATASETS = {
    "forex":         {"dir": DATA_ROOT / "forex/processed",  "glob": "*.parquet", "min_files": 1},
    "cfpb":          {"dir": DATA_ROOT / "cfpb",             "glob": "*.parquet", "min_files": 1},
    "cic_iot":       {"dir": DATA_ROOT / "cic_iot",          "glob": "*.parquet", "min_files": 1},
    "cmapss":        {"dir": DATA_ROOT / "cmapss",           "glob": "*.txt",     "min_files": 1},
    "oasst1":        {"dir": DATA_ROOT / "oasst1",           "glob": "*.parquet", "min_files": 1},
    "stackoverflow": {"dir": DATA_ROOT / "stackoverflow",    "glob": "*.parquet", "min_files": 1},
}


def is_present(ds: dict) -> bool:
    d = ds["dir"]
    return d.exists() and len(list(d.glob(ds["glob"]))) >= ds["min_files"]


def _setup_forex(ds: dict) -> dict:
    import numpy as np
    import pandas as pd

    ds["dir"].mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for i in range(1, 10):
        n = 200
        dates = pd.date_range("2023-01-03 09:00", periods=n, freq="1min")
        base = 1.08 + np.random.normal(0, 0.005, n).cumsum()
        df = pd.DataFrame({
            "open": base + np.random.normal(0, 0.001, n),
            "high": base + abs(np.random.normal(0, 0.002, n)),
            "low": base - abs(np.random.normal(0, 0.002, n)),
            "close": base + np.random.normal(0, 0.001, n),
            "volume": np.random.randint(100, 5000, n).astype(float),
            "spread": abs(np.random.normal(0.0002, 0.0001, n)),
        }, index=dates)
        df.to_parquet(ds["dir"] / f"eurusd_processed_pp-{i:03d}.parquet")
        total_rows += n
    return {"name": "forex", "status": "created", "files": 9, "rows": total_rows}


def _setup_cfpb(ds: dict) -> dict:
    import numpy as np
    import pandas as pd

    ds["dir"].mkdir(parents=True, exist_ok=True)
    products = ["Credit card", "Mortgage", "Student loan", "Checking/savings", "Debt collection"]
    n = 1000
    df = pd.DataFrame({
        "complaint_narrative": [f"Complaint about {products[i % 5]} issue #{i}" for i in range(n)],
        "product": [products[i % 5] for i in range(n)],
        "company": np.random.choice(["BankA", "BankB", "BankC", "LenderX"], n),
        "state": np.random.choice(["CA", "TX", "NY", "FL", "IL"], n),
        "date_received": pd.date_range("2020-01-01", periods=n, freq="D"),
    })
    df.to_parquet(ds["dir"] / "complaints.parquet")
    return {"name": "cfpb", "status": "created", "files": 1, "rows": n}


def _setup_cic_iot(ds: dict) -> dict:
    import numpy as np
    import pandas as pd

    ds["dir"].mkdir(parents=True, exist_ok=True)
    n = 5000
    feature_names = [f"feature_{i:02d}" for i in range(46)]
    data = np.random.randn(n, 46).astype(np.float32)
    labels = np.random.choice(
        ["Normal"] + [f"Attack_{i}" for i in range(1, 35)], n
    )
    df = pd.DataFrame(data, columns=feature_names)
    df["label"] = labels
    df.to_parquet(ds["dir"] / "iot_traffic.parquet")
    return {"name": "cic_iot", "status": "created", "files": 1, "rows": n}


def _setup_cmapss(ds: dict) -> dict:
    import numpy as np

    ds["dir"].mkdir(parents=True, exist_ok=True)
    rows = []
    for unit_id in range(1, 101):
        max_cycle = np.random.randint(50, 300)
        for cycle in range(1, max_cycle + 1):
            op_settings = np.random.uniform(0, 0.01, 3)
            sensors = np.random.uniform(100, 600, 21) + np.random.randn(21) * 5
            row = [unit_id, cycle] + op_settings.tolist() + sensors.tolist()
            rows.append(row)
    with open(ds["dir"] / "train_FD001.txt", "w") as f:
        for row in rows:
            f.write(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
    return {"name": "cmapss", "status": "created", "files": 1, "rows": len(rows)}


def _setup_oasst1(ds: dict) -> dict:
    import pandas as pd

    ds["dir"].mkdir(parents=True, exist_ok=True)
    n = 500
    topics = ["Write a Python function to", "Explain how to", "What is the difference between",
              "How do I implement", "Debug this code:"]
    df = pd.DataFrame({
        "instruction": [f"{topics[i % 5]} topic_{i}" for i in range(n)],
        "response": [f"Here is the answer for topic_{i}. Step 1... Step 2..." for i in range(n)],
    })
    df.to_parquet(ds["dir"] / "oasst1_en.parquet")
    return {"name": "oasst1", "status": "created", "files": 1, "rows": n}


def _setup_stackoverflow(ds: dict) -> dict:
    import numpy as np
    import pandas as pd

    ds["dir"].mkdir(parents=True, exist_ok=True)
    n = 500
    tags_pool = ["java", "spring-boot", "elasticsearch", "hibernate", "maven"]
    df = pd.DataFrame({
        "question": [f"How to configure {tags_pool[i % 5]} for use case #{i}?" for i in range(n)],
        "answer": [f"You can configure {tags_pool[i % 5]} by setting..." for i in range(n)],
        "tags": [tags_pool[i % 5] for i in range(n)],
        "score": np.random.randint(5, 100, n),
    })
    df.to_parquet(ds["dir"] / "so_java_es.parquet")
    return {"name": "stackoverflow", "status": "created", "files": 1, "rows": n}


_SETUP_FNS = {
    "forex": _setup_forex,
    "cfpb": _setup_cfpb,
    "cic_iot": _setup_cic_iot,
    "cmapss": _setup_cmapss,
    "oasst1": _setup_oasst1,
    "stackoverflow": _setup_stackoverflow,
}


def setup_dataset(name: str, ds: dict, force: bool = False) -> dict:
    if is_present(ds) and not force:
        files = list(ds["dir"].glob(ds["glob"]))
        print(f"  {name:20s} SKIP ({len(files)} files)")
        return {"name": name, "status": "skipped", "files": len(files)}

    fn = _SETUP_FNS.get(name)
    if fn is None:
        print(f"  {name:20s} NO SETUP FUNCTION")
        return {"name": name, "status": "missing", "files": 0}

    result = fn(ds)
    print(f"  {name:20s} CREATED ({result.get('rows', 0)} rows)")
    return result


def run(force: bool = False) -> dict:
    print("  WEEK 00 — Dataset Setup")
    results = []
    for name, ds in DATASETS.items():
        results.append(setup_dataset(name, ds, force=force))

    print(f"\n  {'Dataset':20s} {'Status':10s} {'Files':>5s}")
    print("  " + "-" * 40)
    for r in results:
        print(f"  {r['name']:20s} {r['status']:10s} {r['files']:5d}")
    return {"week": "00", "results": results}

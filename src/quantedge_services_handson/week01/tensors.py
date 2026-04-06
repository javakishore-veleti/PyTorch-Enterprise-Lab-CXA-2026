import torch
import pandas as pd
from pathlib import Path


def load_forex_data(data_path: str) -> torch.Tensor:
    path = Path(data_path)
    if path.is_dir():
        parquest_files = [pd.read_parquet(file) for file in path.glob("*.parquet")]
        df = pd.concat(parquest_files, ignore_index=True)
    else:
        df = pd.read_parquet(path)
    # Convert the DataFrame to a PyTorch tensor
    tensor = torch.tensor(df["close"].values, dtype=torch.float32)
    return tensor



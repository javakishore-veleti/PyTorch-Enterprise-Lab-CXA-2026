"""ReproducibilityManager — deterministic seeds across all RNGs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


class ReproducibilityManager:
    """Sets deterministic seeds for Python, NumPy, and PyTorch.

    Inject into any training class that requires reproducible runs.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    def apply(self) -> None:
        """Apply seeds to all RNGs. Call once before training begins."""
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self._seed)

"""DeviceManager — GPU/CPU selection and memory reporting."""

from __future__ import annotations

import torch


class DeviceManager:
    """Manages device selection and GPU memory reporting.

    Instantiate once and inject into any class that needs device access.
    """

    def __init__(self, prefer_gpu: bool = True) -> None:
        self._prefer_gpu = prefer_gpu
        self._device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        if self._prefer_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            total_gb = props.total_memory / (1024**3)
            print(
                f"[DeviceManager] cuda — {props.name} "
                f"({total_gb:.1f} GB VRAM, compute {props.major}.{props.minor})"
            )
            return device
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def memory_summary(self) -> str:
        if self._device.type == "cuda":
            return torch.cuda.memory_summary(device=self._device, abbreviated=True)
        return "Device is CPU — no GPU memory to report."

    def reset_peak_stats(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

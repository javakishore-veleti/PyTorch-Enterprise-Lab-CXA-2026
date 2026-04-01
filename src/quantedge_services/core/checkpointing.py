"""CheckpointManager — save and load model checkpoints (local disk)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from quantedge_services.core.logging import StructuredLogger


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    All state is passed as a dict. The manager owns path resolution
    and atomic write behaviour.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._logger = StructuredLogger(name=__name__)

    def save(
        self,
        state: dict,
        filename: str,
        metadata: dict | None = None,
    ) -> Path:
        """Save a checkpoint dict to disk.

        Args:
            state: Must contain model_state_dict, optimizer_state_dict, epoch.
            filename: File name (e.g. 'epoch_001.pt').
            metadata: JSON-serializable metadata written alongside the .pt file.

        Returns:
            Resolved path of the saved checkpoint.
        """
        path = self._base_dir / filename
        torch.save(state, path)
        if metadata:
            meta_path = path.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(metadata, indent=2))
        self._logger.info("checkpoint_saved", path=str(path), epoch=state.get("epoch"))
        return path

    def load(
        self,
        filename: str,
        device: torch.device | None = None,
    ) -> dict:
        """Load a checkpoint from disk.

        Args:
            filename: File name relative to base_dir.
            device: Map tensors to this device on load.

        Returns:
            The checkpoint dict.
        """
        path = self._base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint: dict = torch.load(path, map_location=device, weights_only=True)
        self._logger.info("checkpoint_loaded", path=str(path), epoch=checkpoint.get("epoch"))
        return checkpoint

    def latest(self) -> Path | None:
        """Return path to the most recently modified .pt file, or None."""
        pts = sorted(self._base_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return pts[-1] if pts else None

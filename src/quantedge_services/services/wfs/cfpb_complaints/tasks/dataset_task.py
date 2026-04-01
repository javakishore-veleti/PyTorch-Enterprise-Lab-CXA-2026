"""CFPBDatasetTask — builds CFPBComplaintsDataset and DataLoaders."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest,
    CFPBDatasetResponse,
)
from quantedge_services.core.logging import StructuredLogger

PRODUCT_COL = "Product"


class CFPBComplaintsDataset(TorchDataset):  # type: ignore[type-arg]
    """PyTorch Dataset wrapping tokenized CFPB records.
    Each item: {'input_ids', 'attention_mask', 'label'}.
    """

    def __init__(self, hf_dataset: Dataset, label_map: dict[str, int]) -> None:
        self._dataset = hf_dataset
        self._label_map = label_map

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._dataset[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "label": torch.tensor(
                self._label_map.get(row[PRODUCT_COL], 0), dtype=torch.long
            ),
        }


class CFPBDatasetTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: CFPBDatasetRequest,
        dataset: Dataset,
        label_map: dict[str, int],
    ) -> tuple[CFPBDatasetResponse, DataLoader[Any], DataLoader[Any]]:
        try:
            full_ds = CFPBComplaintsDataset(hf_dataset=dataset, label_map=label_map)
            val_size = int(len(full_ds) * request.val_split)
            train_size = len(full_ds) - val_size
            train_ds, val_ds = random_split(full_ds, [train_size, val_size])

            train_loader = DataLoader(
                train_ds, batch_size=request.batch_size, shuffle=True,
                num_workers=request.num_workers, pin_memory=request.pin_memory,
                drop_last=False, persistent_workers=request.num_workers > 0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=request.batch_size, shuffle=False,
                num_workers=request.num_workers, pin_memory=request.pin_memory,
                drop_last=False, persistent_workers=request.num_workers > 0,
            )
            self._logger.info(
                "dataloaders_built", execution_id=request.execution_id,
                train=train_size, val=val_size,
            )
            return CFPBDatasetResponse(
                execution_id=request.execution_id,
                train_samples=train_size, val_samples=val_size,
                train_batches=len(train_loader), val_batches=len(val_loader),
                status="success",
            ), train_loader, val_loader

        except Exception as exc:
            self._logger.error("dataset_task_failed", error=str(exc))
            resp = CFPBDatasetResponse(
                execution_id=request.execution_id, train_samples=0, val_samples=0,
                train_batches=0, val_batches=0, status="failed", error=str(exc),
            )
            return resp, DataLoader([]), DataLoader([])  # type: ignore[arg-type]

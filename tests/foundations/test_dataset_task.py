"""Tests for CFPBComplaintsDataset and CFPBDatasetTask."""

from __future__ import annotations

import torch
from datasets import Dataset

from quantedge_services.api.schemas.foundations.cfpb_schemas import CFPBDatasetRequest
from quantedge_services.services.wfs.cfpb_complaints.tasks.dataset_task import (
    CFPBComplaintsDataset,
    CFPBDatasetTask,
)

_LABEL_MAP = {"Mortgage": 0, "Credit card": 1, "Debt collection": 2}


def _make_hf_dataset(n: int = 50) -> Dataset:
    products = list(_LABEL_MAP.keys())
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3] + [0] * 61] * n,
            "attention_mask": [[1, 1, 1] + [0] * 61] * n,
            "Product": [products[i % len(products)] for i in range(n)],
        }
    )


class TestCFPBComplaintsDataset:
    def test_len(self) -> None:
        ds = CFPBComplaintsDataset(hf_dataset=_make_hf_dataset(20), label_map=_LABEL_MAP)
        assert len(ds) == 20

    def test_item_keys(self) -> None:
        ds = CFPBComplaintsDataset(hf_dataset=_make_hf_dataset(10), label_map=_LABEL_MAP)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "label"}

    def test_tensor_types(self) -> None:
        ds = CFPBComplaintsDataset(hf_dataset=_make_hf_dataset(10), label_map=_LABEL_MAP)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["label"].dtype == torch.long


class TestCFPBDatasetTask:
    def test_build_dataloaders_success(self) -> None:
        task = CFPBDatasetTask()
        request = CFPBDatasetRequest(
            execution_id="test-ds-001",
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            val_split=0.2,
        )
        resp, train_loader, val_loader = task.execute(
            request, _make_hf_dataset(50), _LABEL_MAP
        )
        assert resp.status == "success"
        assert resp.train_samples == 40
        assert resp.val_samples == 10

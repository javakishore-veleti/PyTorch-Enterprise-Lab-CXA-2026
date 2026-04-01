"""CFPBComplaintsService — orchestrates all CFPB workflow tasks."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest,
    CFPBDatasetResponse,
    CFPBIngestionRequest,
    CFPBIngestionResponse,
    CFPBPreprocessRequest,
    CFPBPreprocessResponse,
    CFPBTrainRequest,
    CFPBTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.cfpb_complaints.tasks.dataset_task import CFPBDatasetTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.ingest_task import CFPBIngestionTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.preprocess_task import CFPBPreprocessTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.training_task import CFPBTrainingTask

PRODUCT_COL = "Product"


class CFPBComplaintsService:
    """Domain service for CFPB Consumer Financial Complaints workflows."""

    def __init__(
        self,
        ingest_task: CFPBIngestionTask,
        preprocess_task: CFPBPreprocessTask,
        dataset_task: CFPBDatasetTask,
        training_task: CFPBTrainingTask,
    ) -> None:
        self._ingest_task = ingest_task
        self._preprocess_task = preprocess_task
        self._dataset_task = dataset_task
        self._training_task = training_task
        self._logger = StructuredLogger(name=__name__)
        self._session_dataset: Dataset | None = None
        self._label_map: dict[str, int] = {}
        self._train_loader: DataLoader[Any] | None = None
        self._val_loader: DataLoader[Any] | None = None

    def ingest(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        from datasets import load_dataset
        from pathlib import Path
        resp = self._ingest_task.execute(request)
        if resp.status == "success":
            self._session_dataset = load_dataset(
                "cfpb/consumer-finance-complaints",
                split=request.split,
                cache_dir=str(Path(request.cache_dir)),
                trust_remote_code=True,
            )  # type: ignore[assignment]
        return resp

    def preprocess(self, request: CFPBPreprocessRequest) -> CFPBPreprocessResponse:
        if self._session_dataset is None:
            return CFPBPreprocessResponse(
                execution_id=request.execution_id, rows_after_filter=0,
                n_classes=0, class_weights={}, status="failed",
                error="No data. Call ingest first.",
            )
        resp, processed_ds = self._preprocess_task.execute(request, self._session_dataset)
        if resp.status == "success":
            self._session_dataset = processed_ds
            unique_labels = list(resp.class_weights.keys())
            self._label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
        return resp

    def build_dataloaders(self, request: CFPBDatasetRequest) -> CFPBDatasetResponse:
        if self._session_dataset is None:
            return CFPBDatasetResponse(
                execution_id=request.execution_id, train_samples=0, val_samples=0,
                train_batches=0, val_batches=0, status="failed",
                error="No data. Call preprocess first.",
            )
        resp, self._train_loader, self._val_loader = self._dataset_task.execute(
            request, self._session_dataset, self._label_map
        )
        return resp

    def train(self, request: CFPBTrainRequest, model: nn.Module) -> CFPBTrainResponse:
        if self._train_loader is None or self._val_loader is None:
            return CFPBTrainResponse(
                execution_id=request.execution_id, epochs_completed=0,
                final_train_loss=0.0, final_val_loss=0.0, final_val_accuracy=0.0,
                checkpoint_path="", status="failed",
                error="No DataLoaders. Call build_dataloaders first.",
            )
        return self._training_task.execute(request, model, self._train_loader, self._val_loader)

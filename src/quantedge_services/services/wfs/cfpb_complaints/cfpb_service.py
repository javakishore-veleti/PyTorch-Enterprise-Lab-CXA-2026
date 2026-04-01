"""CFPBComplaintsService — orchestrates CFPB workflow tasks (fully async)."""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest, CFPBDatasetResponse,
    CFPBDownloadRequest, CFPBDownloadResponse,
    CFPBIngestionRequest, CFPBIngestionResponse,
    CFPBPreprocessRequest, CFPBPreprocessResponse,
    CFPBTrainRequest, CFPBTrainResponse,
)
from quantedge_services.core.logging import StructuredLogger
from quantedge_services.services.wfs.cfpb_complaints.tasks.dataset_task import CFPBDatasetTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.download_task import CFPBDataDownloadTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.ingest_task import CFPBIngestionTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.preprocess_task import CFPBPreprocessTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.training_task import CFPBTrainingTask


class CFPBComplaintsService:
    def __init__(
        self,
        download_task: CFPBDataDownloadTask,
        ingest_task: CFPBIngestionTask,
        preprocess_task: CFPBPreprocessTask,
        dataset_task: CFPBDatasetTask,
        training_task: CFPBTrainingTask,
    ) -> None:
        self._download_task   = download_task
        self._ingest_task     = ingest_task
        self._preprocess_task = preprocess_task
        self._dataset_task    = dataset_task
        self._training_task   = training_task
        self._logger = StructuredLogger(name=__name__)
        self._session_parquet_path: str = ""
        self._session_df:           pd.DataFrame | None = None
        self._label_map:            dict[str, int] = {}
        self._train_loader:         DataLoader[Any] | None = None
        self._val_loader:           DataLoader[Any] | None = None

    async def download(self, request: CFPBDownloadRequest) -> CFPBDownloadResponse:
        return await asyncio.to_thread(self._download_task.execute, request)

    async def ingest(self, request: CFPBIngestionRequest) -> CFPBIngestionResponse:
        resp = await asyncio.to_thread(self._ingest_task.execute, request)
        if resp.status == "success":
            self._session_parquet_path = resp.parquet_path
            self._session_df = await asyncio.to_thread(pd.read_parquet, resp.parquet_path)
            self._label_map = {}
            self._train_loader = None
            self._val_loader = None
        return resp

    async def preprocess(self, request: CFPBPreprocessRequest) -> CFPBPreprocessResponse:
        if not self._session_parquet_path:
            return CFPBPreprocessResponse(
                execution_id=request.execution_id, rows_after_filter=0,
                n_classes=0, class_weights={}, status="failed",
                error="No data. Call /admin/foundations/cfpb/ingest first.",
            )
        resp, df, label_map = await asyncio.to_thread(
            lambda: self._preprocess_task.execute(request, self._session_parquet_path)
        )
        if resp.status == "success":
            self._session_df = df
            self._label_map = label_map
        return resp

    async def build_dataloaders(self, request: CFPBDatasetRequest) -> CFPBDatasetResponse:
        if self._session_df is None or not self._label_map:
            return CFPBDatasetResponse(
                execution_id=request.execution_id, train_samples=0, val_samples=0,
                train_batches=0, val_batches=0, status="failed",
                error="No preprocessed data. Call /admin/foundations/cfpb/preprocess first.",
            )
        from datasets import Dataset as HFDataset
        hf_ds = await asyncio.to_thread(HFDataset.from_pandas, self._session_df)
        resp, train_loader, val_loader = await asyncio.to_thread(
            lambda: self._dataset_task.execute(request, hf_ds, self._label_map)
        )
        if resp.status == "success":
            self._train_loader = train_loader
            self._val_loader   = val_loader
        return resp

    async def train(self, request: CFPBTrainRequest) -> CFPBTrainResponse:
        if self._train_loader is None or self._val_loader is None:
            return CFPBTrainResponse(
                execution_id=request.execution_id, epochs_completed=0,
                final_train_loss=0.0, final_val_loss=0.0, final_val_accuracy=0.0,
                checkpoint_path="", status="failed",
                error="No DataLoaders. Call /admin/foundations/cfpb/dataloaders first.",
            )
        from transformers import AutoModelForSequenceClassification
        model = await asyncio.to_thread(
            AutoModelForSequenceClassification.from_pretrained,
            request.model_name,
            num_labels=len(self._label_map),
        )
        return await asyncio.to_thread(
            lambda: self._training_task.execute(request, model, self._train_loader, self._val_loader)
        )

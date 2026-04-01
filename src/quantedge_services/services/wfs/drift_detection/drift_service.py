"""DriftDetectionService — async wrapper around drift detection tasks."""
from __future__ import annotations
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    DataDriftRequest, DataDriftResult,
    ConceptDriftRequest, ConceptDriftResult,
)
from quantedge_services.services.wfs.drift_detection.tasks.data_drift_task import DataDriftTask
from quantedge_services.services.wfs.drift_detection.tasks.concept_drift_task import ConceptDriftTask


class DriftDetectionService:
    """Orchestrates data drift and concept drift detection."""

    def __init__(self, data_drift_task: DataDriftTask, concept_drift_task: ConceptDriftTask) -> None:
        self._data_drift = data_drift_task
        self._concept_drift = concept_drift_task

    async def detect_data_drift(self, request: DataDriftRequest) -> DataDriftResult:
        return self._data_drift.execute(request)

    async def detect_concept_drift(self, request: ConceptDriftRequest) -> ConceptDriftResult:
        return self._concept_drift.execute(request)

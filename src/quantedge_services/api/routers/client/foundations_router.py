"""FoundationsClientRouter — client-facing async endpoints."""
from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status as http_status
from quantedge_services.api.middleware.nexus.auth import JWTAuthMiddleware
from quantedge_services.api.schemas.job_schemas import JobListResponse, JobSubmittedResponse, JobStatusResponse
from quantedge_services.api.schemas.foundations.cfpb_schemas import CFPBPredictRequest, CFPBPredictResponse
from quantedge_services.api.schemas.foundations.forex_schemas import ForexTensorOpsRequest
from quantedge_services.core.jobs import JobRegistry, JobStatus
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade

_auth = JWTAuthMiddleware()


class FoundationsClientRouter:
    def __init__(self, facade: FoundationsServiceFacade, registry: JobRegistry) -> None:
        self._facade   = facade
        self._registry = registry
        self.router = APIRouter(
            prefix="/client/foundations",
            tags=["Client — Foundations"],
            dependencies=[Depends(_auth)],
        )
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.get("/jobs",          response_model=JobListResponse)(self.list_jobs)
        self.router.get("/jobs/{job_id}", response_model=JobStatusResponse)(self.get_job_status)
        self.router.post("/forex/signals", response_model=JobSubmittedResponse, status_code=202)(self.get_forex_signals)
        self.router.post("/cfpb/predict",  response_model=CFPBPredictResponse)(self.predict_complaint_product)

    async def list_jobs(
        self,
        task_name: str | None = None,
        status: JobStatus | None = None,
    ) -> JobListResponse:
        records = self._registry.list_all(task_name=task_name, status=status)
        items = [
            JobStatusResponse(
                job_id=r.id, task_name=r.task_name, status=r.status,
                submitted_at=r.submitted_at, started_at=r.started_at,
                completed_at=r.completed_at, result=r.result, error=r.error,
            )
            for r in records
        ]
        return JobListResponse(jobs=items, total=len(items))

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        record = self._registry.get(job_id)
        if record is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
        return JobStatusResponse(
            job_id=record.id, task_name=record.task_name, status=record.status,
            submitted_at=record.submitted_at, started_at=record.started_at,
            completed_at=record.completed_at, result=record.result, error=record.error,
        )

    async def get_forex_signals(self, request: ForexTensorOpsRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_tensor_ops")
        bg.add_task(self._run_job, job.id, self._facade.forex_tensor_ops, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def _run_job(self, job_id: str, method, *args) -> None:
        self._registry.set_running(job_id)
        try:
            result = await method(*args)
            self._registry.set_success(job_id, result.model_dump())
        except Exception as exc:
            self._registry.set_failed(job_id, str(exc))

    async def predict_complaint_product(self, request: CFPBPredictRequest) -> CFPBPredictResponse:
        return CFPBPredictResponse(
            execution_id=request.execution_id,
            predicted_product="pending",
            confidence=0.0,
            status="failed",
            error="Inference endpoint requires a trained checkpoint (Week 2).",
        )

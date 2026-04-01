"""FoundationsAdminRouter — async 202 job-submission endpoints for Foundations workflows."""
from __future__ import annotations
from typing import Callable, Any
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status as http_status
from quantedge_services.api.middleware.nexus.auth import JWTAuthMiddleware
from quantedge_services.api.schemas.job_schemas import JobListResponse, JobSubmittedResponse, JobStatusResponse
from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBDatasetRequest, CFPBIngestionRequest, CFPBPreprocessRequest, CFPBTrainRequest,
    CFPBDownloadRequest,
)
from quantedge_services.api.schemas.foundations.forex_schemas import (
    ForexAutogradRequest, ForexDownloadRequest, ForexIngestionRequest,
    ForexPreprocessRequest, ForexTensorOpsRequest,
)
from quantedge_services.core.jobs import JobRegistry, JobStatus
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade

_auth = JWTAuthMiddleware()


class FoundationsAdminRouter:
    """Admin router — all POST endpoints submit background jobs (202 Accepted).
    
    Pipeline for Forex: download → ingest → preprocess → autograd → tensor-ops
    Pipeline for CFPB:  download → ingest → preprocess → dataloaders → train
    
    Poll GET /admin/foundations/jobs/{job_id} to check status and retrieve result.
    """

    def __init__(self, facade: FoundationsServiceFacade, registry: JobRegistry) -> None:
        self._facade   = facade
        self._registry = registry
        self.router = APIRouter(
            prefix="/admin/foundations",
            tags=["Admin — Foundations"],
            dependencies=[Depends(_auth)],
        )
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.get("/jobs",          response_model=JobListResponse)(self.list_jobs)
        self.router.get("/jobs/{job_id}", response_model=JobStatusResponse)(self.get_job_status)

        self.router.post("/forex/download",   response_model=JobSubmittedResponse, status_code=202)(self.forex_download)
        self.router.post("/forex/ingest",      response_model=JobSubmittedResponse, status_code=202)(self.forex_ingest)
        self.router.post("/forex/preprocess",  response_model=JobSubmittedResponse, status_code=202)(self.forex_preprocess)
        self.router.post("/forex/autograd",    response_model=JobSubmittedResponse, status_code=202)(self.forex_autograd)
        self.router.post("/forex/tensor-ops",  response_model=JobSubmittedResponse, status_code=202)(self.forex_tensor_ops)

        self.router.post("/cfpb/download",     response_model=JobSubmittedResponse, status_code=202)(self.cfpb_download)
        self.router.post("/cfpb/ingest",       response_model=JobSubmittedResponse, status_code=202)(self.cfpb_ingest)
        self.router.post("/cfpb/preprocess",   response_model=JobSubmittedResponse, status_code=202)(self.cfpb_preprocess)
        self.router.post("/cfpb/dataloaders",  response_model=JobSubmittedResponse, status_code=202)(self.cfpb_build_dataloaders)
        self.router.post("/cfpb/train",        response_model=JobSubmittedResponse, status_code=202)(self.cfpb_train)

    async def _run_job(self, job_id: str, method: Callable, *args: Any) -> None:
        """Runs an async service method, updates registry with result or error."""
        self._registry.set_running(job_id)
        try:
            result = await method(*args)
            self._registry.set_success(job_id, result.model_dump())
        except Exception as exc:
            self._registry.set_failed(job_id, str(exc))

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

    async def forex_download(self, request: ForexDownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_download")
        bg.add_task(self._run_job, job.id, self._facade.forex_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def forex_ingest(self, request: ForexIngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_ingest")
        bg.add_task(self._run_job, job.id, self._facade.forex_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def forex_preprocess(self, request: ForexPreprocessRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_preprocess")
        bg.add_task(self._run_job, job.id, self._facade.forex_preprocess, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def forex_autograd(self, request: ForexAutogradRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_autograd")
        bg.add_task(self._run_job, job.id, self._facade.forex_autograd, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def forex_tensor_ops(self, request: ForexTensorOpsRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("forex_tensor_ops")
        bg.add_task(self._run_job, job.id, self._facade.forex_tensor_ops, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cfpb_download(self, request: CFPBDownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cfpb_download")
        bg.add_task(self._run_job, job.id, self._facade.cfpb_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cfpb_ingest(self, request: CFPBIngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cfpb_ingest")
        bg.add_task(self._run_job, job.id, self._facade.cfpb_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cfpb_preprocess(self, request: CFPBPreprocessRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cfpb_preprocess")
        bg.add_task(self._run_job, job.id, self._facade.cfpb_preprocess, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cfpb_build_dataloaders(self, request: CFPBDatasetRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cfpb_dataloaders")
        bg.add_task(self._run_job, job.id, self._facade.cfpb_build_dataloaders, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cfpb_train(self, request: CFPBTrainRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cfpb_train")
        bg.add_task(self._run_job, job.id, self._facade.cfpb_train, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

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
from quantedge_services.api.schemas.foundations.nn_schemas import (
    NNTrainRequest, NNEvalRequest, NNPredictRequest,
)
from quantedge_services.api.schemas.foundations.profiler_schemas import (
    CICIoTDownloadRequest, CICIoTIngestionRequest,
    DataloaderTuneRequest, MemorySummaryRequest, ProfilerRunRequest,
)
from quantedge_services.api.schemas.foundations.attention_schemas import (
    CMAPSSDownloadRequest, CMAPSSIngestionRequest,
    AttentionTrainRequest, AttentionEvalRequest, AttentionPredictRequest,
)
from quantedge_services.api.schemas.foundations.viz_schemas import (
    AttentionExtractRequest, AttentionHeatmapRequest, ArchDecisionRequest,
)
from quantedge_services.api.schemas.foundations.lora_schemas import (
    OAsst1DownloadRequest, OAsst1IngestionRequest,
    LoRATrainRequest, LoRAEvalRequest, LoRAPredictRequest, LoRAMergeRequest,
)
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest, StackOverflowIngestionRequest,
    DomainAdaptTrainRequest, DomainAdaptEvalRequest,
    OllamaInferRequest, OllamaMergeRequest,
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

        self.router.post("/nn/train",    response_model=JobSubmittedResponse, status_code=202)(self.nn_train)
        self.router.post("/nn/evaluate", response_model=JobSubmittedResponse, status_code=202)(self.nn_evaluate)
        self.router.post("/nn/predict",  response_model=JobSubmittedResponse, status_code=202)(self.nn_predict)

        self.router.post("/cic-iot/download",         response_model=JobSubmittedResponse, status_code=202)(self.cic_iot_download)
        self.router.post("/cic-iot/ingest",           response_model=JobSubmittedResponse, status_code=202)(self.cic_iot_ingest)
        self.router.post("/profiler/run",             response_model=JobSubmittedResponse, status_code=202)(self.profiler_run)
        self.router.post("/profiler/memory",          response_model=JobSubmittedResponse, status_code=202)(self.profiler_memory)
        self.router.post("/profiler/tune-dataloader", response_model=JobSubmittedResponse, status_code=202)(self.profiler_tune_dataloader)

        self.router.post("/cmapss/download",    response_model=JobSubmittedResponse, status_code=202)(self.cmapss_download)
        self.router.post("/cmapss/ingest",      response_model=JobSubmittedResponse, status_code=202)(self.cmapss_ingest)
        self.router.post("/attention/train",    response_model=JobSubmittedResponse, status_code=202)(self.attention_train)
        self.router.post("/attention/evaluate", response_model=JobSubmittedResponse, status_code=202)(self.attention_evaluate)
        self.router.post("/attention/predict",  response_model=JobSubmittedResponse, status_code=202)(self.attention_predict)
        self.router.post("/attention/extract-weights", response_model=JobSubmittedResponse, status_code=202)(self.attention_extract_weights)
        self.router.post("/attention/heatmap", response_model=JobSubmittedResponse, status_code=202)(self.attention_heatmap)
        self.router.post("/attention/arch-decision", response_model=JobSubmittedResponse, status_code=202)(self.attention_arch_decision)

        self.router.post("/oasst1/download",  response_model=JobSubmittedResponse, status_code=202)(self.oasst1_download)
        self.router.post("/oasst1/ingest",    response_model=JobSubmittedResponse, status_code=202)(self.oasst1_ingest)
        self.router.post("/lora/train",       response_model=JobSubmittedResponse, status_code=202)(self.lora_train)
        self.router.post("/lora/evaluate",    response_model=JobSubmittedResponse, status_code=202)(self.lora_evaluate)
        self.router.post("/lora/predict",     response_model=JobSubmittedResponse, status_code=202)(self.lora_predict)
        self.router.post("/lora/merge",       response_model=JobSubmittedResponse, status_code=202)(self.lora_merge)

        self.router.post("/stackoverflow/download", response_model=JobSubmittedResponse, status_code=202)(self.stackoverflow_download)
        self.router.post("/stackoverflow/ingest",   response_model=JobSubmittedResponse, status_code=202)(self.stackoverflow_ingest)
        self.router.post("/domain-adapt/train",     response_model=JobSubmittedResponse, status_code=202)(self.domain_adapt_train)
        self.router.post("/domain-adapt/evaluate",  response_model=JobSubmittedResponse, status_code=202)(self.domain_adapt_eval)
        self.router.post("/ollama/infer",            response_model=JobSubmittedResponse, status_code=202)(self.ollama_infer)
        self.router.post("/ollama/merge",            response_model=JobSubmittedResponse, status_code=202)(self.ollama_merge)

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

    async def nn_train(self, request: NNTrainRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("nn_train")
        bg.add_task(self._run_job, job.id, self._facade.nn_train, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def nn_evaluate(self, request: NNEvalRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("nn_evaluate")
        bg.add_task(self._run_job, job.id, self._facade.nn_evaluate, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def nn_predict(self, request: NNPredictRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("nn_predict")
        bg.add_task(self._run_job, job.id, self._facade.nn_predict, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cic_iot_download(self, request: CICIoTDownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cic_iot_download")
        bg.add_task(self._run_job, job.id, self._facade.cic_iot_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cic_iot_ingest(self, request: CICIoTIngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cic_iot_ingest")
        bg.add_task(self._run_job, job.id, self._facade.cic_iot_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def profiler_run(self, request: ProfilerRunRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("profiler_run")
        bg.add_task(self._run_job, job.id, self._facade.profiler_run, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def profiler_memory(self, request: MemorySummaryRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("profiler_memory")
        bg.add_task(self._run_job, job.id, self._facade.memory_summary, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def profiler_tune_dataloader(self, request: DataloaderTuneRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("profiler_tune_dataloader")
        bg.add_task(self._run_job, job.id, self._facade.tune_dataloader, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cmapss_download(self, request: CMAPSSDownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cmapss_download")
        bg.add_task(self._run_job, job.id, self._facade.cmapss_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def cmapss_ingest(self, request: CMAPSSIngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("cmapss_ingest")
        bg.add_task(self._run_job, job.id, self._facade.cmapss_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_train(self, request: AttentionTrainRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_train")
        bg.add_task(self._run_job, job.id, self._facade.attention_train, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_evaluate(self, request: AttentionEvalRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_evaluate")
        bg.add_task(self._run_job, job.id, self._facade.attention_evaluate, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_predict(self, request: AttentionPredictRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_predict")
        bg.add_task(self._run_job, job.id, self._facade.attention_predict, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_extract_weights(self, request: AttentionExtractRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_extract_weights")
        bg.add_task(self._run_job, job.id, self._facade.attention_extract, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_heatmap(self, request: AttentionHeatmapRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_heatmap")
        bg.add_task(self._run_job, job.id, self._facade.attention_heatmap, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def attention_arch_decision(self, request: ArchDecisionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("attention_arch_decision")
        bg.add_task(self._run_job, job.id, self._facade.architecture_decision, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def oasst1_download(self, request: OAsst1DownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("oasst1_download")
        bg.add_task(self._run_job, job.id, self._facade.oasst1_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def oasst1_ingest(self, request: OAsst1IngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("oasst1_ingest")
        bg.add_task(self._run_job, job.id, self._facade.oasst1_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def lora_train(self, request: LoRATrainRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("lora_train")
        bg.add_task(self._run_job, job.id, self._facade.lora_train, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def lora_evaluate(self, request: LoRAEvalRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("lora_evaluate")
        bg.add_task(self._run_job, job.id, self._facade.lora_evaluate, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def lora_predict(self, request: LoRAPredictRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("lora_predict")
        bg.add_task(self._run_job, job.id, self._facade.lora_predict, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def lora_merge(self, request: LoRAMergeRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("lora_merge")
        bg.add_task(self._run_job, job.id, self._facade.lora_merge, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def stackoverflow_download(self, request: StackOverflowDownloadRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("stackoverflow_download")
        bg.add_task(self._run_job, job.id, self._facade.submit_stackoverflow_download, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def stackoverflow_ingest(self, request: StackOverflowIngestionRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("stackoverflow_ingest")
        bg.add_task(self._run_job, job.id, self._facade.submit_stackoverflow_ingest, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def domain_adapt_train(self, request: DomainAdaptTrainRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("domain_adapt_train")
        bg.add_task(self._run_job, job.id, self._facade.submit_domain_adapt_train, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def domain_adapt_eval(self, request: DomainAdaptEvalRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("domain_adapt_eval")
        bg.add_task(self._run_job, job.id, self._facade.submit_domain_adapt_eval, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def ollama_infer(self, request: OllamaInferRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("ollama_infer")
        bg.add_task(self._run_job, job.id, self._facade.submit_ollama_infer, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

    async def ollama_merge(self, request: OllamaMergeRequest, bg: BackgroundTasks) -> JobSubmittedResponse:
        job = self._registry.create("ollama_merge")
        bg.add_task(self._run_job, job.id, self._facade.submit_ollama_merge, request)
        return JobSubmittedResponse(job_id=job.id, task_name=job.task_name)

import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

// Forex admin DTOs
export interface ForexIngestionRequest { data_dir?: string; years?: number[]; nrows?: number | null; }
export interface ForexIngestionResponse { execution_id: string; rows_loaded: number; files_loaded: number; status: string; error?: string | null; }
export interface ForexPreprocessRequest { execution_id: string; fill_gaps?: boolean; normalize?: boolean; }
export interface ForexPreprocessResponse { execution_id: string; rows_after: number; nan_filled: number; status: string; error?: string | null; }
export interface ForexAutogradRequest { execution_id: string; window_size?: number; }
export interface ForexAutogradResponse { execution_id: string; manual_loss: number; autograd_loss: number; max_grad_diff: number; status: string; error?: string | null; }

// CFPB admin DTOs
export interface CFPBIngestionRequest { cache_dir?: string; split?: string; streaming?: boolean; }
export interface CFPBIngestionResponse { execution_id: string; rows_loaded: number; status: string; error?: string | null; }
export interface CFPBPreprocessRequest { execution_id: string; model_name?: string; max_length?: number; batch_size?: number; }
export interface CFPBPreprocessResponse { execution_id: string; rows_after_filter: number; n_classes: number; class_weights: Record<string, number>; status: string; error?: string | null; }
export interface CFPBDatasetRequest { execution_id: string; batch_size?: number; num_workers?: number; pin_memory?: boolean; val_split?: number; }
export interface CFPBDatasetResponse { execution_id: string; train_samples: number; val_samples: number; train_batches: number; val_batches: number; status: string; error?: string | null; }
export interface CFPBTrainRequest { execution_id: string; model_name?: string; epochs?: number; learning_rate?: number; seed?: number; checkpoint_dir?: string; resume_from?: string | null; }
export interface CFPBTrainResponse { execution_id: string; epochs_completed: number; final_train_loss: number; final_val_loss: number; final_val_accuracy: number; checkpoint_path: string; status: string; error?: string | null; }

// Job response types
export interface JobSubmittedResponse { job_id: string; task_name: string; status: string; message: string; }
export interface JobStatusResponse { job_id: string; task_name: string; status: string; result: any; error: string | null; created_at: string; updated_at: string; }
export interface JobListResponse { jobs: JobStatusResponse[]; total: number; }

// Forex extended (202 pattern)
export interface ForexDownloadRequest { output_dir?: string; }
export interface ForexTensorOpsRequest { execution_id: string; volatility_window?: number; momentum_window?: number; inject_nan_fraction?: number; }

// Week 3 — Neural Networks
export interface NNTrainRequest { execution_id: string; model_type?: string; epochs?: number; learning_rate?: number; }
export interface NNEvalRequest { execution_id: string; model_type?: string; checkpoint_path?: string; }
export interface NNPredictRequest { execution_id: string; model_type?: string; checkpoint_path?: string; }

// Week 4 — Profiling
export interface ProfilingRequest { execution_id: string; num_batches?: number; }
export interface CICIoTDownloadRequest { dataset_source?: string; output_dir?: string; }

// Week 5-6 — Attention + Viz
export interface AttentionTrainRequest { execution_id: string; epochs?: number; }
export interface AttentionExtractRequest { execution_id: string; checkpoint_path?: string; }
export interface AttentionHeatmapRequest { execution_id: string; layer_index?: number; }
export interface ArchDecisionRequest { task_type: string; input_modality?: string; }

// Week 7 — LoRA
export interface LoRATrainRequest { execution_id: string; lora_rank?: number; lora_alpha?: number; epochs?: number; }
export interface LoRAEvalRequest { execution_id: string; checkpoint_path?: string; }
export interface LoRAPredictRequest { execution_id: string; checkpoint_path?: string; }
export interface LoRAMergeRequest { execution_id: string; checkpoint_path?: string; output_dir?: string; }

// Week 8 — Domain Adapt + Ollama
export interface DomainAdaptTrainRequest { data_path?: string; model_name?: string; output_dir?: string; lora_rank?: number; max_steps?: number; }
export interface DomainAdaptEvalRequest { data_path?: string; checkpoint_path?: string; model_name?: string; }
export interface OllamaInferRequest { model_name?: string; prompt: string; max_tokens?: number; temperature?: number; ollama_base_url?: string; }
export interface OllamaMergeRequest { adapter_checkpoint_path?: string; base_model_name?: string; output_dir?: string; }

// Week 9 — Export
export interface TorchScriptExportRequest { output_dir?: string; export_mode?: string; }
export interface ONNXExportRequest { output_dir?: string; opset_version?: number; dynamic_batch?: boolean; }
export interface ONNXValidateRequest { onnx_path: string; }
export interface TensorRTExportRequest { torchscript_path: string; output_dir?: string; precision?: string; }
export interface BenchmarkRequest { eager_checkpoint_path?: string; torchscript_path?: string; onnx_path?: string; num_runs?: number; }

// Week 10 — Quantization + Serving
export interface QuantizeStaticRequest { output_dir?: string; calibration_batches?: number; }
export interface QuantizeDynamicRequest { output_dir?: string; }
export interface QuantizeQATRequest { output_dir?: string; train_steps?: number; }
export interface QuantizeCompareRequest { output_dir?: string; num_runs?: number; }
export interface ModelInferRequest { model_path?: string; model_format?: string; }
export interface ServingBenchmarkRequest { model_path?: string; model_format?: string; num_requests?: number; }

// Week 11 — Tracking + Canary
export interface MLflowLogRequest { experiment_name: string; run_name: string; params?: Record<string,string>; metrics?: Record<string,number>; }
export interface MLflowRegisterRequest { run_id: string; model_name: string; stage?: string; }
export interface CanaryDeployRequest { deployment_id: string; baseline_model_path?: string; candidate_model_path?: string; canary_traffic_pct?: number; }
export interface CanaryEvalRequest { deployment_id: string; num_eval_requests?: number; }
export interface ModelRegistryListRequest { filter_name?: string; max_results?: number; }
export interface ModelRegistryPromoteRequest { model_name: string; version: string; target_stage: string; }

// Week 12 — Drift + Monitoring + ADR
export interface DataDriftRequest { reference_data_path?: string; current_data_path?: string; psi_threshold?: number; }
export interface ConceptDriftRequest { predictions_log_path?: string; window_size?: number; drift_threshold?: number; }
export interface PrometheusMetricsRequest { output_path?: string; }
export interface AuditLogRequest { event_type: string; actor: string; resource: string; severity?: string; }
export interface ADRGenerateRequest { output_dir?: string; }
export interface ADRListRequest { adr_dir?: string; }

/** QuantEdgeAdminApiService — typed HTTP client for all admin endpoints. */
@Injectable({ providedIn: 'root' })
export class QuantEdgeAdminApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  // Forex
  forexDownload(req: ForexDownloadRequest): Observable<JobSubmittedResponse> {
    return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/forex/download`, req);
  }
  forexIngest(req: ForexIngestionRequest): Observable<ForexIngestionResponse> {
    return this.http.post<ForexIngestionResponse>(`${this.base}/admin/foundations/forex/ingest`, req);
  }
  forexPreprocess(req: ForexPreprocessRequest): Observable<ForexPreprocessResponse> {
    return this.http.post<ForexPreprocessResponse>(`${this.base}/admin/foundations/forex/preprocess`, req);
  }
  forexAutograd(req: ForexAutogradRequest): Observable<ForexAutogradResponse> {
    return this.http.post<ForexAutogradResponse>(`${this.base}/admin/foundations/forex/autograd`, req);
  }
  forexTensorOps(req: ForexTensorOpsRequest): Observable<JobSubmittedResponse> {
    return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/forex/tensor-ops`, req);
  }

  // CFPB
  cfpbIngest(req: CFPBIngestionRequest): Observable<CFPBIngestionResponse> {
    return this.http.post<CFPBIngestionResponse>(`${this.base}/admin/foundations/cfpb/ingest`, req);
  }
  cfpbPreprocess(req: CFPBPreprocessRequest): Observable<CFPBPreprocessResponse> {
    return this.http.post<CFPBPreprocessResponse>(`${this.base}/admin/foundations/cfpb/preprocess`, req);
  }
  cfpbBuildDataloaders(req: CFPBDatasetRequest): Observable<CFPBDatasetResponse> {
    return this.http.post<CFPBDatasetResponse>(`${this.base}/admin/foundations/cfpb/dataloaders`, req);
  }
  cfpbTrain(req: CFPBTrainRequest): Observable<CFPBTrainResponse> {
    return this.http.post<CFPBTrainResponse>(`${this.base}/admin/foundations/cfpb/train`, req);
  }

  // Jobs
  listJobs(taskName?: string, status?: string): Observable<JobListResponse> {
    let params = '';
    if (taskName) params += `?task_name=${taskName}`;
    if (status) params += `${params ? '&' : '?'}status=${status}`;
    return this.http.get<JobListResponse>(`${this.base}/admin/foundations/jobs${params}`);
  }
  getJob(jobId: string): Observable<JobStatusResponse> {
    return this.http.get<JobStatusResponse>(`${this.base}/admin/foundations/jobs/${jobId}`);
  }

  // Week 3 Neural Nets
  nnTrain(req: NNTrainRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/nn/train`, req); }
  nnEval(req: NNEvalRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/nn/evaluate`, req); }
  nnPredict(req: NNPredictRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/nn/predict`, req); }

  // Week 4 Profiling
  runProfiling(req: ProfilingRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/profiling/run`, req); }
  cicIotDownload(req: CICIoTDownloadRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/cic-iot/download`, req); }

  // Week 5-6 Attention
  attentionTrain(req: AttentionTrainRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/attention/train`, req); }
  attentionExtract(req: AttentionExtractRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/attention/extract`, req); }
  attentionHeatmap(req: AttentionHeatmapRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/attention/heatmap`, req); }
  archDecision(req: ArchDecisionRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/attention/arch-decision`, req); }

  // Week 7 LoRA
  loraTrain(req: LoRATrainRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/lora/train`, req); }
  loraEval(req: LoRAEvalRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/lora/evaluate`, req); }
  loraPredict(req: LoRAPredictRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/lora/predict`, req); }
  loraMerge(req: LoRAMergeRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/lora/merge`, req); }

  // Week 8 Domain Adapt + Ollama
  domainAdaptTrain(req: DomainAdaptTrainRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/domain-adapt/train`, req); }
  domainAdaptEval(req: DomainAdaptEvalRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/domain-adapt/evaluate`, req); }
  ollamaInfer(req: OllamaInferRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/ollama/infer`, req); }
  ollamaMerge(req: OllamaMergeRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/ollama/merge`, req); }

  // Week 9 Export
  exportTorchScript(req: TorchScriptExportRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/export/torchscript`, req); }
  exportOnnx(req: ONNXExportRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/export/onnx`, req); }
  validateOnnx(req: ONNXValidateRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/export/onnx/validate`, req); }
  exportTensorRT(req: TensorRTExportRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/export/tensorrt`, req); }
  benchmark(req: BenchmarkRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/export/benchmark`, req); }

  // Week 10 Quantization + Serving
  quantizeStatic(req: QuantizeStaticRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/quantize/static`, req); }
  quantizeDynamic(req: QuantizeDynamicRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/quantize/dynamic`, req); }
  quantizeQat(req: QuantizeQATRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/quantize/qat`, req); }
  quantizeCompare(req: QuantizeCompareRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/quantize/compare`, req); }
  modelInfer(req: ModelInferRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/serving/infer`, req); }
  servingBenchmark(req: ServingBenchmarkRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/serving/benchmark`, req); }

  // Week 11 Tracking + Canary
  mlflowLog(req: MLflowLogRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/mlflow/log`, req); }
  mlflowRegister(req: MLflowRegisterRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/mlflow/register`, req); }
  canaryDeploy(req: CanaryDeployRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/canary/deploy`, req); }
  canaryEval(req: CanaryEvalRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/canary/evaluate`, req); }
  registryList(req: ModelRegistryListRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/registry/list`, req); }
  registryPromote(req: ModelRegistryPromoteRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/registry/promote`, req); }

  // Week 12 Drift + Monitoring + ADR
  dataDrift(req: DataDriftRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/drift/data`, req); }
  conceptDrift(req: ConceptDriftRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/drift/concept`, req); }
  prometheusMetrics(req: PrometheusMetricsRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/monitoring/metrics`, req); }
  auditLog(req: AuditLogRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/monitoring/audit`, req); }
  adrGenerate(req: ADRGenerateRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/adr/generate`, req); }
  adrList(req: ADRListRequest): Observable<JobSubmittedResponse> { return this.http.post<JobSubmittedResponse>(`${this.base}/admin/foundations/adr/list`, req); }
}

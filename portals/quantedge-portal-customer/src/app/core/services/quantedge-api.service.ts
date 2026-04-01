import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  ForexIngestionRequest, ForexIngestionResponse,
  ForexPreprocessRequest, ForexPreprocessResponse,
  ForexAutogradRequest, ForexAutogradResponse,
  ForexTensorOpsRequest, ForexTensorOpsResponse,
} from '../models/forex.models';
import { CFPBPredictRequest, CFPBPredictResponse } from '../models/cfpb.models';

// Shared job types
export interface JobSubmittedResponse { job_id: string; task_name: string; status: string; message: string; }
export interface JobStatusResponse { job_id: string; task_name: string; status: string; result: any; error: string | null; created_at: string; updated_at: string; }

// Model inference
export interface ModelInferRequest { model_path?: string; model_format?: string; input_size?: number; seq_len?: number; }
export interface OllamaInferRequest { model_name?: string; prompt: string; max_tokens?: number; temperature?: number; ollama_base_url?: string; }

// Drift status (read-only for customer)
export interface DataDriftRequest { reference_data_path?: string; current_data_path?: string; }

/** QuantEdgeApiService — typed HTTP client wrapping all backend endpoints.
 *
 * All methods accept a DTO and return an Observable<DTO>.
 * No loose parameters — consistent with the backend contract.
 */
@Injectable({ providedIn: 'root' })
export class QuantEdgeApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  // ── Client — Forex ──────────────────────────────────────────────────────

  getForexSignals(request: ForexTensorOpsRequest): Observable<ForexTensorOpsResponse> {
    return this.http.post<ForexTensorOpsResponse>(
      `${this.base}/client/foundations/forex/signals`, request,
    );
  }

  // ── Client — CFPB ───────────────────────────────────────────────────────

  predictComplaintProduct(request: CFPBPredictRequest): Observable<CFPBPredictResponse> {
    return this.http.post<CFPBPredictResponse>(
      `${this.base}/client/foundations/cfpb/predict`, request,
    );
  }

  // ── Jobs ─────────────────────────────────────────────────────────────────

  listJobs(taskName?: string, status?: string): Observable<any> {
    const params = new URLSearchParams();
    if (taskName) params.set('task_name', taskName);
    if (status) params.set('status', status);
    const qs = params.toString() ? '?' + params.toString() : '';
    return this.http.get(`${this.base}/client/foundations/jobs${qs}`);
  }

  getJob(jobId: string): Observable<JobStatusResponse> {
    return this.http.get<JobStatusResponse>(`${this.base}/client/foundations/jobs/${jobId}`);
  }

  // ── Model inference (customer-facing) ───────────────────────────────────

  requestInference(req: ModelInferRequest): Observable<JobSubmittedResponse> {
    return this.http.post<JobSubmittedResponse>(`${this.base}/client/foundations/serving/infer`, req);
  }

  // ── Ollama chat ──────────────────────────────────────────────────────────

  ollamaChat(req: OllamaInferRequest): Observable<JobSubmittedResponse> {
    return this.http.post<JobSubmittedResponse>(`${this.base}/client/foundations/ollama/infer`, req);
  }
}

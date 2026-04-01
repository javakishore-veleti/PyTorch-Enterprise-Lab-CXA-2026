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

/** QuantEdgeAdminApiService — typed HTTP client for all admin endpoints. */
@Injectable({ providedIn: 'root' })
export class QuantEdgeAdminApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  // Forex
  forexIngest(req: ForexIngestionRequest): Observable<ForexIngestionResponse> {
    return this.http.post<ForexIngestionResponse>(`${this.base}/admin/foundations/forex/ingest`, req);
  }
  forexPreprocess(req: ForexPreprocessRequest): Observable<ForexPreprocessResponse> {
    return this.http.post<ForexPreprocessResponse>(`${this.base}/admin/foundations/forex/preprocess`, req);
  }
  forexAutograd(req: ForexAutogradRequest): Observable<ForexAutogradResponse> {
    return this.http.post<ForexAutogradResponse>(`${this.base}/admin/foundations/forex/autograd`, req);
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
}

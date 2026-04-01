/** DTOs mirroring Python Pydantic schemas in quantedge_services/api/schemas/foundations/ */

export interface ForexIngestionRequest {
  data_dir?: string;
  years?: number[];
  nrows?: number | null;
}

export interface ForexIngestionResponse {
  execution_id: string;
  rows_loaded: number;
  files_loaded: number;
  status: 'success' | 'failed';
  error?: string | null;
}

export interface ForexPreprocessRequest {
  execution_id: string;
  fill_gaps?: boolean;
  normalize?: boolean;
}

export interface ForexPreprocessResponse {
  execution_id: string;
  rows_after: number;
  nan_filled: number;
  status: 'success' | 'failed';
  error?: string | null;
}

export interface ForexAutogradRequest {
  execution_id: string;
  window_size?: number;
}

export interface ForexAutogradResponse {
  execution_id: string;
  manual_loss: number;
  autograd_loss: number;
  max_grad_diff: number;
  status: 'success' | 'failed';
  error?: string | null;
}

export interface ForexTensorOpsRequest {
  execution_id: string;
  volatility_window?: number;
  momentum_window?: number;
  inject_nan_fraction?: number;
}

export interface ForexTensorOpsResponse {
  execution_id: string;
  volatility_points: number;
  momentum_points: number;
  nan_injected: number;
  nan_remaining: number;
  status: 'success' | 'failed';
  error?: string | null;
}

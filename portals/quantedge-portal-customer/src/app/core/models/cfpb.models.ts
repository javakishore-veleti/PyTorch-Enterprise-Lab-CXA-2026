/** DTOs mirroring Python Pydantic schemas — CFPB Complaints */

export interface CFPBIngestionRequest {
  cache_dir?: string;
  split?: string;
  streaming?: boolean;
}

export interface CFPBIngestionResponse {
  execution_id: string;
  rows_loaded: number;
  status: 'success' | 'failed';
  error?: string | null;
}

export interface CFPBPredictRequest {
  execution_id: string;
  text: string;
  checkpoint_path: string;
}

export interface CFPBPredictResponse {
  execution_id: string;
  predicted_product: string;
  confidence: number;
  status: 'success' | 'failed';
  error?: string | null;
}

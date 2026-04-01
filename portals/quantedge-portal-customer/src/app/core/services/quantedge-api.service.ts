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
}

import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';
import { CFPBPredictRequest, CFPBPredictResponse } from '../../../core/models/cfpb.models';

@Injectable({ providedIn: 'root' })
export class ComplaintsAnalyticsService {
  private readonly api = inject(QuantEdgeApiService);

  predict(request: CFPBPredictRequest): Observable<CFPBPredictResponse> {
    return this.api.predictComplaintProduct(request);
  }
}

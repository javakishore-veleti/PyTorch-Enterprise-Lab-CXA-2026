import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';
import { ForexTensorOpsRequest } from '../../../core/models/forex.models';

export interface JobSubmittedResponse { job_id: string; task_name: string; }

/** ForexSignalsService — business logic for the Forex feature (async 202 job pattern). */
@Injectable({ providedIn: 'root' })
export class ForexSignalsService {
  private readonly api = inject(QuantEdgeApiService);

  getSignals(request: ForexTensorOpsRequest): Observable<JobSubmittedResponse> {
    return this.api.getForexSignals(request) as Observable<JobSubmittedResponse>;
  }
}

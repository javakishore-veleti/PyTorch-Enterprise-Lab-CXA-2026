import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';
import { ForexTensorOpsRequest, ForexTensorOpsResponse } from '../../../core/models/forex.models';

/** ForexSignalsService — business logic for the Forex feature. */
@Injectable({ providedIn: 'root' })
export class ForexSignalsService {
  private readonly api = inject(QuantEdgeApiService);

  getSignals(request: ForexTensorOpsRequest): Observable<ForexTensorOpsResponse> {
    return this.api.getForexSignals(request);
  }
}

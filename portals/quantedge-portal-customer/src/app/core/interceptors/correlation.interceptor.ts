import { Injectable } from '@angular/core';
import {
  HttpInterceptor, HttpRequest, HttpHandler, HttpEvent,
} from '@angular/common/http';
import { Observable } from 'rxjs';

const CORRELATION_HEADER = 'X-Correlation-ID';

/** Injects a per-request correlation ID — mirrors backend CorrelationIdMiddleware. */
@Injectable()
export class CorrelationInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    const correlationId = crypto.randomUUID();
    const cloned = req.clone({
      setHeaders: { [CORRELATION_HEADER]: correlationId },
    });
    return next.handle(cloned);
  }
}

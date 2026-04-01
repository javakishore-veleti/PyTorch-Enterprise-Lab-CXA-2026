import { Injectable } from '@angular/core';
import {
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor,
} from '@angular/common/http';
import { Observable } from 'rxjs';

/** Injects a X-Correlation-ID header mirroring the Python CorrelationIdMiddleware. */
@Injectable()
export class CorrelationInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    const correlationId = crypto.randomUUID();
    return next.handle(
      req.clone({ setHeaders: { 'X-Correlation-ID': correlationId } })
    );
  }
}

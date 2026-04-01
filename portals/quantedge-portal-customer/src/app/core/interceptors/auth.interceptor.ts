import { Injectable } from '@angular/core';
import {
  HttpInterceptor, HttpRequest, HttpHandler,
  HttpEvent, HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

/** Attaches the API key Bearer token to every outgoing request. */
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  /** API key read from sessionStorage (set on login). */
  private get apiKey(): string {
    return sessionStorage.getItem('quantedge_api_key') ?? '';
  }

  intercept(req: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    const authed = this.apiKey
      ? req.clone({ setHeaders: { Authorization: `Bearer ${this.apiKey}` } })
      : req;

    return next.handle(authed).pipe(
      catchError((err: HttpErrorResponse) => {
        if (err.status === 401) {
          sessionStorage.removeItem('quantedge_api_key');
          // Redirect to login — handled by router guard in a real app
        }
        return throwError(() => err);
      }),
    );
  }
}

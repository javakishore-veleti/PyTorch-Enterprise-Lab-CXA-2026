import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, switchMap, takeWhile, shareReplay } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface JobStatusResponse {
  job_id:       string;
  task_name:    string;
  status:       'pending' | 'running' | 'success' | 'failed';
  submitted_at: string;
  started_at?:  string;
  completed_at?: string;
  result?:      Record<string, unknown>;
  error?:       string;
}

export interface JobListResponse {
  jobs:  JobStatusResponse[];
  total: number;
}

/** Polls the QuantEdge job status API until a job reaches a terminal state. */
@Injectable({ providedIn: 'root' })
export class JobPollingService {
  private readonly jobsBase: string;

  constructor(private readonly http: HttpClient) {
    this.jobsBase = `${environment.apiUrl}/client/foundations/jobs`;
  }

  /** Poll every intervalMs until status is 'success' or 'failed', then emit final value. */
  pollUntilDone(jobId: string, intervalMs = 3000): Observable<JobStatusResponse> {
    return interval(intervalMs).pipe(
      switchMap(() => this.http.get<JobStatusResponse>(`${this.jobsBase}/${jobId}`)),
      takeWhile(r => r.status === 'pending' || r.status === 'running', /* inclusive */ true),
      shareReplay(1),
    );
  }

  getJobStatus(jobId: string): Observable<JobStatusResponse> {
    return this.http.get<JobStatusResponse>(`${this.jobsBase}/${jobId}`);
  }

  listJobs(taskName?: string, status?: string): Observable<JobListResponse> {
    const params: string[] = [];
    if (taskName) params.push(`task_name=${encodeURIComponent(taskName)}`);
    if (status)   params.push(`status=${encodeURIComponent(status)}`);
    const qs = params.length ? `?${params.join('&')}` : '';
    return this.http.get<JobListResponse>(`${this.jobsBase}${qs}`);
  }
}

import { Component, inject, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService, JobStatusResponse } from '../../../core/services/job-polling.service';

type PipelineStep = 'download' | 'ingest' | 'preprocess' | 'autograd' | 'tensor-ops';

interface StepState {
  job?: JobStatusResponse;
  loading: boolean;
}

/** TrainingJobsDashboardComponent — async 202 job-based Forex pipeline UI. */
@Component({
  selector: 'qe-admin-training-jobs',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="training-jobs">
      <h2>Training Jobs — Foundations</h2>

      <section class="pipeline">
        <h3>🟡 Forex EUR/USD Pipeline (Week 1)</h3>

        <div *ngFor="let step of pipelineSteps" class="step"
             [class.done]="steps[step]?.job?.status === 'success'"
             [class.running]="steps[step]?.job?.status === 'running' || steps[step]?.job?.status === 'pending'"
             [class.failed]="steps[step]?.job?.status === 'failed'">
          <span class="step-label">{{ stepLabels[step] }}</span>
          <button (click)="runStep(step)" [disabled]="steps[step]?.loading">
            {{ steps[step]?.loading ? '⏳ Queued…' : 'Run' }}
          </button>

          <span *ngIf="steps[step]?.job as job" class="job-badge" [attr.data-status]="job.status">
            {{ statusEmoji(job.status) }} {{ job.status | uppercase }}
            <span *ngIf="job.status === 'running'"> — polling every 3s…</span>
          </span>

          <pre *ngIf="steps[step]?.job?.result" class="result-json">{{ steps[step]?.job?.result | json }}</pre>
          <span *ngIf="steps[step]?.job?.error" class="error">{{ steps[step]?.job?.error }}</span>
        </div>
      </section>

      <!-- Live job list -->
      <section class="jobs-list">
        <h3>Recent Jobs <button (click)="refreshJobList()">↻ Refresh</button></h3>
        <table *ngIf="recentJobs.length">
          <thead><tr><th>Task</th><th>Status</th><th>Submitted</th><th>Duration</th></tr></thead>
          <tbody>
            <tr *ngFor="let job of recentJobs">
              <td>{{ job.task_name }}</td>
              <td [attr.data-status]="job.status">{{ statusEmoji(job.status) }} {{ job.status }}</td>
              <td>{{ job.submitted_at | date:'HH:mm:ss' }}</td>
              <td>{{ durationSec(job) }}s</td>
            </tr>
          </tbody>
        </table>
        <p *ngIf="!recentJobs.length" class="muted">No jobs yet. Run a pipeline step above.</p>
      </section>
    </div>
  `,
})
export class TrainingJobsDashboardComponent implements OnDestroy {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly subs    = new Subscription();

  readonly pipelineSteps: PipelineStep[] = ['download', 'ingest', 'preprocess', 'autograd', 'tensor-ops'];
  readonly stepLabels: Record<PipelineStep, string> = {
    'download':   '0. Download data',
    'ingest':     '1. Ingest → parquet',
    'preprocess': '2. Preprocess → tensor',
    'autograd':   '3. Autograd comparison',
    'tensor-ops': '4. Tensor ops (volatility / momentum)',
  };

  steps: Partial<Record<PipelineStep, StepState>> = {};
  recentJobs: JobStatusResponse[] = [];

  runStep(step: PipelineStep): void {
    this.steps[step] = { loading: true };
    const call$ = this._stepCall(step);
    this.subs.add(
      call$.subscribe({
        next: (submitted: { job_id: string; task_name: string }) => {
          this.steps[step] = { loading: false, job: { job_id: submitted.job_id, task_name: submitted.task_name, status: 'pending', submitted_at: new Date().toISOString() } };
          this._pollJob(step, submitted.job_id);
        },
        error: (e: Error) => { this.steps[step] = { loading: false, job: { job_id: '', task_name: step, status: 'failed', submitted_at: new Date().toISOString(), error: e.message } }; },
      })
    );
  }

  refreshJobList(): void {
    this.subs.add(this.polling.listJobs().subscribe(resp => { this.recentJobs = resp.jobs; }));
  }

  statusEmoji(status: string): string {
    return ({ pending: '⏳', running: '🔄', success: '✅', failed: '❌' } as Record<string, string>)[status] ?? '❓';
  }

  durationSec(job: JobStatusResponse): number {
    if (!job.completed_at || !job.started_at) return 0;
    return Math.round((new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000);
  }

  ngOnDestroy(): void { this.subs.unsubscribe(); }

  private _pollJob(step: PipelineStep, jobId: string): void {
    this.subs.add(
      this.polling.pollUntilDone(jobId).subscribe({
        next: job => { this.steps[step] = { loading: false, job }; },
        complete: () => this.refreshJobList(),
      })
    );
  }

  private _stepCall(step: PipelineStep) {
    switch (step) {
      case 'download':   return this.api.forexDownload({});
      case 'ingest':     return this.api.forexIngest({ data_dir: 'data/forex/raw', parquet_dir: 'data/forex/parquet', resample_freq: '1min', years: [2019, 2020, 2021, 2022, 2023] });
      case 'preprocess': return this.api.forexPreprocess({ execution_id: crypto.randomUUID() });
      case 'autograd':   return this.api.forexAutograd({ execution_id: crypto.randomUUID(), window_size: 20 });
      case 'tensor-ops': return this.api.forexTensorOps({ execution_id: crypto.randomUUID(), volatility_window: 20, momentum_window: 14, inject_nan_fraction: 0.01 });
    }
  }
}

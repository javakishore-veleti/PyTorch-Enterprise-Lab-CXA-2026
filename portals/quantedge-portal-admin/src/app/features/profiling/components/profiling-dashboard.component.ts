import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-profiling-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div class="qe-panel">
      <h2>⚡ Profiling (Week 4)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Execution ID</label>
          <input formControlName="execution_id" type="text" />
        </div>
        <div class="form-group">
          <label>Num Batches</label>
          <input formControlName="num_batches" type="number" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('profiling')">
            {{ loading() && currentAction()==='Run Profiling' ? 'Running…' : 'Run Profiling' }}
          </button>
          <button type="button" [disabled]="loading()" (click)="submit('ciciot')">
            {{ loading() && currentAction()==='Download CIC-IoT' ? 'Running…' : 'Download CIC-IoT Dataset' }}
          </button>
        </div>
      </form>
      @if (jobId()) {
        <div class="job-status">
          <span class="badge" [class]="statusClass()">{{ jobStatus() }}</span>
          <code>Job: {{ jobId() }}</code>
        </div>
      }
      @if (result()) {
        <pre class="result-box">{{ result() | json }}</pre>
      }
      @if (error()) {
        <div class="error-box">{{ error() }}</div>
      }
    </div>
  `,
})
export class ProfilingDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    execution_id: ['profiling-001'],
    num_batches:  [10],
  });

  loading       = signal(false);
  jobId         = signal('');
  jobStatus     = signal('');
  result        = signal<any>(null);
  error         = signal('');
  currentAction = signal('');

  statusClass = computed(() => {
    const s = this.jobStatus();
    return {
      'badge-completed': s === 'completed' || s === 'success',
      'badge-failed':    s === 'failed',
      'badge-running':   s === 'running' || s === 'pending',
      'badge-pending':   !s,
    };
  });

  submit(action: 'profiling' | 'ciciot'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;

    const actionLabel = action === 'profiling' ? 'Run Profiling' : 'Download CIC-IoT';
    this.currentAction.set(actionLabel);

    const call$ = action === 'profiling'
      ? this.api.runProfiling({ execution_id: v.execution_id!, num_batches: v.num_batches! })
      : this.api.cicIotDownload({});

    call$.subscribe({
      next: (submitted) => {
        this.jobId.set(submitted.job_id);
        this.jobStatus.set('pending');
        this.polling.pollUntilDone(submitted.job_id).subscribe({
          next:     (status) => { this.jobStatus.set(status.status); if (status.result) this.result.set(status.result); },
          error:    (err)    => { this.error.set(err.message); this.loading.set(false); },
          complete: ()       => this.loading.set(false),
        });
      },
      error: (err) => { this.error.set(err.message); this.loading.set(false); },
    });
  }
}

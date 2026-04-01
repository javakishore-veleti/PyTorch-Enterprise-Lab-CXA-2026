import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

/** Model Registry / Canary Deployment (Week 11). */
@Component({
  selector: 'qe-admin-model-registry',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>🚦 Canary Deployment</h1>
        <p>Probabilistic traffic splitting with auto promote/rollback · Week 11</p>
      </div>
    <div class="qe-panel">
      <h2>🚦 Canary Deployment (Week 11)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Deployment ID</label>
          <input formControlName="deployment_id" type="text" />
        </div>
        <div class="form-group">
          <label>Canary Traffic %</label>
          <input formControlName="canary_traffic_pct" type="number" min="0" max="100" />
        </div>
        <div class="form-group">
          <label>Eval Requests</label>
          <input formControlName="num_eval_requests" type="number" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('deploy')">Deploy Canary</button>
          <button type="button" [disabled]="loading()" (click)="submit('eval')">Evaluate Canary</button>
        </div>
      </form>
      @if (currentAction()) {
        <p style="color:#a0aec0;font-size:12px;margin:8px 0">Running: {{ currentAction() }}</p>
      }
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
export class ModelRegistryComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    deployment_id:      ['deploy-001'],
    canary_traffic_pct: [10],
    num_eval_requests:  [50],
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

  submit(action: 'deploy' | 'eval'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    this.currentAction.set(action === 'deploy' ? 'Deploy Canary' : 'Evaluate Canary');

    const call$ = action === 'deploy'
      ? this.api.canaryDeploy({ deployment_id: v.deployment_id!, canary_traffic_pct: v.canary_traffic_pct! })
      : this.api.canaryEval({ deployment_id: v.deployment_id!, num_eval_requests: v.num_eval_requests! });

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

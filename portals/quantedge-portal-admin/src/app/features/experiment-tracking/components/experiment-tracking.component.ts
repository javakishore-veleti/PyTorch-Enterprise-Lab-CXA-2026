import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

/** Experiment Tracking — MLflow log/register/list/promote (Week 11). */
@Component({
  selector: 'qe-admin-experiment-tracking',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>📊 MLflow Experiment Tracking</h1>
        <p>Log runs, register models, manage model registry · Week 11</p>
      </div>
    <div class="qe-panel">
      <h2>📊 MLflow Tracking (Week 11)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Experiment Name</label>
          <input formControlName="experiment_name" type="text" />
        </div>
        <div class="form-group">
          <label>Run Name</label>
          <input formControlName="run_name" type="text" />
        </div>
        <div class="form-group">
          <label>Params JSON (e.g. {"lr":"0.001"})</label>
          <textarea formControlName="params_json" rows="2"></textarea>
        </div>
        <div class="form-group">
          <label>Metrics JSON (e.g. {"accuracy":0.95})</label>
          <textarea formControlName="metrics_json" rows="2"></textarea>
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('log')">Log Run</button>
          <button type="button" [disabled]="loading()" (click)="submit('register')">Register Model</button>
          <button type="button" [disabled]="loading()" (click)="submit('list')">List Registry</button>
          <button type="button" [disabled]="loading()" (click)="submit('promote')">Promote Stage</button>
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
export class ExperimentTrackingComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    experiment_name: ['quantedge-run'],
    run_name:        ['run-001'],
    params_json:     ['{"lr": "0.001", "epochs": "5"}'],
    metrics_json:    ['{"accuracy": 0.95, "loss": 0.05}'],
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

  submit(action: 'log' | 'register' | 'list' | 'promote'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels = { log: 'Log Run', register: 'Register Model', list: 'List Registry', promote: 'Promote Stage' };
    this.currentAction.set(labels[action]);

    let params: Record<string,string> | undefined;
    let metrics: Record<string,number> | undefined;
    try { params  = v.params_json  ? JSON.parse(v.params_json)  : undefined; } catch { params = undefined; }
    try { metrics = v.metrics_json ? JSON.parse(v.metrics_json) : undefined; } catch { metrics = undefined; }

    const call$ = action === 'log'      ? this.api.mlflowLog({ experiment_name: v.experiment_name!, run_name: v.run_name!, params, metrics })
                : action === 'register' ? this.api.mlflowRegister({ run_id: 'latest', model_name: v.experiment_name! })
                : action === 'list'     ? this.api.registryList({ filter_name: v.experiment_name! })
                : this.api.registryPromote({ model_name: v.experiment_name!, version: '1', target_stage: 'Production' });

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
